import torch
import torch.nn as nn
import random
import tqdm

"""
representation:
    takes: environment observation
    returns: internal representation

model_dynamics:
    takes: internal representation
    takes: action
    returns: next internal representation
    returns: next reward

predictor:
    takes: internal representation
    returns: next action
    returns: next action value
"""


def construct_representation_network(observation_size, internal_representation_size):
    return nn.Sequential(
        nn.Linear(in_features=observation_size, out_features=internal_representation_size),
        nn.ReLU(),
    )


class RLDynamicsNetwork(nn.Module):
    def __init__(self, internal_representation_size, actions_size, reward_size, hidden_size=128):
        super().__init__()

        self.reward_size = reward_size
        self.internal_representation_size = internal_representation_size
        self.actions_size = actions_size

        self.model = nn.Sequential(
            nn.Linear(in_features=internal_representation_size + actions_size,
                      out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size,
                      out_features=internal_representation_size + reward_size),
            nn.Tanh(),
        )

    def forward(self, state, action):
        res = self.model(torch.concat([state, action]))
        return res[:self.internal_representation_size], res[-self.reward_size:]

    def forward_batch(self, state, actions):
        batches_list = [torch.concat([state, action]) for action in actions]
        batches = torch.stack(batches_list)
        return self.model(batches)


class RLPredictionNetwork(nn.Module):
    def __init__(self, internal_representation_size, actions_size, reward_size):
        super().__init__()
        self.reward_size = reward_size
        self.internal_representation_size = internal_representation_size
        self.actions_size = actions_size
        self.model = nn.Sequential(
            nn.Linear(in_features=internal_representation_size, out_features=actions_size + reward_size),
            nn.Tanh(),
        )

    def forward(self, state):
        res = self.model(state)
        return res[:self.actions_size], res[-self.reward_size:]


class RLReplayRecord:
    def __init__(self, episode, step, observed_state_before, executed_action, observed_state_after, observed_reward):
        self.episode=episode
        self.step=step
        self.observed_state_before = observed_state_before.detach()
        self.executed_action = executed_action.detach()
        self.observed_state_after = observed_state_after.detach()
        self.observed_reward = observed_reward.detach()


class RLRoutine(nn.Module):
    def __init__(self,
                 actions_size,
                 exploration_rate,
                 internal_representation_size,
                 observation_size,
                 reward_size,
                 search_depth,
                 game,
                 lr=0.001):
        super().__init__()
        assert(reward_size == 1)
        self.search_depth = search_depth
        self.internal_representation_size = internal_representation_size
        self.actions_size = actions_size
        self.reward_size = reward_size
        self.exploration_rate = exploration_rate
        self.representation = construct_representation_network(
            internal_representation_size=internal_representation_size,
            observation_size=observation_size,
        )  # takes `real env state`, returns
        self.dynamics = RLDynamicsNetwork(
            actions_size=actions_size,
            internal_representation_size=internal_representation_size,
            reward_size=reward_size,
        )
        self.prediction = RLPredictionNetwork(
            actions_size=actions_size,
            internal_representation_size=internal_representation_size,
            reward_size=reward_size,
        )
        self.game = game(observation_size=observation_size)
        self.replay_buffer: list[RLReplayRecord] = []
        self.loss_fn = nn.L1Loss()
        self.lr = lr
        self.optimizer = self.create_optimizer()

    def play_n_steps(self, n_steps=10, reset=False):
        if reset:
            self.game.reset()

        for step in range(n_steps):
            observed_state_before_action = self.game.state
            encoded_state = self.representation(observed_state_before_action)
            predicted_action, predicted_action_value = self.prediction(encoded_state)
            if random.random() < self.exploration_rate:
                predicted_action = torch.rand(self.actions_size)
            observed_state_after_action, observed_reward = self.game.run_action(predicted_action)

            self.replay_buffer = self.replay_buffer + [RLReplayRecord(episode=self.game.episode,
                                                                      step=step,
                                                                      observed_state_before=observed_state_before_action,
                                                                      executed_action=predicted_action,
                                                                      observed_state_after=observed_state_after_action,
                                                                      observed_reward=observed_reward)]

    def replay(self, n_steps=10, pbar=None):
        losses = []
        for rec_idx in random.sample(range(0, len(self.replay_buffer) - 1), n_steps):
            replay_record = self.replay_buffer[rec_idx]

            #
            # projecting observed game state into encoded `latent` representation
            #
            encoded_states = self.representation(torch.stack([
                replay_record.observed_state_before,
                replay_record.observed_state_after
            ]))
            encoded_state_before = encoded_states[0]
            encoded_state_after  = encoded_states[1]

            #
            # here we try to imagine ourselves in the situation getting:
            #   decided-action (what we'd do in that state) and decided-action-value (how much we expect to get)
            #
            decided_action, decided_action_value = self.prediction(encoded_state_before)

            #
            # here we get - encoded-next-state and predicted-reward using our `dynamics` function
            #
            # `predicted_next_encoded_state, predicted_reward = self.dynamics(encoded_state_before,
            #                                                               replay_record.executed_action)`

            #
            # validating our choice in `dynamics` function, to see what state do expect
            # to find ourselves after action execution and what reward we're going to get according
            # to our `dynamics` function
            #
            # `ignored_next_state, predicted_reward_on_decided_action = self.dynamics(encoded_state_before,
            #                                                                       decided_action)`

            raw_result = self.dynamics.forward_batch(state=encoded_state_before, actions=[
                replay_record.executed_action,
                decided_action
            ])

            prediction_on_recorded_action = raw_result[0]
            prediction_on_decided_action  = raw_result[1]

            predicted_next_encoded_state = prediction_on_recorded_action[:self.internal_representation_size]
            predicted_reward = prediction_on_recorded_action[-self.reward_size:]
            predicted_reward_on_decided_action = prediction_on_decided_action[-self.reward_size:]

            # also look for best possible action for search_depth
            decided_action_value_norm = torch.sum(decided_action_value).item()
            better_action = None

            with torch.no_grad():
                cnt = 0
                while better_action is None:
                    test_actions = [torch.rand(self.actions_size) for _ in range(self.search_depth)]
                    results = self.dynamics.forward_batch(encoded_state_before, test_actions)
                    #
                    # WARNING!!!
                    #   code below assumes, reward_size == 1
                    #
                    best_item = torch.argmax(results.T[self.internal_representation_size]).item()
                    best_item_value = results.T[self.internal_representation_size][best_item].item()
                    if best_item_value > decided_action_value_norm:
                        better_action = test_actions[best_item]
                        better_action_value = results.T[self.internal_representation_size][best_item]
                        break
                    cnt = cnt + 1
                    if cnt > 100:
                        break

            # trained by:
            # dynamics(representation(observed-state-before), recorded-action) ==
            #                                               (representation(observed-state-after), recorded-award)
            #
            # prediction(representation(observed-state-before) ==
            #                                               find_best_action(dynamics, observed-state-before)
            #

            loss_2 = self.loss_fn(predicted_next_encoded_state, encoded_state_after)
            loss_3 = self.loss_fn(predicted_reward, replay_record.observed_reward)
            step_loss = loss_2 + loss_3
            if better_action is not None:
                step_loss = step_loss + self.loss_fn(decided_action, better_action)
                step_loss = step_loss + self.loss_fn(torch.sum(decided_action_value), better_action_value)
            else:
                step_loss = step_loss + self.loss_fn(decided_action_value, predicted_reward_on_decided_action)

            losses = losses + [step_loss]

            if pbar is not None:
                pbar.set_postfix(ba=better_action,
                                 bav=best_item_value,
                                 da=decided_action.detach(),
                                 pav=decided_action_value_norm,
                                 loss=step_loss.item(),
                                 refresh=True)
        return torch.sum(torch.stack(losses))

    def create_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_run(self,
                  epochs=5000,
                  self_play_steps=50,
                  replay_steps=10,
                  reset_game=True):
        for epoch in range(epochs):
            self.eval()
            self.play_n_steps(self_play_steps, reset=reset_game)
            self.train()

            with tqdm.tqdm(range(int(len(self.replay_buffer) / replay_steps))) as pbar:
                pbar.set_description("e:%d gs:%f gr:%f" % (
                    epoch,
                    torch.sum(self.game.state).item(),
                    self.game.compute_reward().item(),
                ))

                for _ in pbar:
                    self.optimizer.zero_grad()
                    loss = self.replay(replay_steps, pbar=pbar)
                    loss.backward()
                    self.optimizer.step()
