from nu_zero_test_games import RLGame
import gym
import torch
import torch.nn.functional as func


class RLBlackJackGame(RLGame):
    def __init__(self):
        blackjack_observation_size = 10
        super(RLBlackJackGame, self).__init__(blackjack_observation_size)
        self.env = gym.make("Blackjack-v1")
        self._reward_acc = torch.zeros(1)
        self.done: bool = False
        self.alphabet = func.one_hot(torch.as_tensor([
            range(32),
            [(x % 11) for x in range(32)],
            [(x % 2) for x in range(32)]
        ])).float()

    def _encode_observation(self, blackjack_observation):
        (score, dealer_hand, ace) = blackjack_observation
        encoded_score = self.alphabet[0][score]
        encoded_dealer_hand = self.alphabet[1][dealer_hand]
        encoded_ace = self.alphabet[2][1 if ace else 0]
        return torch.concat([
            encoded_score,
            encoded_dealer_hand,
            encoded_ace
        ])
    """
        observation = env.reset()
        for step in range(50):
            action = env.action_space.sample()  # or given a custom model, action = policy(observation)
            observation, reward, done, info = env.step(action)
    """

    def run_action(self, action):
        game_action = torch.argmax(action).item()
        observation, reward, done, info = self.env.step(game_action)
        self.state = self._encode_observation(observation)
        self.done = done
        print("observation: ", observation, reward, done)
        self._reward_acc = self._reward_acc + torch.as_tensor(reward)
        return self.state, torch.as_tensor([reward])

    def compute_reward(self, s=None):
        if s is not None:
            print("need to get reward for ", s)
        return self._reward_acc

    def reset(self, skip_reward_acc=True):
        if not skip_reward_acc:
            self._reward_acc = torch.zeros(1)
        self.state = self._encode_observation(self.env.reset())
        return self.state


