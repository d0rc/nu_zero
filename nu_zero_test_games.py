import torch


class RLGame:
    def __init__(self, observation_size):
        self.observation_size = observation_size
        self.state = torch.zeros(observation_size)
        self.episode = 1

    def compute_reward(self, s=None):
        pass

    def reset(self):
        self.state = torch.zeros(self.observation_size)
        self.episode = self.episode + 1

    def run_action(self, action):
        pass


class RLMakeItOneGame(RLGame):
    def __init__(self, observation_size, max_bucket_value=0.2):
        super().__init__(observation_size=observation_size)
        self.hidden_state = 0
        self.max_bucket_value = max_bucket_value

    """
        reward reflects how close sum of state is to 1
    """
    def compute_reward(self, s=None):
        if s is None:
            s = self.state
        return torch.exp(torch.pow(torch.sum(s) - torch.ones(1), 2) * torch.as_tensor(-10.0))

    def run_action(self, action):
        current_score = self.compute_reward()
        int_action = torch.argmax(action).item()
        if int_action == 0:
            self.state.data[self.hidden_state] = self.state.data[self.hidden_state] + torch.as_tensor(0.01)
            if self.max_bucket_value is not None and self.state.data[self.hidden_state] > self.max_bucket_value:
                self.state.data[self.hidden_state] = 0.0
        if int_action == 1:
            self.state.data[self.hidden_state] = self.state.data[self.hidden_state] - torch.as_tensor(0.01)
            if self.max_bucket_value is not None and self.state.data[self.hidden_state] > self.max_bucket_value:
                self.state.data[self.hidden_state] = 0.0
        if int_action >= 2:
            self.hidden_state = (self.hidden_state + int_action) % len(self.state)

        return self.state, self.compute_reward() - current_score
