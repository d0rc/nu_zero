import random
import pandas as pd


class RLLogger:
    def __init__(self, instance=None):
        super(RLLogger, self).__init__()
        if instance is None:
            instance = random.randint(1_000_000, 9_999_999)
        self.instance = instance
        self.info = pd.DataFrame(columns=[
            "epoch",
            "game_state",
            "game_result",
            "step", "max_step",
            "best_action",
            "best_action_value",
            "decided_action",
            "decided_action_value",
            "loss"
        ])
        self.epoch = None
        self.game_state = None
        self.game_result = None
        self.step = None
        self.max_step = None

    def log(self,
            epoch=None,
            game_state=None,
            game_result=None,
            step=None, max_step=None,
            best_action=None,
            best_action_value=None,
            decided_action=None,
            decided_action_value=None,
            loss=None):
        if epoch is not None:
            self.epoch = epoch
            self.game_state = game_state
            self.game_result = game_result
            self.step = step
            self.max_step = max_step
        else:
            self.info = self.info.append(pd.Series([
                self.epoch,
                self.game_state,
                self.game_result,
                self.step, self.max_step,
                best_action,
                best_action_value,
                decided_action,
                decided_action_value,
                loss
            ], index=self.info.columns), ignore_index=True)
            self.info.to_csv("logger-%d.csv" % self.instance)
