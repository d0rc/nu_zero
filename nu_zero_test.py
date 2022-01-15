import multiprocessing

import torch

from nu_zero import RLRoutine
from nu_zero_test_games import RLMakeItOneGame
from nu_zero_logger import RLLogger


def worker(idx):
    torch.random.manual_seed(idx)
    rlr = RLRoutine(
        actions_size=4,
        exploration_rate=0.1,
        internal_representation_size=3,
        observation_size=10,
        reward_size=1,
        search_depth=1_000,
        game=lambda observation_size: RLMakeItOneGame(observation_size=observation_size, max_bucket_value=0.2),
        logger=RLLogger,
    )

    rlr.train_run(
        epochs=150,
        self_play_steps=50,
        replay_steps=10,
        reset_game=True,
        save_best=("best-model-%d.bin" % idx)
    )


if __name__ == '__main__':
    with multiprocessing.Pool(8) as pool:
        pool.map(worker, range(64))
