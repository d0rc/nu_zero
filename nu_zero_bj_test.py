import multiprocessing

import torch

from nu_zero import RLRoutine
from blackjack import RLBlackJackGame
from nu_zero_logger import RLLogger


def worker(idx):
    torch.random.manual_seed(idx)
    rlr = RLRoutine(
        actions_size=2,
        exploration_rate=0.1,
        internal_representation_size=16,
        observation_size=32 * 3,
        reward_size=1,
        search_depth=1_00,
        ignore_predictor_hypothesis=True,
        lr=0.00001,
        game=lambda observation_size: RLBlackJackGame(),
        logger=RLLogger,
    )

    rlr.train_run(
        epochs=150,
        learning_epochs_max=30,
        learning_stop_loss=0.1,
        self_play_steps=50,
        replay_steps=10,
        reset_game=True,
        save_best=("best-model-%d.bin" % idx),
        initial_agent_score_min=100.0,
    )


if __name__ == '__main__':
    with multiprocessing.Pool(1) as pool:
        pool.map(worker, range(64))
