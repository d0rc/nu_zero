from nu_zero import RLRoutine
from nu_zero_test_games import RLMakeItOneGame

rlr = RLRoutine(
    actions_size=4,
    exploration_rate=0.1,
    internal_representation_size=3,
    observation_size=10,
    reward_size=1,
    search_depth=1_000,
    game=lambda observation_size: RLMakeItOneGame(observation_size=observation_size, max_bucket_value=0.2)
)

rlr.train_run(
    epochs=5000,
    self_play_steps=50,
    replay_steps=1,
    reset_game=True,
)
