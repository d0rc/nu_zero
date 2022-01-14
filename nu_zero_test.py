from nu_zero import RLRoutine, RLMakeItOneGame

rlr = RLRoutine(
    actions_size=4,
    exploration_rate=0.1,
    internal_representation_size=3,
    observation_size=10,
    reward_size=1,
    search_depth=1_000,
    game=RLMakeItOneGame
)

rlr.train_run(
    epochs=5000,
    self_play_steps=50,
    replay_steps=1,
    reset_game=True,
)
