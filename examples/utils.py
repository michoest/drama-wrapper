def play(env, policies, *, max_iter=1_000, verbose=False, record_trajectory=False):
    env.reset()
    env.render()

    if record_trajectory:
        trajectory = []

    for i, agent in zip(range(max_iter), env.agent_iter()):
        observation, reward, termination, truncation, info = env.last()
        if verbose:
            print(
                f"{agent=}, {observation=}, {reward=}, {termination=}, {truncation=}, \
                {info=}"
            )

        action = (
            policies[agent](observation) if not termination and not truncation else None
        )
        if verbose:
            print(f"{action=}")

        if record_trajectory:
            trajectory.append((agent, observation, reward, termination, truncation, info, action))

        env.step(action)

    env.render()

    return trajectory if record_trajectory else None


def restriction_aware_random_policy(observation):
    observation, restriction = observation["observation"], observation["restriction"]
    return restriction.sample()
