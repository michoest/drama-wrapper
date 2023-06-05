def play(env, policies, *, max_iter=1_000, verbose=False):
    env.reset()
    env.render()

    for agent in env.agent_iter():
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

        env.step(action)

    env.render()


def restriction_aware_random_policy(observation):
    observation, restriction = observation["observation"], observation["restriction"]
    return restriction.sample()
