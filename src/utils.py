import inspect


def test_environment(env, restrictor=None, *, env_config=None, num_steps=20, num_episodes=5, individual: bool = False):
    """Creates an environment with the given configuration, and run it for a number of
    steps/episodes, using random agent actions and each step.

    Args:
      env: The environment class or object which is tested.
      env_config: The config dict for the environment.
      num_steps: The maximum number of steps for the test.
      num_steps: The maximum number of episodes for the test. If both num_steps
      and num_episodes are Null, the test will run infinitely!
    """
    if env_config is None:
        env_config = {}
    if inspect.isclass(env):
        env = env(env_config=env_config)

    print(f'Testing {type(env)}...')

    obs, info = env.reset()
    print(f'Reset: {obs}')

    current_step = 0
    current_episode = 0
    while True:
        if individual:
            actions = {
                agent_id: {agent: env.action_space['gov'].sample() for agent in env.env.agents
                     } if agent_id == 'gov' else env.action_space['agent'].sample() for agent_id in obs
            }
        else:
            actions = {
                agent_id: restrictor.compute_actions([obs['gov']])
                if restrictor and 'gov' in obs else env.action_space['gov'].sample()
                if agent_id == 'gov' else env.action_space['agent'].sample() for agent_id in obs
            }
        print(f'Actions: {actions}')
        obs, rewards, done, truncated, info = env.step(actions)
        current_step += 1
        print(f'Step:  {obs}, {rewards}, {done}, {truncated}')
        if truncated['__all__']:
            current_episode += 1
            obs, info = env.reset()
            print(f'Reset: {obs}')

        if num_steps is not None and current_step >= num_steps:
            break

        if num_episodes is not None and current_episode >= num_episodes:
            break

    print(f'Test finished!')
