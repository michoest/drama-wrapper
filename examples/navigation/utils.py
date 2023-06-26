import numpy as np
from gymnasium.error import DependencyNotInstalled
from matplotlib import pyplot as plt

from examples.navigation.env import NavigationEnvironment
from examples.navigation.restrictor import NavigationRestrictor
from drama.restrictions import IntervalUnionRestriction
from drama.wrapper import RestrictionWrapper


def do_nothing_on_invalid_violation_fn(env, action, restriction):
    env._cumulative_rewards = {'agent_0': 0.0}
    env.rewards = {'agent_0': 0.0}
    env.trajectory.append([float(env.agent.x),
                           float(env.agent.y)])
    env.current_step += 1
    if env.current_step >= env.STEPS_PER_EPISODE:
        env.truncations = {'agent_0': True}


def render(vis_policy, env_config, restriction_violation_fns, seed):
    try:
        import pygame
        from pygame import gfxdraw
    except ImportError:
        raise DependencyNotInstalled('Pygame is not installed, run `pip install pygame`')

    def draw_polygon_border(polygon_coordinates):
        for index, coordinate in enumerate(polygon_coordinates):
            if index == len(polygon_coordinates) - 1:
                pygame.draw.line(canvas, (0, 0, 0), (int(coordinate[0]),
                                                     int(coordinate[1])),
                                 (int(polygon_coordinates[0][0]),
                                  int(polygon_coordinates[0][1])), 2)
            else:
                pygame.draw.line(canvas, (0, 0, 0), (int(coordinate[0]),
                                                     int(coordinate[1])),
                                 (int(polygon_coordinates[index + 1][0]),
                                  int(polygon_coordinates[index + 1][1])), 2)

    trajectory = []
    vis_restrictor = NavigationRestrictor(obstacle_count=7,
                                          obstacle_position_covariance=[[4.0, 0.0], [0.0, 4.0]],
                                          obstacle_mean_size=1.0,
                                          obstacle_variance_size=0.2,
                                          obstacle_size_range=0.5,
                                          start_seed=seed,
                                          safety_angle=8,
                                          min_angle=-110.0,
                                          max_angle=110.0)
    vis_env = RestrictionWrapper(NavigationEnvironment(env_config),
                                 vis_restrictor,
                                 restriction_violation_fns=restriction_violation_fns,
                                 return_object=True)
    vis_width = vis_env.env.WIDTH
    vis_height = vis_env.env.HEIGHT

    vis_env.reset()
    trajectory.append([float(vis_env.env.agent.x),
                       float(vis_env.env.agent.y)])
    for vis_agent in vis_env.agent_iter():
        obs, rew, term, trunc, inf = vis_env.last()
        if vis_agent == 'agent_0':
            vis_action = vis_policy.select_action(obs['observation'])
        else:
            vis_action = vis_restrictor.act(obs)

        if term or trunc:
            vis_action = None

        vis_env.step(vis_action)
        if vis_agent == 'agent_0':
            trajectory.append([float(vis_env.env.agent.x),
                               float(vis_env.env.agent.y)])

    window = None
    window_scale = 50 if vis_width < 15 else 30

    window_width = vis_width * window_scale
    window_height = vis_height * window_scale
    if window is None:
        pygame.init()
        pygame.font.init()
        window = pygame.Surface((window_width, window_height))

    canvas = pygame.Surface((window_width, window_height))
    canvas.fill((232, 232, 232))

    for obstacle in vis_restrictor.obstacles:
        gfxdraw.filled_polygon(canvas, obstacle.coordinates * window_scale, (136, 136, 136))
        draw_polygon_border(obstacle.coordinates * window_scale)

    gfxdraw.circle(canvas, int(vis_env.env.goal.centroid.coords[0][0] * window_scale),
                   int(vis_env.env.goal.centroid.coords[0][1] * window_scale),
                   int((vis_env.env.goal.bounds[3] -
                        vis_env.env.goal.centroid.coords[0][1]) * window_scale), (34, 139, 34))

    if len(trajectory) > 1:
        pygame.draw.aalines(canvas, (232, 232, 232), False, np.multiply(trajectory, window_scale), 0)

    gfxdraw.pie(canvas, int(float(vis_env.env.agent.x) * window_scale),
                int(float(vis_env.env.agent.y) * window_scale),
                int(float(vis_env.env.agent.step_size + vis_env.env.agent.radius) * window_scale),
                int(float(vis_env.env.agent.perspective) - float(vis_env.env.ACTION_RANGE) / 2),
                int(float(vis_env.env.agent.perspective) + float(vis_env.env.ACTION_RANGE) / 2), (0, 0, 0))

    gfxdraw.filled_circle(canvas, int(float(vis_env.env.agent.x) * window_scale),
                          int(float(vis_env.env.agent.y) * window_scale),
                          int(float(vis_env.env.agent.radius) * window_scale), (65, 105, 225))

    canvas = pygame.transform.flip(canvas, False, True)
    window.blit(canvas, (0, 0))
    plt.imshow(np.transpose(
        np.array(pygame.surfarray.pixels3d(window)), axes=(1, 0, 2)
    ))
    plt.axis('off')
    plt.show()
