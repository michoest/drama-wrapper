from ray import rllib

from ..utils import Restriction


class RestrictedRLlibEnvironment(rllib.env.MultiAgentEnv):
  def __init__(self) -> None:
    super().__init__()