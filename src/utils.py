import gymnasium as gym


class Restriction(gym.spaces.Space):
  def __init__(self) -> None:
    super().__init__()