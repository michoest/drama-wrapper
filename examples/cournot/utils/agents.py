import numpy as np
from gymnasium.spaces import Box


# In the unrestricted scenario, players always choose the best response to the
# opponent's action
class UnrestrictedCournotAgent:
    def __init__(self, maximum_price: float, cost: float) -> None:
        self.maximum_price = maximum_price
        self.cost = cost

        self.action_space = Box(0, maximum_price)

    def act(self, observation) -> float:
        if (opponent_action := observation[0]) is None:
            return float(self.action_space.sample())
        else:
            return np.clip(
                (self.maximum_price - self.cost - opponent_action) / 2,
                0,
                self.maximum_price,
            )


# Players always choose the best response to the opponent's action,
# given the restriction
class RestrictedCournotAgent:
    def __init__(self, maximum_price: float, cost: float) -> None:
        self.maximum_price = maximum_price
        self.cost = cost

        self.action_space = Box(0, maximum_price)

    def act(self, observation) -> float:
        observation, restriction = (
            observation["observation"],
            observation["restriction"],
        )
        if (opponent_action := observation[0]) is None:
            return np.random.uniform(0, self.maximum_price)
        else:
            if restriction.contains(
                unrestricted_best_response := (
                    self.maximum_price - self.cost - opponent_action
                )
                / 2
            ):
                return unrestricted_best_response
            else:
                [ll, lu], _ = restriction.last_interval_before_or_within(
                    unrestricted_best_response
                )
                [ul, uu], _ = restriction.first_interval_after_or_within(
                    unrestricted_best_response
                )

                if ll is None:
                    return float(ul)
                elif ul is None:
                    return float(lu)
                else:
                    ll, lu, ul, uu = float(ll), float(lu), float(ul), float(uu)

                    return (
                        lu
                        if (unrestricted_best_response - lu)
                        < 2 * (ul - unrestricted_best_response)
                        else ul
                    )
