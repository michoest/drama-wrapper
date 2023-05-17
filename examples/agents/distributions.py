from typing import List

from ray.rllib.models.torch.torch_action_dist import TorchDeterministic
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import override, try_import_torch
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()


class TorchDynamicIntervals(TorchDeterministic):
    """ Rllib implementation of a dynamic interval action space / distribution

    Args:
        inputs (list): The first two values determine the chosen interval, and the rest the entire action space
        model (TorchModelV2)
    """

    @override(TorchDeterministic)
    def __init__(self,
                 inputs: List[TensorType],
                 model: TorchModelV2):
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.from_numpy(inputs)
            if isinstance(model, TorchModelV2):
                inputs = inputs.to(next(model.parameters()).device)
        super().__init__(inputs[:, 0].reshape(-1, 1), model)
        self.last_sample = None
        self.chosen_interval = inputs[:, 1:3]
        self.allowed_intervals = inputs[:, 3:].reshape(inputs[:, 3:].size(0), -1, 2)
        self.width_intervals = torch.subtract(self.allowed_intervals[:, :, 1], self.allowed_intervals[:, :, 0])
        self.total_width = torch.sum(self.width_intervals, 1).reshape(-1, 1)

    @override(TorchDeterministic)
    def deterministic_sample(self) -> TensorType:
        return self.inputs

    def random_sample(self) -> TensorType:
        sample = []
        for index, observation in enumerate(self.allowed_intervals):
            random_number = torch.mul(torch.rand(1), self.total_width)
            for i, interval in enumerate(observation):
                if random_number > interval[1] - interval[0]:
                    random_number -= interval[1] - interval[0]
                else:
                    sample.append(torch.add(random_number, interval[0]))
                    break

        return torch.cat(sample).reshape(-1, 1)