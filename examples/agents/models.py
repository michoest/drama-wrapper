from typing import Optional, List

import numpy as np
import ray
from gymnasium.spaces import Box
from ray import rllib
import tensorflow as tf
import gymnasium as gym
from ray.rllib.algorithms.ddpg.ddpg_torch_model import DDPGTorchModel
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType

torch, nn = try_import_torch()


class DiscreteRestrictionAwareAgentModel(rllib.models.tf.fcnet.FullyConnectedNetwork):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, *args, **kwargs):
        # Original observation space consists of two components: Real observation and allowed actions
        original_obs_space = getattr(obs_space, 'original_space', obs_space)
        assert (isinstance(original_obs_space, gym.spaces.Dict) and
                'allowed_actions' in original_obs_space.spaces and
                'observation' in original_obs_space.spaces)
        super(DiscreteRestrictionAwareAgentModel, self).__init__(obs_space, action_space, num_outputs, model_config,
                                                                 name, *args, **kwargs)

        # Use only the "real" observation to define the model, not the allowed actions
        self.internal_model = ray.rllib.models.tf.fcnet.FullyConnectedNetwork(original_obs_space['observation'],
                                                                              action_space, num_outputs, model_config,
                                                                              name + "_internal")

    def forward(self, input_dict, state, seq_lens):
        # Get logits from internal model
        logits, _ = self.internal_model({'obs': input_dict['obs']['observation']})

        # Transform allowed_actions into logit mask
        allowed_action_mask = tf.maximum(tf.math.log(input_dict['obs']['allowed_actions']), tf.float32.min)

        # Mask logits with allowed_action_mask to prevent forbidden actions
        masked_logits = logits + allowed_action_mask

        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()


class BatchScaling(nn.Module):
    """ Parameterizable scaling layer

    Args:
        action_space (gym.Space): action space defining the boundaries for the scaling function
    """

    def __init__(self, action_space):
        super().__init__()
        assert isinstance(action_space, Box)

        self.bounded = np.logical_and(
            action_space.bounded_above, action_space.bounded_below
        ).any()

        if self.bounded:
            self.low = action_space.low[0]
            self.high = action_space.high[0]
        else:
            self.low = 0.0
            self.high = 1.0

    def forward(self, action, interval_action_space):
        if not self.bounded:
            action = nn.Sigmoid()(action)

        scaled_actions = torch.add(
            torch.mul(
                (action - self.low).reshape(action.size(0), -1), (torch.subtract(
                    interval_action_space[:, :, 1], interval_action_space[:, :, 0]) / (self.high - self.low))),
            interval_action_space[:, :, 0])

        if self.bounded:
            return torch.min(
                torch.max(scaled_actions, interval_action_space[:, :, 0]),
                interval_action_space[:, :, 1])
        else:
            return scaled_actions


class MPSTD3Model(DDPGTorchModel):

    def __init__(
            self,
            obs_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            num_outputs: int,
            model_config: ModelConfigDict,
            name: str,
            actor_hiddens: Optional[List[int]] = None,
            actor_hidden_activation: str = "relu",
            critic_hiddens: Optional[List[int]] = None,
            critic_hidden_activation: str = "relu",
            twin_q: bool = False,
            add_layer_norm: bool = False,
            **kwargs
    ):
        assert isinstance(action_space, Box)
        DDPGTorchModel.__init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            actor_hiddens,
            actor_hidden_activation,
            critic_hiddens,
            critic_hidden_activation,
            twin_q,
            add_layer_norm
        )

        self.scaling_layer = BatchScaling(self.action_space)

    def max_q_action(self, observation: TensorType, intervals: TensorType) -> TensorType:
        max_action = torch.zeros(observation.size(0), 1)
        max_interval = torch.zeros(intervals.size(0), 2)
        max_q = torch.full((observation.size(0), 1), -np.inf)

        for i in range(0, int(intervals.size(1) / 2), 2):
            new_observation = torch.cat([intervals[:, i:i + 2], observation], 1)
            action = self.policy_model(new_observation)
            q_value = self.q_model(torch.cat([new_observation, action], -1))

            empty = intervals[:, i] != intervals[:, i + 1]
            improved = q_value > max_q
            update = torch.logical_and(empty.reshape(-1, 1), improved)

            max_action[update] = action[update]
            max_interval[torch.cat([update, update], 1)] = intervals[:, i:i+2][torch.cat([update, update], 1)]
            max_q[update] = q_value[update]

        scaled_max_action = self.scaling_layer(max_action, max_interval.reshape(max_interval.size(0), -1, 2))

        return scaled_max_action, max_interval,
