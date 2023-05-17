from typing import Tuple, Type, List

import numpy as np
from gymnasium.spaces import Box
from ray.rllib import SampleBatch
from ray.rllib.algorithms.ddpg.ddpg_torch_policy import DDPGTorchPolicy
from ray.rllib.algorithms.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.models import ModelV2, ModelCatalog
from ray.rllib.models.torch.noop import TorchNoopModel
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.utils import override, try_import_torch
from ray.rllib.utils.exploration import ParameterNoise
from ray.rllib.utils.torch_utils import huber_loss, l2_loss
from ray.rllib.utils.typing import AlgorithmConfigDict, TensorType

import gymnasium as gym

from examples.agents.distributions import TorchDynamicIntervals
from examples.agents.models import MPSTD3Model

torch, nn = try_import_torch()


class MPSTD3Policy(DDPGTorchPolicy):

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            config: AlgorithmConfigDict
    ):
        # Define DDPG observation space of shape [observation] out of original [allowed_actions + observation]
        orig_space = getattr(observation_space, "original_space", observation_space)
        self.allowed_actions_len = orig_space['allowed_actions'].max_len
        only_observation_shape = (observation_space.shape[0] - self.allowed_actions_len * 2 + 1,)
        self.internal_observation_space = Box(low=-np.inf, high=np.inf, shape=only_observation_shape)

        DDPGTorchPolicy.__init__(
            self,
            observation_space,
            action_space,
            config
        )

    @override(DDPGTorchPolicy)
    def make_model_and_action_dist(
            self,
    ) -> Tuple[ModelV2, Type[TorchDistributionWrapper]]:

        num_outputs = int(np.product(self.internal_observation_space.shape))
        add_layer_norm = (
                isinstance(getattr(self, "exploration", None), ParameterNoise)
                or self.config["exploration_config"]["type"] == "ParameterNoise"
        )
        config = {
            'obs_space': self.observation_space,
            'action_space': self.action_space,
            'num_outputs': num_outputs,
            'model_config': self.config["model"],
            'framework': self.config["framework"],
            'model_interface': MPSTD3Model,
            'default_model': TorchNoopModel,
            'name': 'actor_model',
            'internal_observation_space': self.internal_observation_space,
            'actor_hidden_activation': self.config["actor_hidden_activation"],
            'actor_hiddens': self.config["actor_hiddens"],
            'critic_hidden_activation': self.config["critic_hidden_activation"],
            'critic_hiddens': self.config["critic_hiddens"],
            'add_layer_norm': add_layer_norm,
            'twin_q': self.config["twin_q"]
        }

        model = ModelCatalog.get_model_v2(**config)

        return model, TorchDynamicIntervals

    @override(DDPGTorchPolicy)
    def action_distribution_fn(
            self,
            model: ModelV2,
            *,
            obs_batch: TensorType,
            state_batches: TensorType,
            is_training: bool = False,
            **kwargs
    ) -> Tuple[TensorType, type, List[TensorType]]:
        model_out, _ = model(
            SampleBatch(obs=obs_batch[SampleBatch.CUR_OBS], _is_training=is_training)
        )

        # Split in [observation] and [available_intervals]
        observation = model_out[:, self.allowed_actions_len * 2 + 1:]
        available_intervals = model_out[:, 1:self.allowed_actions_len * 2 + 1]

        action, selected_interval = model.max_q_action(observation, available_intervals)

        dist_inputs = torch.cat([action, selected_interval, available_intervals], 1)

        return dist_inputs, TorchDynamicIntervals, []

    @override(DDPGTorchPolicy)
    def loss(
            self,
            model: ModelV2,
            dist_class: Type[TorchDistributionWrapper],
            train_batch: SampleBatch,
    ) -> List[TensorType]:
        assert isinstance(self.action_space, Box)

        def interval(out: TensorType, intervals: TensorType):
            """ Returns the interval in which a value lies

            Args:
                out (TensorType): Value to search the interval for
                intervals (TensorType): Tensor of action space intervals

            Returns:
                interval (TensorType): Interval in which out is contained
            """
            results = torch.ones(intervals.size(dim=0), 2)

            for i in range(0, int(intervals.size(1) / 2)):
                is_in_interval = torch.logical_and(intervals[:, i * 2] <= out[:, 0],
                                                   out[:, 0] <= intervals[:, i * 2 + 1])
                results[is_in_interval] = intervals[:, i * 2:i * 2 + 2][is_in_interval]

            return results

        target_model = self.target_models[model]

        twin_q = self.config["twin_q"]
        gamma = self.config["gamma"]
        n_step = self.config["n_step"]
        use_huber = self.config["use_huber"]
        huber_threshold = self.config["huber_threshold"]
        l2_reg = self.config["l2_reg"]

        input_dict = SampleBatch(
            obs=train_batch[SampleBatch.OBS], _is_training=True
        )
        input_dict_next = SampleBatch(
            obs=train_batch[SampleBatch.NEXT_OBS], _is_training=True
        )

        model_out_t, _ = model(input_dict, [], None)  # [batch, available_intervals + observation]
        model_out_tp1, _ = model(input_dict_next, [], None)
        target_model_out_tp1, _ = target_model(input_dict_next, [], None)

        # Transform SampleBatch - output shape [batch, selected_interval + observation]
        available_intervals = model_out_t[:, 1:self.allowed_actions_len * 2 + 1]
        selected_interval = interval(train_batch[SampleBatch.ACTIONS], available_intervals)
        model_out_t = torch.cat(
            [selected_interval, model_out_t[:, self.allowed_actions_len * 2 + 1:]], dim=1)

        # Run through target network - output shape [batch, selected_interval + observation]
        observation = target_model_out_tp1[:, self.allowed_actions_len * 2 + 1:]
        available_intervals_tp1 = target_model_out_tp1[:, 1:self.allowed_actions_len * 2 + 1]
        policy_tp1, selected_interval_tp1 = target_model.max_q_action(observation, available_intervals_tp1)
        target_model_out_tp1 = torch.cat([selected_interval_tp1, target_model_out_tp1[
                                                                 :, self.allowed_actions_len * 2 + 1:]], 1)

        # Standard DDPG loss calculation
        policy_t = model.get_policy_output(model_out_t)

        if self.config["smooth_target_policy"]:
            target_noise_clip = self.config["target_noise_clip"]
            clipped_normal_sample = torch.clamp(
                torch.normal(
                    mean=torch.zeros(policy_tp1.size()), std=self.config["target_noise"]
                ).to(policy_tp1.device),
                -target_noise_clip,
                target_noise_clip,
            )

            policy_tp1_smoothed = torch.min(
                torch.max(
                    policy_tp1 + clipped_normal_sample,
                    torch.tensor(
                        self.action_space.low,
                        dtype=torch.float32,
                        device=policy_tp1.device,
                    ),
                ),
                torch.tensor(
                    self.action_space.high,
                    dtype=torch.float32,
                    device=policy_tp1.device,
                ),
            )
        else:
            policy_tp1_smoothed = policy_tp1

        q_t = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])

        q_t_det_policy = model.get_q_values(model_out_t, policy_t)

        actor_loss = -torch.mean(q_t_det_policy)

        if twin_q:
            twin_q_t = model.get_twin_q_values(
                model_out_t, train_batch[SampleBatch.ACTIONS]
            )

        q_tp1 = target_model.get_q_values(target_model_out_tp1, policy_tp1_smoothed)

        if twin_q:
            twin_q_tp1 = target_model.get_twin_q_values(
                target_model_out_tp1, policy_tp1_smoothed
            )

        q_t_selected = torch.squeeze(q_t, axis=len(q_t.shape) - 1)
        if twin_q:
            twin_q_t_selected = torch.squeeze(twin_q_t, axis=len(q_t.shape) - 1)
            q_tp1 = torch.min(q_tp1, twin_q_tp1)

        q_tp1_best = torch.squeeze(input=q_tp1, axis=len(q_tp1.shape) - 1)
        q_tp1_best_masked = (1.0 - train_batch[SampleBatch.DONES].float()) * q_tp1_best

        q_t_selected_target = (
                train_batch[SampleBatch.REWARDS] + gamma ** n_step * q_tp1_best_masked
        ).detach()

        if twin_q:
            td_error = q_t_selected - q_t_selected_target
            twin_td_error = twin_q_t_selected - q_t_selected_target
            if use_huber:
                errors = huber_loss(td_error, huber_threshold) + huber_loss(
                    twin_td_error, huber_threshold
                )
            else:
                errors = 0.5 * (
                        torch.pow(td_error, 2.0) + torch.pow(twin_td_error, 2.0)
                )
        else:
            td_error = q_t_selected - q_t_selected_target
            if use_huber:
                errors = huber_loss(td_error, huber_threshold)
            else:
                errors = 0.5 * torch.pow(td_error, 2.0)

        critic_loss = torch.mean(train_batch[PRIO_WEIGHTS] * errors)

        if l2_reg is not None:
            for name, var in model.policy_variables(as_dict=True).items():
                if "bias" not in name:
                    actor_loss += l2_reg * l2_loss(var)
            for name, var in model.q_variables(as_dict=True).items():
                if "bias" not in name:
                    critic_loss += l2_reg * l2_loss(var)

        if self.config["use_state_preprocessor"]:
            input_dict[SampleBatch.ACTIONS] = train_batch[SampleBatch.ACTIONS]
            input_dict[SampleBatch.REWARDS] = train_batch[SampleBatch.REWARDS]
            input_dict[SampleBatch.DONES] = train_batch[SampleBatch.DONES]
            input_dict[SampleBatch.NEXT_OBS] = train_batch[SampleBatch.NEXT_OBS]
            [actor_loss, critic_loss] = model.custom_loss(
                [actor_loss, critic_loss], input_dict
            )

        model.tower_stats["q_t"] = q_t
        model.tower_stats["actor_loss"] = actor_loss
        model.tower_stats["critic_loss"] = critic_loss
        model.tower_stats["td_error"] = td_error

        return [actor_loss, critic_loss]
