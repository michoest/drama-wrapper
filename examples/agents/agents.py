import ray
from ray import rllib
import tensorflow as tf
import gymnasium as gym


class DiscreteRestrictionAwareAgentModel(rllib.models.tf.fcnet.FullyConnectedNetwork):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, *args, **kwargs):
      # Original observation space consists of two components: Real observation and allowed actions
      original_obs_space = getattr(obs_space, 'original_space', obs_space)
      assert (isinstance(original_obs_space, gym.spaces.Dict) and
              'allowed_actions' in original_obs_space.spaces and
              'observation' in original_obs_space.spaces)
      super(DiscreteRestrictionAwareAgentModel, self).__init__(obs_space, action_space, num_outputs, model_config, name, *args, **kwargs)
      
      # Use only the "real" observation to define the model, not the allowed actions
      self.internal_model = ray.rllib.models.tf.fcnet.FullyConnectedNetwork(original_obs_space['observation'], action_space, num_outputs, model_config, name + "_internal")

    def forward(self, input_dict, state, seq_lens):
      # Get logits from internal model
      logits, _ = self.internal_model({ 'obs': input_dict['obs']['observation'] })

      # Transform allowed_actions into logit mask
      allowed_action_mask = tf.maximum(tf.math.log(input_dict['obs']['allowed_actions']), tf.float32.min)

      # Mask logits with allowed_action_mask to prevent forbidden actions
      masked_logits = logits + allowed_action_mask

      return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()