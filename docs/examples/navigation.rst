Mapless Navigation
==================

.. image:: _static/trajectory-restricted.png
   :width: 400
   :align: center

In this navigation task, an TD3 agent aims to reach a goal on a two-dimensional map with complex and dynamic action spaces.
At each time step, the agent chooses an angle from a limited range to determine the subsequent steps' direction.
Restrictions are not necessarily tied to the agent’s observation but serve as an additional source of information.
The environment can contain temporary obstacles, such as other agents or objects, that may not be directly
sensed by the agent. An external entity can therefore suggest restrictions on the agent’s action space to avoid
collisions, such that the agent must select actions that maximize the expected return over varying interval subsets of
the action space.

The implementation and further details on the environment with the derivation of restrictions
can be found with instructions `here <https://github.com/michoest/drama-wrapper/tree/main/examples/navigation/>`__
or in `Grams (2023) <https://arxiv.org/abs/2306.08008/>`__.
