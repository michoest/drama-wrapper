.. raw:: html

    <h2 align="center">DRAMA at the PettingZoo:<br>Dynamically Restricted Action Spaces for Multi-Agent Reinforcement Learning Frameworks</h2>

**DRAMA is a framework to extend the PettingZoo agent-environment cycle with complex, dynamic,
and potentially self-learning action space restrictions.**
DRAMA contains a wrapper implementing a new agent-restrictor-environment cycle,
restrictions as Gymnasium Spaces, classes for restriction learning agents, and helpful utilities:

.. code-block:: python

   from drama.restrictors import Restrictor
   from drama.wrapper import RestrictionWrapper

   env = ...
   restrictor = Restrictor(...)
   wrapper = RestrictionWrapper(env, restrictor)
   policies = {...}

   wrapper.reset()
   for agent in wrapper.agent_iter():
       observation, reward, termination, truncation, info = wrapper.last()
       action = policies[agent](observation)
       wrapper.step(action)


.. note::
   Further documentation is in development ...

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Introduction

   introduction/drama
   introduction/usage

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API

   api/wrapper
   api/restrictors
   api/restrictions
   api/utils

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Examples

   examples/cournot
   examples/navigation
   examples/traffic

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Development

   Github <https://github.com/michoest/hicss-2024>
   Contribute to the Docs <https://github.com/michoest/hicss-2024/tree/main/docs>
