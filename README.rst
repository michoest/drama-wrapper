DRAMA at the PettingZoo: Dynamically Restricted Action Spaces for Multi-Agent Reinforcement Learning Frameworks
===============================================================================================================

Purpose of this repository
--------------------------

This repository contains the reference implementation of the *DRAMA*
framework as introduced in *Oesterle et al. (2024): DRAMA at the
PettingZoo: Dynamically Restricted Action Spaces for Multi-Agent
Reinforcement Learning Frameworks. Submitted to HICSS 2024.*

Installation
------------

To install the DRAMA library:

.. code-block::

    $  pip install drama-wrapper

Usage
-----

In analogy to the AEC of *PettingZoo*

::

   env.reset()
   for agent in env.agent_iter():
       observation, reward, termination, truncation, info = env.last()
       action = env.action_space(agent).sample() # this is where you would insert your policy
       env.step(action)

the *DRAMA* loop can be imported and used as follows:

::
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

Please refer to ``getting-started.ipynb`` for a first full example.

Documentation
-------------

The full documentation of the code can be found at
https://drama-wrapper.readthedocs.io/.

Citation
--------

To cite this project in a publication, please use

::

   @misc{oesterle-2023-drama,
       author = {Oesterle, Michael and Grams, Tim},
       title = {DRAMA},
       year = {2023},
       url = {https://github.com/michoest/hicss-2024}
   }

or use the ``CITATION.cff`` file which is part of the package.
