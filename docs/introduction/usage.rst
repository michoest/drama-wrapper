Basic Usage
===========

Installation
------------

To install the DRAMA library: ``pip install drama-wrapper``.

Note that the PyPI package does not include the examples and their corresponding dependencies.
To run the reference implementations, clone this GitHub repository,
execute ``pip install -r requirements.txt``, and click through the Jupyter notebooks.

Initializing Restrictors and Environments
-----------------------------------------

Using environments with DRAMA is similar to to PettingZoo but with an additional wrapper.
The environment may stay unaltered and the wrapper is highly configurable.
The restrictor is initialized similarly to agents in most common reinforcement learning frameworks.

.. code-block:: python

   from drama.restrictors import Restrictor
   from drama.wrapper import RestrictionWrapper

   env = ...
   restrictor = Restrictor(...)
   wrapper = RestrictionWrapper(env, restrictor)

Interacting With Environments
-----------------------------------------

The interaction with the environment stays unaltered:

.. code-block:: python

   wrapper.reset()
   for agent in wrapper.agent_iter():
       observation, reward, termination, truncation, info = wrapper.last()
       action = policies[agent](observation)
       wrapper.step(action)

However, note that the policies in the cycle must be extended by the restrictors which compute the restrictions
as actions prior to the original agents.

Please refer to ``getting-started.ipynb`` for a first full example.
