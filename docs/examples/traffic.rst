Traffic Control
===============

.. image:: _static/traffic-setup.png
   :width: 400
   :align: center

In traffic control, multiple agents select the shortest route from their starting points to the respective destinations.
However, the travel time along each road segment is influenced by the relative number of agents using it.
Therefore, this example trains a DQN restrictor in the above traffic network to determine the optimal configuration
of available roads for each agent.

The implementation can be found with instructions `here <https://github.com/michoest/drama-wrapper/tree/main/examples/traffic/>`__.
