Cournot Game
============

In the Parameterized Cournot Game, as defined in Oesterle and Sharon (in press), two players choose their production
quantities :math:`\bm{q} = (q_1, q_2) \in \mathbb{R}^2` for a good with
the price defined as :math:`p(\bm{q}) = \max(p_{max} - q_1 - q_2, 0)`. Both players have a constant production cost
of :math:`c \geq 0` per unit.
The players’ utilities, representing their profits, are given
by :math:`u_i(\bm{q}) = q_i \cdot \left(p(\bm{q}) - c\right)`. The restrictor patiently waits
until the agents’ strategies converge and then formulates an optimal restriction to increase social welfare.

The implementation can be found with instructions `here <https://github.com/michoest/drama-wrapper/tree/main/examples/cournot/>`__.
