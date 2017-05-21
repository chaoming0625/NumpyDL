=========
Objective
=========

1. What is the Objective Function?
==================================

The objective of a linear programming problem will be to maximize or to minimize
some numerical value. This value may be the expected net present value of a project
or a forest property; or it may be the cost of a project; it could also be the amount
of wood produced, the expected number of visitor-days at a park, the number of
endangered species that will be saved, or the amount of a particular type of habitat
to be maintained. Linear programming is an extremely general technique, and its
applications are limited mainly by our imaginations and our ingenuity.

The objective function indicates how much each variable contributes to the value to
be optimized in the problem. The objective function takes the following general form:

.. math::

    maximize or minimize  Z = \sum_{i=1}^{n}c_i X_i

where

* :math:`c_i` = the objective function coefficient corresponding to the ith variable
* :math:`X_i` = the :math:`i`-th decision variable.
* The summation notation used here was discussed in the section above on linear
  functions. The summation notation for the objective function can be expanded out as
  follows: :math:`Z = \sum_{i=1}^{n} c_i X_i = c_1 X_1 + c_2 X_2 + \cdots + c_n X_n`


The coefficients of the objective function indicate the contribution to the value of
the objective function of one unit of the corresponding variable. For example, if the
objective function is to maximize the present value of a project, and :math:`X_i` is
the :math:`i`-th possible activity in the project, then :math:`c_i` (the objective
function coefficient corresponding to :math:`X_i` ) gives the net present value generated
by one unit of activity :math:`i`. As another example, if the problem is to minimize
the cost of achieving some goal, :math:`X_i` might be the amount of resource :math:`i`
used in achieving the goal. In this case, :math:`c_i` would be the cost of using one
unit of resource :math:`i`.

Note that the way the general objective function above has been written implies that
each variable has a coefficient in the objective function. Of course, some variables
may not contribute to the objective function. In this case, you can either think of
the variable as having a coefficient of zero, or you can think of the variable as
not being in the objective function at all.



