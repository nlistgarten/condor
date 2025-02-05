Condor
======

NASA's Condor is a framework for mathematical modeling of engineering systems in Python. For engineers with a deadline.

`Documentation <https://condor.pages.fit.nasa.gov/condor/>`_

Condor is a new mathematical modeling framework for Python, developed at NASA's Ames Research Center. Initial development began in April 2023 to address modeling challenges for aircraft synthesis and robust orbital trajectory design.
Condor emphasizes modern approaches from the scientific python community, and leverages many open-source software packages to expedite development and ensure robust and efficient run-time.
Condor follows modern python principles and best practices and leverages existing solvers and tools wherever possible.
Condor is unique in that it uses "metaprogramming" to create an mathematical and expressive "domain specific language" (DSL) for defining models. The declarative nature of this DSL means that user models look nearly identical to mathematical descriptions making it easy to write and maintain models. Condor has been used for aircraft hybrid-electric propulsion conceptual design analysis and optimal robust orbital trajectory design. However, Condor is a general modeling framework and is not designed to solve any particular problem; the user is responsible for modeling choices and Condor makes it easy to implement them.

To best understand Condor, we can consider a simple benchmark problem which consists of a set of coupled algebraic expressions, which can be represented as a system of algebraic equations

.. code-block:: python

  class Coupling(co.AlgebraicSystem):
      x = parameter(shape=3)
      y1 = variable(initializer=1.)
      y2 = variable(initializer=1.)

      residual(y1 == x[0] ** 2 + x[1] + x[2] - 0.2 * y2)
      residual(y2 == y1**0.5 + x[0] + x[1])

This parametric model can be evaluated by providing the values for the parameters; the resulting object has values for its inputs and outputs bound, so the solved values for ``y1`` and ``y2`` can be acccessed easily:

.. code-block:: python

   coupling = Coupling([5., 2., 1]) # evaluate the model numerically
   print(coupling.y1, coupling.y2) # individual elements are bound numerically
   print(coupling.variable) # fields are bound as a dataclass

Models can also be seamlessly built-up, with parent models accessing any input or output of the child models. For example, we can optimize this coupled algebraic system,

.. code-block:: python

  class Sellar(co.OptimizationProblem):
      x = variable(shape=3, lower_bound=0, upper_bound=10)
      coupling = Coupling(x)
      y1, y2 = coupling

      objective = x[2]**2 + x[1] + y1 + exp(-y2)
      constraint(y1 > 3.16)
      constraint(24. > y2)

This ``OptimizationProblem`` can be solved and the sub-model can be accesed directly,

.. code-block:: python

   Sellar.set_initial(x=[5,2,1])
   sellar = Sellar()
   print(sellar.objective) # scalar value
   print(sellar.constraint) # field
   print(sellar.coupling.y1) # sub-model element



In Condor, users construct parameterized ``Model``'s from a particular `Model Template` which defines the fields from which the Model can draw elements., which defines , when called, performs the numerical evaluation of the model and binds the values to a Python object. For example, 

Condor is a new mathematical modeling framework for Python, developed at NASA's Ames Research Center. Initial development began in April 2023 to address modeling challenges for aircraft synthesis and robust orbital trajectory design. Condor emphasizes modern approaches from the scientific python community, and leverages many open-source software packages to expedite development and ensure robust and efficient run-time. Most of the modifications needed to complete the user exercises are not Condor-specific, but general Python programming.

Condor is unique in that it uses "metaprogramming" to create an mathematical and expressive "domain specific language" (DSL) for defining models. The declrative nature of this DSL can be seen in the definition of the LinCov models in ``DemoCW.py``. In Condor, ``ODESystem``'s have fields for ``state``, ``initial`` values, ``dynamic_output``, and ``parameter`` elements. The ``TrajectoryAnalysis`` inner model simulates the inner model, inheriting the ``ODESystem``'s field's elements  and adding the ``trajectory_output`` field for defining overall performance metrics. When a model like ``TrajectoryAnalysis`` is created, it defines a new class that can be instantiated with values for each of the ``parameter`` elements. The object that is created has "dot" access to the ``parameter`` and ``trajectory_output`` values, as well as time-histories for ``state`` and ``dynamic_output``. The ``TrajectoryAnalysis`` model will simulate the ``ODESystem`` along with any ``Event``'s that have been defined at the time the ``TrajectoryAnalysis`` model is created. A raw datastructure with the simulation time ``t``, state ``x``, dynamic output ``y``, and event log ``e`` is available from a simulation's ``_res`` attribute, e.g., ``sim._res.t`` is a list of the timesteps for the simulation. See the functions in ``plot.py`` for examples of accessing and manipulating time histories.

Installation
------------

To install, clone the repository and install with pip

.. code:: bash

   $ git clone https://github.com/nasa/condor.git
   $ cd condor/
   $ pip install .


License
-------

This software is released under the `NASA Open Source Agreement Version 1.3 <https://github.com/nasa/condor/raw/master/license.pdf>`_.

Notices
-------

Copyright © 2024 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

Disclaimers
-----------

No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
