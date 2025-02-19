=================
Condor Design
=================

Overall goals and API
======================

Condor is a new mathematical modeling framework for Python, developed at
NASA's Ames Research Center. Initial development began in April 2023 to
address model implementation challenges for aircraft synthesis and
robust orbital trajectory design.  Condor emphasizes modern approaches
from the scientific python community, and leverages many open-source
software packages to expedite development and ensure robust and
efficient run-time.

The goal is for Condor to help evaluate numerical models and then get
out of the way. One key aspect to achieve this goal was to create an API
that looked as much like the mathematical description as possible with
as little distraction from programming cruft as possible.  For example,
Sellar [sellar]_ introduces an arbitrary system of algebraic equations to
represent coupling in multi-disciplinary analysis,

.. math::
   \begin{align}
   y_{1}&=x_{0}^{2}+x_{1}+x_{2}-0.2\,y_{2} \\
   y_{2}&=\sqrt{y_{1}}+x_{0}+x_{1}
   \end{align}

should be writable as

.. code-block:: python

     y1 == x[0] ** 2 + x[1] + x[2] - 0.2 * y2
     y2 == y1**0.5 + x[0] + x[1]

Of course, in both the mathematic and programmatic description, the source of each
symbol must be defined. In an engineering memo, we might say "where :math:`y_1,y_2`
are the variables to solve and :math:`x \in \mathbb{R}^3` parameterizes the system of
equations," which suggests the API for an algebraic system of equations as 

.. code-block:: python

    import condor as co
    class Coupling(co.AlgebraicSystem):
        x = parameter(shape=3)
        y1 = variable(initializer=1.)
        y2 = variable(initializer=1.)

        residual(y1 == x[0] ** 2 + x[1] + x[2] - 0.2 * y2)
        residual(y2 == y1**0.5 + x[0] + x[1])

which can be evaluated by instantiating the model with numerical values for hte
parameter, which *binds* the result from the iterative solver to the named *element* and
*field* attributes on the instance,

.. code-block:: python

    coupling = Coupling([5., 2., 1]) # evaluate the model numerically
    print(coupling.y1, coupling.y2) # individual elements are bound numerically
    print(coupling.variable) # fields are bound as a dataclass

This pythonic datastructure allows Condor to be integrated into larger analysis workflows
with as little Condor-specific coding as possible. 

Condor uses metaprogramming to to turn the class *declaration* mechanism into a
blackboard-like environment to achieve the desired API. This approach helps us see
these mathematical models as datastructures that can then be transformed as needed to
automate the process that is typically performed manually for defining and evaluating
mathematical models in engineering analysis,

.. figure:: /images/math-model-process.png
   :width: 100%


Architecture
============

We followed modern pythonic best-practices and patterns to settle on a multi-layered
architecture like the Model-View-Controller paradigm in web development. The
three key components of the architecture are:

- The model layer, which provides an API for users to write their model. Condor models
  are ultimately a data structure which represents the represents the user's
  mathematical intent for the model.
- The backend layer provides a consistent interface to a third party *Computational
  Engine*, a symbolic-computational library which provides symbolic representation of
  *elements* and *operations* with awareness for basic differential calculus. The goal
  for the backend is provide a thin wrapper with a consistent interface so the
  computational engine implementation could be swapped out. Currently, we ship with
  CasADi as the only engine, although we hope to demonstrate a backend module for an
  alternate backend in the future.
- The implementation layer is the glue code that operates on the model data structure,
  using the backend to form the numerical functions needed to call the third-party
  solvers which implement the nuemrical algorithms of interest. The implementation
  layer then calls the solver and binds the results to the model instance.

.. figure:: /images/architecture.png
   :width: 50%


The Model Layer
================

Each user model is declared as a subclass of a *Model Template*, a ``class`` with a
``ModelType`` metaclass, which defines the *fields* from which *elements* are drawn to
define the model. Condor currently ships with 5 model templates:

+---------------------------+---------------+-----------------------+----------------------+
|                           |         fields                                               |
|                           +---------------+-----------------------+----------------------+
| built-in template         | input         | internal              | output               |
+===========================+===============+=======================+======================+
| ``ExplicitSystem``        | - input       |                       | - output             |
+---------------------------+---------------+-----------------------+----------------------+
| ``TableLookup``           | - input       | - input_data          | - output             |
|                           |               | - output_data         |                      |
+---------------------------+---------------+-----------------------+----------------------+
| ``AlgebraicSystem``       | - parameter   | - residual            | - variable           |
|                           |               |                       | - output             |
+---------------------------+---------------+-----------------------+----------------------+
| ``TrajectoryAnalysis``    | - parameter   | - state               | - trajectory_output  |
|                           |               | - modal.action        |                      |
+---------------------------+---------------+-----------------------+----------------------+
| ``OptimizationProblem``   | - parameter   | - objective           | - variable           |
|                           |               | - constraint          |                      |
+---------------------------+---------------+-----------------------+----------------------+

Models can be used recursively, building up more sophisticated models by *embedding*
models within another. However, system encapsolation is enforced so only elements from input and
output fields are accessible after the model has been defined. For example, we may
wish to optimize Sellar's algebraic system of equations. Mathematically, we can define
the optimization as

.. math::
   \begin{aligned}
   \operatorname*{minimize}_{x \in \mathbb{R}^3} &  &  & x_{2}^{2}+x_{1}+y_{1}+e^{-y_{2}} \\
   \text{subject to} &  &  & 3.16\le y_{1}\\
    &  &  & y_{2}\le24.0
   \end{aligned}

where :math:`y_1` and :math:`y_2` are the solution to the system of algebraic
equations described above. In condor, we can write this as

.. code-block:: python

    from condor import operators as ops
    class Sellar(co.OptimizationProblem):
        x = variable(shape=3, lower_bound=0, upper_bound=10)
        coupling = Coupling(x)
        y1, y2 = coupling

        objective = x[2]**2 + x[1] + y1 + ops.exp(-y2)
        constraint(y1 > 3.16)
        constraint(24. > y2)

As with the system of algebraic equations, we can numerically solve this optimization
problem by providing an initial value for the variables and instantiating the model.
The resulting object will have a dot-able data structure with the bound results,
including the embedded ``Coupling`` model:

.. code-block:: python

    Sellar.set_initial(x=[5,2,1])
    sellar = Sellar()
    print()
    print("objective value:", sellar.objective) # scalar value
    print(sellar.constraint) # field
    print(sellar.coupling.y1) # embedded-model element

The built-in model types provide a useful library to build small or one-off modeling capabilities.
We also ensured that there were good mechanisms for customizing models and creating new models to
address repeat and sophisticated modeling tasks.

Fields and Elements
-------------------

Fields contain the elements. Can be 

Metaprogramming class declaration
---------------------------------

..
    as clarified by Ionel: https://blog.ionelmc.ro/2015/02/09/understanding-python-metaclasses/#putting-it-all-together
    call order is:
      Meta.__prepare__ creates class dict
      process attributes of Class (class definition fills in class dict)
      Meta.__new__ creates class (via type.__call__) and returns Class
      (and Meta.__init__, but not as powerful, can do post-ops on constructed Class)

    note, similar construction for object, instance of Class:
      Meta.__call__ (classmethod, but located in Meta)
      Class.__new__ (classmethod)
      Class.__init__ (with instantiated class instance)

    note, inheritance is done ~ by checking bases (well, really MRO) if the attribute is not
    found on the leaf node. Can use this for Model's definition of __init__ etc for binding
    the IO to the model instance, but cannot rely on it for the magic name space injection

Ionel provides a nice overview of Python3 process for class declaration 
and object instantiation. Relevant for us is the following call-order. For 
class declaration,

1. :meth:`Metaclass.__prepare__` creates a class dictionary at the entry
   of the ``class`` declaration.
2. Each assignment within the class declaration uses the 
   :meth:`__setitem__` of the class dictionary
3. :func:`Metaclass.__new__` is passed the (filled) class dictionary and
   creates the class via :meth:`type.__call__`. Note that
   :meth:`Metaclass.__init__` is also called after this but is not as
   useful because the :code:`class` is already fully constructed by this point;
   the :code:`__init__` can only be used to organize post-processing. 

Is there any shared flow between a Template and Model? Yes, show Template first then Model.

:class:`ModelTemplate` declaration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Annotated flow of how a :class:`ModelTemplate` is created

:class:`Model` declaration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Annotated flow of how a :class:`Model` is created -- 


Calling and binding
-------------------

the :code:`__class__` is :meth:`__call__`\ed which calls the :meth:`__new__` which creates
the :code:`self` object which is then passed to :meth:`__init__`.

The :meth:`condor.Model.__init__` parses the positional and keyword arguments to bind
the values for the input field(s). Then the 

The implementation layer
========================

The implementation layer is responsible for using the backend to create
the numerical functions needed to evaluate model and call any solvers as
needed.

The embedded :class:`Options` class inside a model provides a name-space.
Attributes without a leading underscore are placed into a ``dict`` for
the keyword arguments to the implementation's :func:`construct`
method. Special behavior for ``_implementation`` and ...

Solver options can be passed from throught he Options attribute. Ultimately it is the
implementations job to parse the Options, but except where different solvers for the
same model-type conflict the intention is to make the argument manipulation at the
implementation layer as thin as possible.

the :attr:`Options` can be considered model inputs that make sense to have a default. They
are also intended to be inputs that don't define the mathematical meaning of the model. 

..
    #
    # In the case of the TableLookup, Options can be used to specify the boundary conditions
    # and interpolant degree (in each direction.) Options can be declared during the model
    # declaration, as in:


    class SinTable(condor.TableLookup):
        x = input()
        y = output()

        input_data[x] = np.linspace(-1, 1, 5)*np.pi
        output_data[y] = ops.sin(input_data[x])
        class Options:
            degrees = 0

    print(SinTable(np.pi/4))


    # %%
    # or by assigning an attribute directly on the Model's Option attribute, which will be
    # injected if it is not declared. For example, we can iterate over piecewise constant,
    # piecewise linear, and piecewise cubic polynomials to 





The backend
============

The backend layer provides a common interface to potential
"computational engine" libraries. Currently, we support only the
CasADi engine. Condor uses a "shim" so that the capability needed by the computational
engine can be accessed from the same import within the library. For each engine, a 
:mod:`backends` module must be provided to adapt the engine to a common API.


Using Condor for a "tool" or library
=====================================



Useful engineering analysis tools can be built as a Python library simply by
constructing the desired model witht he :mod:`contrib` models. Since the Model is
defined by constructing a :code:`class`, Python class variable scoping prevents the dynamic
definition of models inside a factory function. To get around this, a "configuration" pattern 
was defined with a :meth:`dynamic_link` helper. The Systems Analysis Office at NASA's Ames
Research Center has used this approach to build an aircraft synthesis and analysis tool using
Condor.

More recently, the metaprogramming back-bone of Condor was refactored to facilitate
the customization of symbolic processing to facilitate the creation of custom 
:class:`ModelTemplate`\s. To create a new type of analysis tool, we now recommend leveraging
this capability. A design process might include:

  1. Identify the data required to specify the analysis, and identify the :class:`Field` (or
     create a custom :class:`Field`) that would be appropriate for holding that data
  2. Identify (or create) what solver and implementation is needed, including a mapping
     from the new type of Model to the an existing model or solver.
  3. Implement a :meth:`process_placeholder` for processing the models data so the implementation
     can call the solver.

.. rubric:: References
.. [sellar] Sellar, R., Batill, S., and Renaud, J., "Response Surface Based, Concurrent Subspace Optimization for Multidisciplinary System Design," 1996. https://doi.org/10.2514/6.1996-714

..
    OLD
    ====
    The design of Condor was heavily inspired by Django. Some key design principles include:
     - Loose coupling
     - Do not Repeat Yourself
     - Explicit is better than Implicit
     - Do not reinvent the wheel

    The authors followed a process that could be called "example-driven development", writing the user code they would want to work and then implementing it.
    The goal of Condor was to automate and facilitate as many of the steps for numerical modeling as possible and to do so with an API that is as natural and expressive as possible.

    .. figure:: /images/math-model-process.png
       :width: 100%


    Like the Model-View-Controller paradigm in web development, the Condor architecture has 3 key components:

    1. The Condor model layer, which provides an API for users to write their model. Condor models are ultimately a data structure which represents the represents the user's mathematical intent for the model.

    2. The Computational Engine or Computational Backend, a symbolic-computational library which provides symbolic representation of *elements* and *operations* with awareness for basic differential calculus.

    3. The solvers, which implement the nuemrical algorithms of interest, and the implementaiton layer that which acts as glue code operating on the model data structure using the specific backend to form the numerical function callbacks which the solvers need.

    .. figure:: /images/architecture.png
       :width: 50%

    This loosly coupled approach allows any particular realization of each layer to be replaced. The computational engines and solver layers are generally external software, which greatly reduces the burden on the Condor team.

    Most users will focus on writing models using symbolic, declarative syntax that closely matches mathematical definitions.

    New algorithm development only requires implementation and solver layers object-oriented declarative syntax. Use previously-written models as test cases!

    Performance improvements (parallelization, compilation, etc) in back-end. Use previously-written models and algorithms to test. 
    Each layer can be tested and documented independently (or inherited), making it easier to maintain high-quality software products.


    The Model Layer
    ===============

    A *Model Template* is a ``class`` with a ``ModelType`` metaclass that defines the fields from which elements are drawn to define a model. Condor currently ships with 5 Model templates:

    User models are defined by writing a class that inherits from one of the Model Templates. Each template defines the *fields* from which the model *elements* are drawn. Models can be used recursively, building up more complex *embedding* models within another. However, system encapsolation is enforced so only elements from input and output fields are accessible after the model has been defined. For convenience, the ``AlgebraicSystem`` provides the ``output`` field for related computations; ``OptimizationSystem`` models can add related computations to the constraint field with (the default) +/- infinity values for the bounds.
    **TODO: should TableLookup get a similar convenience?**

    Each Model Template defines available *fields* from which *elements* are drawn to build up that model.


    +---------------------------+---------------+-----------------------+----------------------+
    |                           |         fields                                               |
    |                           +---------------+-----------------------+----------------------+
    | built-in template         | input         | internal              | output               |
    +===========================+===============+=======================+======================+
    | TrajectoryAnalysis        | - parameter   | - state               | - trajectory_output  |
    |                           |               | - modal.action        |                      |
    +---------------------------+---------------+-----------------------+----------------------+
    | OptimizationProblem       | - parameter   | - objective           | - variable           |
    |                           |               | - constraint          |                      |
    +---------------------------+---------------+-----------------------+----------------------+

    .. list-table:: Example table
       :header-rows: 1

       * - built-in template
         - input
         - internal
         - output
       * - TrajectoryAnalysis
         - 
             * parameter
         - 
             * state
             * dot
             * initial
             * modal.action
             * event.update
         - 
             * trajectory_output
       * - OptimizationProblem
         - 
             * parameter
         - 
             * objective
             * constraint
         - 
             * variable


    Models:
     - ExplicitSystem
     - ExternalSolverSystem
     - TableLookup
     - AlgebraicSystem
     - OptimizationProblem
     - ODESystem

    Metaprogramming is sometimes called "a solution looking for a problem" with advise to avoid using it. While there are some neat syntax sugar that can be implemented in either meta-programming or by other means, meta-programming is the ideal way to implement a domain specific language (DSL) since it provides enough hooks to modify the behavior sufficiently while keeping that modified syntax enclosed to a specific work area (the class definition).

    Inside a model declaration, the syntax has minimal boilerplate and allows for expressive mathematical declarations using any operations appropriate for the computational backend's, including calculus operations and the evaluation of other Condor models.


    Modeling Patterns
    ===================

    During the first 18 months of Condor's usage, several patterns have emerge; 

    For many optimizations, it is useful to create an analysis model, an ``ExplicitSystem`` that assembles all of the sub-models needed for the analysis to create a input field for the larger model. This analysis model is often useful to store 
    ** is this actually useful to say? And the next one should just get implemented





