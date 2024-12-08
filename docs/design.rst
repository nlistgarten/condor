=================
Design Principles
=================

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





