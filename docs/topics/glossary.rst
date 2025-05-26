=======================
Glossary of Terms
=======================

Condor Specific Terms
======================

.. glossary::

    computational engine
      the external library that is used to perform symbolic differentiation and generate numeric functions

    backend
      The Condor "shim" submodule that provides a consistent interface to any supported computational engine

    ``backend_repr``
      an expression representable by the computational engine

    element
      a wrapper around a ``backend_repr`` that includes relevant metadata such as shape and name

    field
      organization of elements according to purpose, e.g. variable field on Optimization Problem models.

    model
      mathematical representation of an engineering system that the user writes by using Condor's mathematical
      domain-specific language in Python's class declaration format

    solver
      a piece of software "code" that can evaluate a type of model represented in a particular canonical form

    implementation
      a class that uses the backend (1) to transform the model as written by the engineer into the canonical form expected
      by the solver, call the solver, and transform the solver results back into the form of the model, and (2) provide
      the symbolic metadata for differentiation rules back to the computational engine

    embedded model
       a model being evaluated as part of another model definition

    model instance
       an evaluation of the model with the specified input values; may be symbolic when embedded into another model

    model template
      a class that defines what fields and placeholder values a particular model type can use; a model subclasses a template

    model metaclass
      a metaclass that processes the model declaration so it can be evalauted into a model instance

    model metadata class
      a dataclass for holding metadata for a model

    placeholder
      a field provided to model templates to define singleton reserved word values, like ``t0`` and ``tf`` for a trajectory
      or the ``objective`` for an optimization problem


General Object-Oriented and Metaprogramming Terms
=====================================================

.. glossasry::

   base
      A relatively complete class to inherit from; inheritors will generally make behavior more specific by overwriting methods.
      Inheritors can re-use base's methods in python by using super()

   mixin
      A class that provides specific behavior, an inheritor may also inherit from other mixins and even a "base" to maximize 
      code reuse

   type
      The class of a class (i.e., a class is an object of type, type); used as a suffix for a metaclass

   metaclass
      The class of a particular class, does name space preparation before user's class declaration and processing at closure




