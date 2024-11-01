import condor as co

class CustomMetaprogrammedType(co.ModelType):
    @classmethod
    def process_placeholders(cls, new_cls, attrs):
        print(f"CustomMetaprogrammedType.processs_placeholders for {new_cls} is a good place for manipulating substitutions")

    @classmethod
    def __prepare__(
        cls, *args,
        new_kwarg=None,
        **kwargs
    ):
        """
        custom processing behavior use cases:
        - condor-flight cross-substitution of vehicle state to environment models
        (atmosphere/wind/gravity/etc)
        - lincov generation of augmented covariance state, propagation/correction/update
          equations, etc.


        catching custom metaclass arguments use cases:
        big gascon: component inputs and sub-assembly declaration (like a primary model)
        then singleton submodels for "outputs", but these are not condor built-in
        submodels, gascon will define a template for each:
         - weight with "subtotal" placheolder
         - aero with lift-curve slope, equivalent wetted area, etc, placeholder +
           potentially dynamic inputs?
         - propulsion with dynamic inputs (maybe base aircraft provides all the core
           state plus flight conditions, etc.) and placeholder for thrust, incidence
           angle, and state field for adding new state
        so primary component model metaclass will have to have a slot for these gascon
        output submodels
        so the gascon output submodel will at least bind to the primary model template
        its when the primary is inserted into a parent assembly that the output
        submodels really get processed, and then combined into the parents as needed?
        will the assembly user model

        class Component(ComponentTemplate):
            input = field()

        class Weight(GasconComponentOutput):
            subtotal = placeholder()

        class Aero(GasconComponentOutput):
            lift_curve_slope = placeholder()
            flat_plate_equivalent_area = placeholder()

        class Propulsion(GasconComponentOutput):
            state = field()
            thrust_lbf = placeholder()
            incidence_angle = placeholder()


        # this is a bad example since we will use a custom metaclass for lots of the
        # propulsion models for the engine decks...
        class Turbofan(Component):
            fan_diameter_ft = input()
            sls_airflow = input()
            # require location?

        class Performance(Turbofan):
            thrust = ...

        class EngineOutDrag(Turbofan):
            drag_lbf = ...

        # I kinda like this syntax, use the name of the "subclass" to figure out which
        # template to use?






        """
        print(f"CustomMetaprogrammedType.__prepare__ with new_kwarg={new_kwarg}")
        return super().__prepare__(*args, **kwargs)

    def __new__(
        cls, *args,
        new_kwarg=None,
        **kwargs
    ):
        print(f"CustomMetaprogrammedType.__new__ with new_kwarg={new_kwarg}")
        new_cls = super().__new__(cls, *args, **kwargs)
        return new_cls

class CustomMetaprogrammed(co.ModelTemplate, model_metaclass=CustomMetaprogrammedType):
    pass

class MyModel0(CustomMetaprogrammed):
    pass

class MyModel1(CustomMetaprogrammed, new_kwarg="handle a string"):
    pass


class ModelsCouldAlwaysTakeNonCondorInputs(co.ExplicitSystem):
    x = input()
    output.y = x**2

    def __init__(self, *args, my_kwarg=None, **kwargs):
        print("Use for something like ReferenceFrame, it's also possible to modify the kwargs ")
        # store frame state as condor elements, then can use normal python type logic to
        # compose bigger expressions like finding path from arbitrary frames -- but
        # would need to do something to make sure that the extra kwargs were saved and
        # re-bound as needed... or maybe it's okay if they're not? the final expression
        # which will get used with the bound thing will be right? not sure.

        # it's also possible to manipulate the (k)wargs that get passed to super based
        # on my_kwarg -- is that a way this could work?
        super().__init__(*args, **kwargs)

ModelsCouldAlwaysTakeNonCondorInputs(1.2)


