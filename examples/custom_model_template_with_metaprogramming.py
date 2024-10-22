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
