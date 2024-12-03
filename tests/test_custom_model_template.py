import condor as co


def test_defaults():
    class CustomModelType(co.ModelType):
        pass

    class CustomModelTemplate(co.ModelTemplate, model_metaclass=CustomModelType):
        pass

    class MyModel(CustomModelTemplate):
        pass


def test_new_kwarg():
    class CustomModelType(co.ModelType):
        def __new__(cls, *args, new_kwarg=None, **kwargs):
            assert new_kwarg == "handle a string"

    class CustomModelTemplate(co.ModelTemplate, model_metaclass=CustomModelType):
        pass

    class MyModel(CustomModelTemplate, new_kwarg="handle a string"):
        pass
