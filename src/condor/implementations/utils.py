
def options_to_kwargs(new_cls):
    """ Process a model clasa and create the kwarg dictionary for the :class:`Options`
    """
    Options = getattr(new_cls, "Options", None)
    if Options is not None:
        backend_option = {
            k: v for k, v in Options.__dict__.items() if not k.startswith("_")
        }
    else:
        backend_option = {}
    return backend_option


