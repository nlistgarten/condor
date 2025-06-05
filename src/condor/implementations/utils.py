def options_to_kwargs(new_cls):
    """Process a model clasa and create the kwarg dictionary for the :class:`Options`"""
    opts = getattr(new_cls, "Options", None)
    if opts is not None:
        backend_option = {
            k: v for k, v in opts.__dict__.items() if not k.startswith("_")
        }
    else:
        backend_option = {}
    return backend_option
