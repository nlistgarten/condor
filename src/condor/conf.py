"""Module configuration"""

import importlib
import sys


class Settings:
    """Configuration manager"""

    def __init__(self):
        # settings is a stack of dicts which should allow arbitrary nested deferred
        # modules
        self.settings = [{}]

    def get_module(self, module, **kwargs):
        """Load a module by path with specified options set

        Parameters
        ----------
        module : str
            Module path that would be used in an import statement, e.g.
            ``"my_package.module"``
        **kwargs
            Settings declared by `module` (see :meth:`get_settings`).

        Returns
        -------
        mod : module
            Configured module object.
        """
        self.settings.append(kwargs)

        if module not in sys.modules:
            mod = importlib.import_module(module)
        else:
            mod = sys.modules[module]
            mod = importlib.reload(mod)
        self.settings.pop()
        return mod

    def get_settings(self, **defaults):
        """Declare available settings with default values

        Parameters
        ----------
        **defaults
            Declared available settings with default values.

        Returns
        -------
        settings : dict
            Module configuration as set by :meth:`get_module`.
        """
        configured_kwargs = {k: self.settings[-1].get(k, defaults[k]) for k in defaults}
        extra_kwargs = {}
        for k, v in self.settings[-1].items():
            if k not in defaults:
                extra_kwargs[k] = v
        if extra_kwargs:
            # TODO warn instead?
            raise ValueError(
                "Extra keyword arguments were provided to configuration", extra_kwargs
            )

        return configured_kwargs


#: singleton :class:`Settings` instance
settings = Settings()
