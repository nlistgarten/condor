import importlib
import sys
settings = {}

class Settings:
    def __init__(self):
        # settings is a stack of dicts which should allow arbitrary nested deferred
        # modules
        self.settings = [{}]

    def get_module(self, module, **kwargs):
        """
        module is a string such that `import {module}` would work from calling file
        kwargs are settings to use
        """
        self.settings.append(kwargs)
        print(kwargs)

        if module not in sys.modules:
            mod = importlib.import_module(module)
        else:
            mod = sys.modules[module]
            mod = importlib.reload(mod)
        self.settings.pop()
        return mod

    def get_settings(self, *args, **defaults):
        if args and defaults or len(args) > 1:
            raise ValueError
        if args:
            defaults = args[0]

        # TODO: defensive check on unused self.settings? Or maybe it's a feature to
        # allow unused settings so a big dict can be used for a project

        return {
            k: self.settings[-1].get(k, defaults[k])
            for k in defaults
        }


settings = Settings()
