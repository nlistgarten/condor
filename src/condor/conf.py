import importlib
settings = {}

class Settings:
    def __init__(self):
        # settings is a stack of dicts which should allow arbitrary nested deferred
        # modules
        self.settings = []

    def get_module(self, module, **kwargs):
        """
        module is a string such that `import {module}` would work from calling file
        kwargs are settings to use
        """
        self.settings.append(kwargs)
        mod = importlib.import_module(module)
        importlib.reload(mod)
        self.settings.pop()
        return mod

    def get_settings(self, defaults):
        # TODO: defensive check on unused self.settings? Or maybe it's a feature to
        # allow unused settings so a big dict can be used for a project

        return {
            k: self.settings[-1].get(k, defaults[k])
            for k in defaults
        }


settings = Settings()
