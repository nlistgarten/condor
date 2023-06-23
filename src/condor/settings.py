import importlib
settings = {}

def set_settings(new_settings):
    settings.update(new_settings)

def get_settings(defaults):
    return {k: settings.get(k, defaults[k]) for k in defaults}


class Settings:

    def __init__(self):

    def get_module(self, module, **kwargs):
        self.settings = kwargs
        mod = importlib.import_module(module)
        importlib.reload(module)

    def get_settings(self, defaults):
        return {k: self.settings.get(k, defaults[k]) for k in defaults}


settings = Settings()
