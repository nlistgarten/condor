import importlib
settings = {}

def set_settings(new_settings):
    settings.update(new_settings)

def get_settings(defaults):
    return {k: settings.get(k, defaults[k]) for k in defaults}


