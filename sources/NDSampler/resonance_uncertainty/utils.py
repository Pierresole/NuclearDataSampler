from importlib import reload
import types

def recursive_reload(module):
    """Recursively reloads all submodules of the given module."""
    if hasattr(module, "__all__"):
        for submodule_name in module.__all__:
            submodule = getattr(module, submodule_name, None)
            if isinstance(submodule, types.ModuleType):  # Check if it's an actual module
                recursive_reload(submodule)
    reload(module)
