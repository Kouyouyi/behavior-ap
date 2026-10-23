"""Standalone simulation; no planner, model client, or inference dependency."""

__version__ = "0.1.0"


def __getattr__(name):
    if name == "RadioEnvironment":
        from .environment import RadioEnvironment
        return RadioEnvironment
    raise AttributeError(name)


__all__ = ["RadioEnvironment"]
