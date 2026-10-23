"""Standalone simulation; no planner, model client, or inference dependency."""

__version__ = "0.2.0"


def __getattr__(name):
    if name in ("BehaviorEnvironment", "RadioEnvironment"):
        from .environment import BehaviorEnvironment, RadioEnvironment
        return {"BehaviorEnvironment": BehaviorEnvironment, "RadioEnvironment": RadioEnvironment}[name]
    raise AttributeError(name)


__all__ = ["BehaviorEnvironment", "RadioEnvironment"]
