"""A-share factor mining and backtesting model."""

__all__ = ["StackVM"]


def __getattr__(name):
    # Metadata validation must be usable before importing the torch runtime.
    if name == "StackVM":
        from .vm import StackVM
        return StackVM
    raise AttributeError(name)
