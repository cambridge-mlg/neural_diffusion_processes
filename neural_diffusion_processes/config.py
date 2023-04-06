import contextlib
from dataclasses import dataclass, replace
from typing import Any, Dict, Generator, List, Mapping, Optional, Union

__config: Optional["Config"] = None


@dataclass(frozen=True)
class Config:
    jitter: float = 1e-6
    """
    Jitter value. Mainly used for for making badly conditioned matrices more stable.
    """


def get_config() -> Config:
    """Returns current active config."""
    assert __config is not None, "__config is None. This should never happen."
    return __config


def set_config(new_config: Config) -> None:
    """Update GPflow config with new settings from `new_config`."""
    global __config
    __config = new_config


@contextlib.contextmanager
def as_context(temporary_config: Optional[Config] = None) -> Generator[None, None, None]:
    """Ensure that global configs defaults, with a context manager. Useful for testing."""
    current_config = config()
    temporary_config = replace(current_config) if temporary_config is None else temporary_config
    try:
        set_config(temporary_config)
        yield
    finally:
        set_config(current_config)


# Set global config.
set_config(Config())