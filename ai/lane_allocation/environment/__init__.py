from typing import Any, Callable, Optional

from .highway import HighwayEnv as _HighwayEnv
from .vehicle import Vehicle

load_rl_env: Callable[[Optional[dict[str, Any]]], _HighwayEnv] = (
    lambda config: _HighwayEnv(config)
)

__all__ = ["Vehicle", "load_rl_env"]
