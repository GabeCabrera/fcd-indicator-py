"""BTC-specific regime configuration for Mode D (Drift) and Mode F (Becoming) classification.

This module defines threshold parameters that classify market states based on FCD metrics:
- Drift: High coherence, low TV product (trending markets)
- Becoming: Moderate coherence and TV (transition states)
- Chop: Everything else (ranging/noisy markets)
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class BTCRegimeConfig:
    """Configuration for BTC regime classification.
    
    Attributes:
        COH_DRIFT_HIGH: Minimum coherence threshold for drift regime (trending)
        COH_BECO_MIN: Minimum coherence for becoming regime (transition)
        COH_BECO_MAX: Maximum coherence for becoming regime (transition)
        TV_LOW_MAX: Maximum TV product for drift regime (low noise)
        TV_BECO_MIN: Minimum TV product for becoming regime
        TV_BECO_MAX: Maximum TV product for becoming regime
        ALLOWED_VOL_DRIFT: Volatility regimes allowed during drift
        ALLOWED_VOL_BECO: Volatility regimes allowed during becoming
    """
    COH_DRIFT_HIGH: float = 0.95
    COH_BECO_MIN: float = 0.60
    COH_BECO_MAX: float = 0.90
    TV_LOW_MAX: float = 0.02
    TV_BECO_MIN: float = 0.02
    TV_BECO_MAX: float = 0.40
    ALLOWED_VOL_DRIFT: List[str] = field(default_factory=lambda: ["low", "medium"])
    ALLOWED_VOL_BECO: List[str] = field(default_factory=lambda: ["medium", "high"])


# Global BTC configuration instance
BTC_CONFIG = BTCRegimeConfig()
