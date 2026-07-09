"""
RouteOS: Adaptive Small Language Model Inference Engine.

Implements TriRoute-inspired joint routing for attention, MoE experts,
and KV-cache allocation at token granularity.
"""

from routeos.engine import RouteOSEngine
from routeos.router import TokenRouter
from routeos.moe import MixtureOfExperts
from routeos.kv_cache import KVCacheManager

__version__ = "0.1.0"
__all__ = ["RouteOSEngine", "TokenRouter", "MixtureOfExperts", "KVCacheManager"]
