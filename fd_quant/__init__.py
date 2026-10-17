"""fd-quant 契约层：Rating 与 TargetWeights。

契约先行、计算后置：本层只定义形状与校验，不做任何计算、不联网、不读库。
校验纯本地且确定性——同输入两次结果完全一致（见 tests）。
"""
from fd_quant.contracts.models import (
    ContractError,
    Evidence,
    Producer,
    Rating,
    TargetWeights,
    validate,
    validate_rating,
    validate_target_weights,
)
from fd_quant.contracts.unsourced import unsourced_fields

__all__ = [
    "ContractError",
    "Evidence",
    "Producer",
    "Rating",
    "TargetWeights",
    "validate",
    "validate_rating",
    "validate_target_weights",
    "unsourced_fields",
]
