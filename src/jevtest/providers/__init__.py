"""Provider 抽象层：四臂统一接口（PRD §5.1 条款 2 输出同构）。

契约冻结：接口形状以 contracts/triage-result.schema.json 与 jevtest.models 为准，
本包不得引入契约之外的字段或语义。
"""

from jevtest.providers.base import InvalidProviderOutput, Provider
from jevtest.providers.fallback import sanitize
from jevtest.providers.jev_mock import JevMockProvider
from jevtest.providers.rule import RuleProvider

__all__ = [
    "InvalidProviderOutput",
    "Provider",
    "JevMockProvider",
    "RuleProvider",
    "sanitize",
]
