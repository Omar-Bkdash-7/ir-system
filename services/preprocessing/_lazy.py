"""PEP 562 helper: a package exposes its public names without importing their (heavy) modules until first use.

Eager `__init__` re-exports in `core` and `processors` used to close an import
cycle (core.pipeline -> processors.tokenizer -> core.resources, and core.pipeline
-> processors.filters, which also needs core.resources). With lazy exports any
import path -- package root or submodule -- works regardless of entry order.
"""
from importlib import import_module
from typing import Any, Dict, List


def install_lazy_exports(package_globals: Dict[str, Any], package: str, exports: Dict[str, str]) -> None:
    """exports: public name -> fully qualified module that defines it."""

    def __getattr__(name: str) -> Any:
        target = exports.get(name)
        if target is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        value = getattr(import_module(target), name)
        package_globals[name] = value  # cache: the next access is a plain attribute lookup
        return value

    def __dir__() -> List[str]:
        return sorted(set(package_globals) | set(exports))

    package_globals["__getattr__"] = __getattr__
    package_globals["__dir__"] = __dir__
