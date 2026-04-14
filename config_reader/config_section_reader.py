from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar

T = TypeVar("T")

class ConfigSectionReader(ABC, Generic[T]):
    """Base class for config section readers.

    Subclasses implement `read_section(raw) -> T` and are pure parsers: they
    must not perform file I/O or path resolution. Path/file handling is the
    responsibility of the top-level config reader which reads the YAML once
    and forwards the raw mappings to section readers.
    """

    @abstractmethod
    def read_section(self, raw: Dict[str, Any]) -> T:
        """Parse and validate the relevant section from `raw` and
        return the concrete dataclass/config object.
        """

    def require_mapping(self, parent: Dict[str, Any], key: str) -> Dict[str, Any]:
        value = parent.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"Missing or invalid mapping: '{key}'")
        return value

    def require_value(self, parent: Dict[str, Any], key: str) -> Any:
        if key not in parent:
            raise ValueError(f"Missing required config key: '{key}'")
        return parent[key]

    def optional_value(self, parent: Dict[str, Any], key: str, default: Optional[Any] = None) -> Any:
        return parent.get(key, default)

    def optional_list(self, parent: Dict[str, Any], key: str) -> Optional[list]:
        v = parent.get(key)
        if v is None:
            return None
        if not isinstance(v, (list, tuple)):
            raise ValueError(f"Config key '{key}' must be a list")
        return list(v)
