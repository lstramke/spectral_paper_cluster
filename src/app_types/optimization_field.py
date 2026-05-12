from dataclasses import dataclass
from typing import Generic, Type, TypeVar

T = TypeVar("T", int, float)

@dataclass(slots=True)
class OptimizationField(Generic[T]):
    name: str                   
    min_value: T      
    max_value: T
    value_type: Type[T]