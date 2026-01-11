# Generic class to define any splitting algorithm

from abc import ABC, abstractmethod
from typing import List, Any, Tuple

class SplitAlgo(ABC):
    """Algorithm-specific preprocessing e.g. converting to a graph. Default: no preprocessing"""
    def preprocess(self, model: Any) -> Any:
        return model
    
    @abstractmethod
    def find_split_points(self, model: Any) -> Any:#List[Tuple[Any, str]], Any:
        pass
