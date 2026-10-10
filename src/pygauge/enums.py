# enums.py
from enum import Enum, auto

class HookMode(Enum):
    LEAF = auto()
    ALL = auto()
    FILTERED = auto()