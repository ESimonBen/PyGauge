# enums.py
from enum import Enum, auto

class JoinPolicy(Enum):
    STRICT = auto()
    TIMING_LEFT = auto()
    OUTER = auto()