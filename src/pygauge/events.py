# events.py
from torch import nn
from dataclasses import dataclass

@dataclass(frozen=True)
class ModuleEvent:
    module: nn.Module
    name: str
    module_type: str
    invocation_id: int