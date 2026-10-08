# hooks.py
import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

class HookManager:
    def __init__(self):
        self._handles: list[RemovableHandle] = []
        self._module_names: dict[nn.Module, str] = {}

    def attach(self, model: nn.Module) -> None:
        # To prevent hook duplications
        self.detach()

        for name, module in model.named_modules():
            # Only profiling leaf modules
            if any(module.children()):
                continue

            self._module_names[module] = name

            pre_handle = module.register_forward_pre_hook(
                self._pre_forward_hook
            )

            post_handle = module.register_forward_hook(
                self._forward_hook,
                always_call=True
            )

            self._handles.extend([pre_handle, post_handle])


    def _pre_forward_hook(self, module: nn.Module, inputs: tuple) -> None:
        name = self._module_names[module]
        print(f"[START] {name}: {type(module).__name__}")

    def _forward_hook(self, module: nn.Module, inputs: tuple, output) -> None:
        name = self._module_names[module]
        print(f"[END] {name}: {type(module).__name__}")

    def detach(self):
        for handle in self._handles:
            handle.remove()

        self._handles.clear()
        self._module_names.clear()