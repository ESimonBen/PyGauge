# hooks.py
from torch import nn
from typing import Callable, Optional
from torch.utils.hooks import RemovableHandle

from pygauge.enums import HookMode
from pygauge.events import ModuleEvent

class HookManager:
    def __init__(self, on_start: Optional[Callable[[ModuleEvent], None]] = None, on_end: Optional[Callable[[ModuleEvent], None]] = None,
                 mode: HookMode = HookMode.LEAF, module_filter: Optional[Callable[[str, nn.Module], bool]] = None):
        self.on_start = on_start
        self.on_end = on_end
        self.mode = mode
        self.module_filter = module_filter

        self._handles: list[RemovableHandle] = []
        self._module_names: dict[nn.Module, str] = {}
        self._active_stacks: dict[nn.Module, list[ModuleEvent]] = {}
        self._invocation_counter = 0

    def attach(self, model: nn.Module) -> None:
        if self.is_attached():
            self.detach()

        self._invocation_counter = 0

        try:
            for name, module in model.named_modules():
                if self.mode == HookMode.LEAF:
                    if any(module.children()):
                        continue
                elif self.mode == HookMode.FILTERED:
                    if self.module_filter and not self.module_filter(name, module):
                        continue
                elif self.mode == HookMode.ALL:
                    pass

                self._module_names[module] = name
                self._active_stacks[module] = []

                pre_handle = module.register_forward_pre_hook(
                    lambda m, inputs, mod=module: self._pre_forward_hook(mod, inputs)
                )

                post_handle = module.register_forward_hook(
                    lambda m, inputs, output, mod=module: self._forward_hook(mod, inputs, output),
                    always_call=True
                )

                self._handles.extend([pre_handle, post_handle])
        except Exception:
            self.detach()
            raise

    def is_attached(self):
        return len(self._handles) > 0

    def _pre_forward_hook(self, module: nn.Module, inputs: tuple) -> None:
        name = self._module_names.get(module, "")
        self._invocation_counter += 1

        event = ModuleEvent(
            module=module,
            name=name,
            module_type=type(module).__name__,
            invocation_id=self._invocation_counter
        )

        if module not in self._active_stacks:
            self._active_stacks[module] = []

        self._active_stacks[module].append(event)

        if self.on_start:
            try:
                self.on_start(event)
            except Exception:
                if self._active_stacks[module]:
                    self._active_stacks[module].pop()
                raise

    def _forward_hook(self, module: nn.Module, inputs: tuple, output) -> None:
        stack = self._active_stacks.get(module)

        if stack:
            event = stack.pop()
        else:
            name = self._module_names.get(module, "")
            event = ModuleEvent(
                module=module,
                name=name,
                module_type=type(module).__name__,
                invocation_id=0
            )

        if self.on_end:
            try:
                self.on_end(event)
            except Exception:
                raise

    def detach(self):
        for handle in self._handles:
            handle.remove()

        self._handles.clear()
        self._module_names.clear()
        self._active_stacks.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.detach()