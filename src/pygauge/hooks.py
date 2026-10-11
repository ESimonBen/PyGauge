# hooks.py
from torch import nn
from typing import Callable, Optional
from torch.utils.hooks import RemovableHandle

from pygauge.enums import HookMode
from pygauge.events import ModuleEvent

# Is not exactly thread-safe for concurrent model training.
class HookManager:
    def __init__(self, on_start: Optional[Callable[[ModuleEvent], None]] = None, on_end: Optional[Callable[[ModuleEvent], None]] = None,
                 mode: HookMode = HookMode.LEAF, module_filter: Optional[Callable[[str, nn.Module], bool]] = None):
        if mode == HookMode.FILTERED and module_filter is None:
            raise ValueError("FILTERED mode requires a module_filter")

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

        if self.mode == HookMode.FILTERED and self.module_filter is None:
            raise ValueError("FILTERED mode requires a module_filter")

        self._invocation_counter = 0

        try:
            for name, module in model.named_modules():
                if self.mode == HookMode.LEAF:
                    if any(module.children()):
                        continue
                elif self.mode == HookMode.FILTERED:
                    if self.module_filter and not self.module_filter(name, module):
                        continue
                elif self.mode != HookMode.ALL:
                    raise ValueError(f"Unsupported hook mode: {self.mode}")

                self._module_names[module] = name
                self._active_stacks[module] = []

                pre_handle = module.register_forward_pre_hook(
                    self._pre_forward_hook
                )

                self._handles.append(pre_handle)

                post_handle = module.register_forward_hook(
                    self._forward_hook,
                    always_call=True
                )

                self._handles.append(post_handle)

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

        stack = self._active_stacks[module]
        stack.append(event)

        if self.on_start:
            try:
                self.on_start(event)
            except Exception:
                stack.pop()
                raise

    def _forward_hook(self, module: nn.Module, inputs: tuple, output) -> None:
        stack = self._active_stacks.get(module)

        if not stack:
            return

        event = stack.pop()

        if self.on_end:
            self.on_end(event)

    def detach(self):
        for handle in self._handles:
            handle.remove()

        self._handles.clear()
        self._module_names.clear()
        self._active_stacks.clear()
        self._invocation_counter = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.detach()