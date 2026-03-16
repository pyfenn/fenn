from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch.nn as nn


@dataclass(frozen=True)
class _RenderLimits:
    max_depth: Optional[int]
    max_children: Optional[int]
    max_lines: Optional[int]


class ModelPrettyPrinter:
    """Render a readable model summary for logs."""

    def __init__(
        self,
        model: nn.Module,
        *,
        small_model_threshold: int = 25,
        compact_max_depth: int = 3,
        compact_max_children: int = 8,
        compact_max_lines: int = 80,
    ) -> None:
        self._model = model
        self._small_model_threshold = small_model_threshold
        self._compact_limits = _RenderLimits(
            max_depth=compact_max_depth,
            max_children=compact_max_children,
            max_lines=compact_max_lines,
        )

    def render(self) -> str:
        module_count = sum(1 for _ in self._model.modules())
        total_params = sum(param.numel() for param in self._model.parameters())
        trainable_params = sum(
            param.numel() for param in self._model.parameters() if param.requires_grad
        )
        frozen_params = total_params - trainable_params

        limits = self._select_limits(module_count)

        lines = [
            "Model Summary",
            "-------------",
            f"Class: {self._model.__class__.__name__}",
            f"Modules: {module_count}",
            (
                "Parameters: "
                f"total={total_params:,}, "
                f"trainable={trainable_params:,}, "
                f"frozen={frozen_params:,}"
            ),
            "",
            "Architecture:",
            self._format_module_header(self._model, root=True),
        ]

        children = list(self._model.named_children())
        self._append_children(
            lines=lines,
            children=children,
            prefix="",
            depth=1,
            limits=limits,
        )

        if limits.max_lines is not None and len(lines) >= limits.max_lines:
            omitted = max(module_count - (len(lines) - 6), 0)
            truncated_line = f"... output truncated ({omitted} modules omitted)"
            if omitted > 0 and lines[-1] != truncated_line:
                lines.append(truncated_line)

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()

    def _select_limits(self, module_count: int) -> _RenderLimits:
        if module_count <= self._small_model_threshold:
            return _RenderLimits(max_depth=None, max_children=None, max_lines=None)
        return self._compact_limits

    def _append_children(
        self,
        *,
        lines: list[str],
        children: list[tuple[str, nn.Module]],
        prefix: str,
        depth: int,
        limits: _RenderLimits,
    ) -> None:
        if not children:
            return

        if limits.max_depth is not None and depth > limits.max_depth:
            lines.append(f"{prefix}... {len(children)} nested modules omitted")
            return

        display_children = children
        hidden_children = 0
        if limits.max_children is not None and len(children) > limits.max_children:
            display_children = children[: limits.max_children]
            hidden_children = len(children) - len(display_children)

        for index, (name, module) in enumerate(display_children):
            if limits.max_lines is not None and len(lines) >= limits.max_lines:
                return

            lines.append(f"{prefix}{name}: {self._format_module_header(module)}")

            next_prefix = f"{prefix}  "
            self._append_children(
                lines=lines,
                children=list(module.named_children()),
                prefix=next_prefix,
                depth=depth + 1,
                limits=limits,
            )

        if hidden_children > 0 and (
            limits.max_lines is None or len(lines) < limits.max_lines
        ):
            lines.append(f"{prefix}... {hidden_children} more modules")

    def _format_module_header(self, module: nn.Module, *, root: bool = False) -> str:
        details = []
        direct_params = sum(param.numel() for param in module.parameters(recurse=False))
        if direct_params:
            details.append(f"params={direct_params:,}")

        extra = self._normalize_extra_repr(module.extra_repr())
        if extra:
            details.append(extra)

        child_count = sum(1 for _ in module.children())
        if child_count and not root:
            details.append(f"children={child_count}")

        if not details:
            return module.__class__.__name__

        return f"{module.__class__.__name__} ({', '.join(details)})"

    @staticmethod
    def _normalize_extra_repr(extra_repr: str) -> str:
        if not extra_repr:
            return ""

        collapsed = " ".join(part.strip() for part in extra_repr.splitlines() if part.strip())
        if len(collapsed) <= 90:
            return collapsed

        return f"{collapsed[:87]}..."
