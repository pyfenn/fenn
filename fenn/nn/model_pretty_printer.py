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
    """Render a human-readable model summary for logs.

    Produces a tree-style architecture summary with parameter counts. Small
    models (module count ≤ ``small_model_threshold``) are printed in full;
    larger models are compacted to avoid overwhelming the log output.

    Args:
        model: The PyTorch module to summarise.
        small_model_threshold: Module count below which the full architecture
            is printed with no depth or child limits.
        compact_max_depth: Maximum nesting depth shown for large models.
        compact_max_children: Maximum number of children shown per module for
            large models.
        compact_max_lines: Maximum total lines in the rendered summary for
            large models.

    Example:
        >>> printer = ModelPrettyPrinter(my_model)
        >>> print(printer.render())
    """

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
        """Build and return the formatted model summary string.

        Returns:
            A multi-line string containing the model class name, parameter
            counts, and a tree view of the module hierarchy.
        """
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
        """Return the model summary string (delegates to :meth:`render`)."""
        return self.render()

    def _select_limits(self, module_count: int) -> _RenderLimits:
        """Choose render limits based on model size.

        Args:
            module_count: Total number of modules in the model.

        Returns:
            A :class:`_RenderLimits` with ``None`` fields (no limits) for
            small models, or the compact limits for larger ones.
        """
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
        """Recursively append child module lines to the summary.

        Args:
            lines: Accumulator list of rendered lines (mutated in place).
            children: Named children of the current module.
            prefix: Indentation string for the current nesting level.
            depth: Current recursion depth (1-indexed from the root).
            limits: Active render limits controlling depth, children,
                and line count.
        """
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
        """Format a single module's summary line.

        Includes the class name and, where non-zero, the direct parameter
        count, ``extra_repr`` string, and child count.

        Args:
            module: The module to format.
            root: If ``True``, child count is omitted (root is always
                expanded).

        Returns:
            A formatted string, e.g.
            ``"Linear (params=512, in_features=256, out_features=2)"``.
        """
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
        """Collapse and truncate a module's ``extra_repr`` to a single line.

        Joins multi-line repr strings into one line and truncates to 90
        characters with an ellipsis if longer.

        Args:
            extra_repr: Raw string from ``module.extra_repr()``.

        Returns:
            A collapsed single-line string, or an empty string if the input
            was empty.
        """
        if not extra_repr:
            return ""

        collapsed = " ".join(
            part.strip() for part in extra_repr.splitlines() if part.strip()
        )
        if len(collapsed) <= 90:
            return collapsed

        return f"{collapsed[:87]}..."
