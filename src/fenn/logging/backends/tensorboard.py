from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore


class TensorboardBackend:
    def __init__(
        self,
        system_info: Callable[[str], None],
        system_warning: Callable[[str], None],
        system_exception: Callable[[str], None],
    ) -> None:
        self._system_info = system_info
        self._system_warning = system_warning
        self._system_exception = system_exception
        self._writer: Optional[Any] = None

    def start(self, args: Dict[str, Any]) -> None:
        if SummaryWriter is None:
            self._system_warning(
                "TensorBoard requested but torch is not installed or SummaryWriter not available."
            )
            return

        tb_conf = args.get("tensorboard", {}) or {}
        base_dir = tb_conf.get("dir", args["logger"]["dir"])
        tb_log_dir = Path(base_dir) / args["project"] / args["session_id"]

        try:
            self._writer = SummaryWriter(log_dir=str(tb_log_dir))
            self._system_info(f"TensorBoard writer initialized at {tb_log_dir}")
        except Exception as exc:
            self._system_exception(f"Failed to initialize TensorBoard: {exc}")

    def stop(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            finally:
                self._writer = None

    @property
    def writer(self) -> Optional[Any]:
        return self._writer