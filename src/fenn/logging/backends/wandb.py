import os
from typing import Any, Callable, Dict, Optional

from fenn.secrets.keystore import KeyStore
try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None  # type: ignore

# ==========================================================
# WANDB BACKEND
# ==========================================================
class WandbBackend:
    def __init__(
        self,
        keystore: KeyStore,
        system_info: Callable[[str], None],
        system_warning: Callable[[str], None],
        system_exception: Callable[[str], None],
    ) -> None:
        self._keystore = keystore
        self._system_info = system_info
        self._system_warning = system_warning
        self._system_exception = system_exception
        self._run: Optional[Any] = None

    def start(self, args: Dict[str, Any]) -> None:
        if wandb is None:
            self._system_warning("wandb requested but wandb is not installed.")
            return

        os.environ["WANDB_SILENT"] = "true"
        wandb_conf = args.get("wandb", {}) or {}

        try:
            wandb_key = self._keystore.get_key("WANDB_API_KEY")
        except Exception as exc:
            self._system_exception("No valid WANDB API key provided in .env")
            raise RuntimeError("No valid WANDB API key provided in .env") from exc

        if not os.environ.get("WANDB_API_KEY"):
            os.environ["WANDB_API_KEY"] = wandb_key

        try:
            self._run = wandb.init(
                entity=wandb_conf.get("entity"),
                project=args.get("project"),
                config=args.get("training"),
                name=args.get("session_id"),
            )
            self._system_info("Wandb session initialized.")
        except Exception as exc:
            self._system_exception("Failed to start wandb session.")
            self._system_warning("Ensure internet connection is active.")
            raise RuntimeError(f"Failed to initialize wandb: {exc}") from exc

    def stop(self) -> None:
        # Either run.finish() or wandb.finish() are acceptable patterns;
        # keeping run reference avoids relying on global state.
        if self._run is not None:
            try:
                self._run.finish()
            finally:
                self._run = None

    @property
    def run(self) -> Optional[Any]:
        return self._run