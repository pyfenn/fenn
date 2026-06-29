from pathlib import Path
from typing import Any, Callable, Optional

from colorama import Fore, Style

from fenn.exporter import Exporter
from fenn.keystore import KeyStore
from fenn.logging import logger, original_print, redirect_prints, restore_prints
from fenn.parser import Parser
from fenn.reproducibility import generate_session_id


class Fenn:
    """Base class for a PyFenn application.

    Wires together argument parsing, logging, secret storage, and result
    export into a single entry point. Subclass this or instantiate it
    directly, register your main logic with :meth:`entrypoint`, then call
    :meth:`run`.

    Example:
        >>> app = Fenn()
        >>> @app.entrypoint
        ... def main(args):
        ...     print(args)
        >>> app.run()
    """

    def __init__(self) -> None:

        self._session_id: str = generate_session_id()

        self._parser: Parser = Parser()
        self._keystore: KeyStore = KeyStore()
        self._exporter: Exporter = Exporter()

        # DISCLAIMER:
        # This class is the base class for all Fenn applications.
        # It is designed to be subclassed, not instantiated directly.
        # Please do not modify this class unless you know what you are doing.
        self._config_file: str = None

        self._entrypoint_fn: Optional[Callable] = None

        self._disable_disclaimer = False

        self._load_configuration()

    def _load_configuration(self):
        # Load config
        self._parser.config_file = (
            self._config_file if self._config_file is not None else "fenn.yaml"
        )
        self._args = self._parser.load_configuration()
        self._args["session_id"] = self._session_id

    def entrypoint(self, entrypoint_fn: Callable) -> Callable:
        """Register the application's main function.

        Use as a decorator. The decorated function receives the parsed
        configuration dict as its only argument and may return any value,
        which :meth:`run` will forward to the caller.

        Args:
            entrypoint_fn: The callable to register as the application entry
                point. Must accept a single ``args`` dict argument.

        Returns:
            The original callable, unchanged (decorator passthrough).
        """
        self._entrypoint_fn = entrypoint_fn
        return entrypoint_fn

    def run(self) -> Any:
        """Execute the application.

        Loads configuration from the YAML file, initialises logging and
        export directories, then calls the registered entrypoint function.
        Logging is always stopped cleanly, even if the entrypoint raises.

        Returns:
            The value returned by the entrypoint function.

        Raises:
            RuntimeError: If no entrypoint has been registered via
                :meth:`entrypoint`.
        """

        if not self._disable_disclaimer:
            original_print(
                "***********************************************************************************\n"
                f"{Style.BRIGHT}Hi, thank you for using the {Fore.GREEN}PyFenn{Style.RESET_ALL}{Style.BRIGHT} framework.{Style.RESET_ALL}\n"
                f"PyFenn is still in {Fore.CYAN}early access{Style.RESET_ALL}.\n"
                "If you find a bug or inconsistency, if you want to contribute or request a feature,\nplease open an issue at "
                f"{Fore.CYAN}https://github.com/pyfenn/fenn/issues{Style.RESET_ALL}.\n"
                f"{Style.BRIGHT}Thank you for your support!{Style.RESET_ALL}\n"
                f"{Fore.LIGHTYELLOW_EX}Use app.disable_disclaimer() to stop seeing this message.{Style.RESET_ALL}\n"
                "***********************************************************************************\n"
            )

        if not self._entrypoint_fn:
            raise RuntimeError(
                f"{Fore.RED}[EXCEPTION] No main function registered. "
                f"Please use {Fore.LIGHTYELLOW_EX}@app.entrypoint{Style.RESET_ALL} "
                "to register your main function."
            )

        redirect_prints()
        try:
            Exporter().configure(self._args)

            # Print parsed config (user logs)
            self._parser.print()

            # System startup message
            logger.info(
                f"Application starting from entrypoint: {self._entrypoint_fn.__name__}"
            )

            # Execute user function
            result = self._entrypoint_fn(self._args)
            return result

        finally:
            restore_prints()
            logger.close()

    def disable_disclaimer(self) -> None:
        """Suppress the startup banner printed before each run."""
        self._disable_disclaimer = True

    def set_config_file(self, config_file: str) -> None:
        """Override the default YAML configuration file path.

        Args:
            config_file: Path to the YAML file to load on :meth:`run`.
                Defaults to ``"fenn.yaml"`` in the working directory.
        """
        self._config_file = config_file
        self._load_configuration()

    def get_environ(self, key: str) -> str:
        return self._keystore.get_key(key)

    @property
    def parameters(self) -> map:
        """"""
        return self._args

    @property
    def config_file(self) -> str:
        """Path to the active YAML configuration file."""
        return self._config_file

    @property
    def export_dir(self) -> Path:
        """Output directory for run artefacts (logs, models, exports)."""
        return self._exporter.export_dir
