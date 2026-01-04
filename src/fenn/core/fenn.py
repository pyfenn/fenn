from colorama import Fore, Style
from typing import Callable, Optional, Any

from fenn.args import Parser
from fenn.logging import Logger
from fenn.secrets.keystore import KeyStore
from fenn.utils import generate_haiku_id


class Fenn:
    """
    The base Fenn application
    """

    def __init__(self) -> None:

        self._session_id: str = generate_haiku_id()

        self._parser: Parser = Parser()
        self._keystore: KeyStore = KeyStore()
        self._logger: Logger = Logger()

        # DISCLAIMER:
        # This class is the base class for all Fenn applications.
        # It is designed to be subclassed, not instantiated directly.
        # Please do not modify this class unless you know what you are doing.
        self._config_file: str = None

        self._entrypoint_fn: Optional[Callable] = None

        self._disable_disclaimer = False

    def entrypoint(self, entrypoint_fn: Callable) -> Callable:
        """
        The decorator to register the main execution function.
        """
        self._entrypoint_fn = entrypoint_fn
        return entrypoint_fn

    def run(self) -> Any:
        """
        The method that executes the application's core logic.
        """

        if not self._disable_disclaimer:
            self._logger._logging_backend._original_print(
                "***********************************************************************************\n"
                f"{Style.BRIGHT}Hi, thank you for using the {Fore.GREEN}PyFenn{Style.RESET_ALL}{Style.BRIGHT} framework.{Style.RESET_ALL}\n"
                f"PyFenn is still in an {Fore.CYAN}alpha version{Style.RESET_ALL}.\n"
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

        # Load config
        self._parser.config_file = (
            self._config_file if self._config_file is not None else "fenn.yaml"
        )
        self._args = self._parser.load_configuration()
        self._args["session_id"] = self._session_id

        # Start logging
        self._logger.start()

        # Print parsed config (user logs)
        self._parser.print()

        try:
            # System startup message
            self._logger.system_info(
                f"Application starting from entrypoint: {self._entrypoint_fn.__name__}"
            )

            # Execute user function
            result = self._entrypoint_fn(self._args)
            return result

        finally:
            self._logger.stop()

    def disable_disclaimer(self) -> None:
        self._disable_disclaimer = True

    def set_config_file(self, config_file: str) -> None:
        """
        The method to set the YAML file.
        """
        self._config_file = config_file

    @property
    def config_file(self) -> str:
        return self._config_file
