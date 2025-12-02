from colorama import Fore, Style
from typing import Callable, Optional, Any

from fenn.args import Parser
from fenn.logging import Logger
from fenn.secrets.keystore import KeyStore

from fenn.utils import generate_haiku_id

class FENN:

    """
    The base FENN application
    """

    def __init__(self) -> None:

        self._session_id: str = generate_haiku_id()

        self._parser: Parser = Parser()
        self._keystore: KeyStore = KeyStore()
        self._logger: Logger = Logger()
        self._config_file: str = None

        self._entrypoint_fn: Optional[Callable] = None

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
        self._logger.system_log(
            "***********************************************************************************\n"
            f"{Style.BRIGHT}Hi, thank you for using the {Fore.GREEN}PyFenn{Style.RESET_ALL}{Style.BRIGHT} framework.{Style.RESET_ALL}\n"
            f"PyFenn is still in an {Fore.YELLOW}alpha version{Style.RESET_ALL}.\n"
            "If you find a bug or inconsistency, if you want to contribute or request a feature,\nplease open an issue at "
            f"{Fore.CYAN}https://github.com/pyfenn/fenn/issues{Style.RESET_ALL}\n"
            "Thank you for your support!\n"
            "***********************************************************************************\n"
        )

        if not self._entrypoint_fn:
            raise RuntimeError(f"{Fore.RED}[SMLE] No main function registered. {Fore.RED}[SMLE] Please use {Fore.LIGHTYELLOW_EX}@app.entrypoint{Fore.RED} to register your main function{Style.RESET_ALL}")

        self._parser.config_file = self._config_file if self._config_file != None else "fenn.yaml"
        self._args = self._parser.load_configuration()
        self._args["session_id"] = self._session_id

        self._logger.start()

        self._parser.print()

        try:
            # The execution of the decorated user function
            print(f"{Fore.GREEN}[SMLE] Application starting from {Fore.LIGHTYELLOW_EX}{self._entrypoint_fn.__name__}{Fore.GREEN} entrypoint.{Style.RESET_ALL}")

            result = self._entrypoint_fn(self._args)

            return result
        finally:
            self._logger.stop()

    @property
    def config_file(self) -> str:
        return self._config_file

    @config_file.setter
    def config_file(self, config_file: str) -> None:
        """
        The method to set the YAML file.
        """
        self._config_file = config_file
