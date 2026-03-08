from abc import ABC, abstractmethod

class baseLogger(ABC):
    @abstractmethod
    def system_info(self, message: str) -> None:
        pass

    @abstractmethod
    def system_warning(self, message: str) -> None:
        pass

    @abstractmethod
    def system_exception(self, message: str) -> None:
        pass

    @abstractmethod
    def user_info(self, message: str) -> None:
        pass

    @abstractmethod
    def user_warning(self, message: str) -> None:
        pass

    @abstractmethod
    def user_exception(self, message: str) -> None:
        pass