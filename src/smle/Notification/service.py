from abc import ABC, abstractmethod

class Service(ABC):
    @abstractmethod
    def send_notification(self,message:str) -> None:
        pass
