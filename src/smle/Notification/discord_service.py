import requests
import os
from service import *

class Discord(Service):
    def __init__(self):
        self._discord_webhook_url = os.getenv("DISCORD_WEBHOOK")
        if not self._discord_webhook_url:
            raise ValueError("DISCORD_WEBHOOK environment variable not set")

    def send_notification(self,message:str) -> None:
        data = {
            "content": message,
            "username": "pysmle"
        }

        result = requests.post(self._discord_webhook_url, json=data)

        try:
            result.raise_for_status()
        except requests.exceptions.HTTPError as err:
            print(f"Error: {err}")