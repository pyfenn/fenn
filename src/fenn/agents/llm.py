class Client:
    def __init__(self):
        self._fn = None
    
    def register(self, fn):
        self._fn = fn

    def chat_complete(self, messages, **kwargs):
        if self._fn:
            return self._fn(messages, **kwargs)
        raise ValueError("No LLM function registered")