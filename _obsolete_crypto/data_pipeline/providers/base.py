from abc import ABC, abstractmethod

class DataProvider(ABC):    
    @abstractmethod
    async def get_trending_tokens(self, limit: int):
        pass

    @abstractmethod
    async def get_token_history(self, session, address: str, days: int, **kwargs):
        pass
