import aiohttp
import asyncio
from datetime import datetime, timedelta
from loguru import logger
from ..config import Config
from .base import DataProvider

class BirdeyeProvider(DataProvider):
    def __init__(self):
        self.base_url = Config.BIRDEYE_BASE_URL
        self.headers = {
            "X-API-KEY": Config.BIRDEYE_API_KEY,
            "accept": "application/json"
        }
        self.semaphore = asyncio.Semaphore(Config.CONCURRENCY)

    @staticmethod
    def _as_float(value, default=0.0):
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default
        
    async def get_trending_tokens(self, limit=100):
        url = f"{self.base_url}/defi/token_trending"
        params = {
            "sort_by": "rank",
            "sort_type": "asc",
            "offset": "0",
            "limit": str(limit)
        }
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        raw_list = data.get('data', {}).get('tokens', [])
                        
                        results = []
                        for t in raw_list:
                            results.append({
                                'address': t['address'],
                                'symbol': t.get('symbol', 'UNKNOWN'),
                                'name': t.get('name', 'UNKNOWN'),
                                'decimals': t.get('decimals', 6),
                                'liquidity': self._as_float(t.get('liquidity')),
                                'fdv': self._as_float(t.get('fdv'))
                            })
                        return results
                    else:
                        logger.error(f"Birdeye Trending Error: {resp.status}")
                        return []
            except Exception as e:
                logger.error(f"Birdeye Trending Exception: {e}")
                return []

    async def get_token_history(self, session, address, days=Config.HISTORY_DAYS, liquidity=None, fdv=None):
        time_to = int(datetime.now().timestamp())
        time_from = int((datetime.now() - timedelta(days=days)).timestamp())
        snapshot_liquidity = self._as_float(liquidity)
        snapshot_fdv = self._as_float(fdv)
        
        url = f"{self.base_url}/defi/ohlcv"
        params = {
            "address": address,
            "type": Config.TIMEFRAME,
            "time_from": time_from,
            "time_to": time_to
        }

        async with self.semaphore:
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        items = data.get('data', {}).get('items', [])
                        if not items: return []
                        
                        formatted = []
                        for item in items:
                            candle_liquidity = self._as_float(item.get('liquidity'), snapshot_liquidity)
                            candle_fdv = self._as_float(item.get('fdv'), snapshot_fdv)
                            formatted.append((
                                datetime.fromtimestamp(item['unixTime']), # time
                                address,                                  # address
                                float(item['o']),                         # open
                                float(item['h']),                         # high
                                float(item['l']),                         # low
                                float(item['c']),                         # close
                                float(item['v']),                         # volume
                                candle_liquidity,                         # liquidity
                                candle_fdv,                               # fdv
                                'birdeye'                                 # source
                            ))
                        return formatted
                    elif resp.status == 429:
                        logger.warning(f"Birdeye 429 for {address}, retrying...")
                        await asyncio.sleep(2)
                        return await self.get_token_history(session, address, days, liquidity=liquidity, fdv=fdv)
                    else:
                        return []
            except Exception as e:
                logger.error(f"Birdeye Fetch Error {address}: {e}")
                return []
