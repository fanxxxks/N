import json
import os
from dotenv import load_dotenv
from solders.keypair import Keypair

load_dotenv()

class ExecutionConfig:
    RPC_URL = os.getenv("QUICKNODE_RPC_URL", "填入RPC地址")
    _PRIV_KEY_STR = os.getenv("SOLANA_PRIVATE_KEY", "")
    _PAYER_KEYPAIR = None

    DEFAULT_SLIPPAGE_BPS = 200 # bps
    
    PRIORITY_LEVEL = "High" 
    
    SOL_MINT = "So11111111111111111111111111111111111111112"
    USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

    @classmethod
    def has_private_key(cls):
        return bool(cls._PRIV_KEY_STR)

    @classmethod
    def get_payer_keypair(cls):
        if cls._PAYER_KEYPAIR is not None:
            return cls._PAYER_KEYPAIR
        if not cls._PRIV_KEY_STR:
            raise ValueError("Missing SOLANA_PRIVATE_KEY in .env")
        try:
            cls._PAYER_KEYPAIR = Keypair.from_base58_string(cls._PRIV_KEY_STR)
        except Exception:
            cls._PAYER_KEYPAIR = Keypair.from_bytes(json.loads(cls._PRIV_KEY_STR))
        return cls._PAYER_KEYPAIR

    @classmethod
    def get_wallet_address(cls):
        return str(cls.get_payer_keypair().pubkey())
