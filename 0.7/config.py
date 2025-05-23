"""
Configuration management for NockNock
"""
import yaml
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class NetworkConfig:
    """Network configuration"""
    node_id: str = "nocknock-node-1"
    listen_port: int = 3006
    bootstrap_peers: Optional[List[str]] = None
    max_peers: int = 50

@dataclass
class BlockchainConfig:
    """Blockchain configuration"""
    genesis_hash: str = "0x000000000000000000000000000000000000000000000000000000000000000"
    block_time: int = 5  # seconds
    max_block_size: int = 1000000  # bytes
    difficulty_adjustment: int = 100  # blocks

@dataclass
class MLOpsConfig:
    """MLOps configuration"""
    max_concurrent_training: int = 10
    training_timeout: int = 3600  # seconds
    model_storage_path: str = "output/models"
    dataset_storage_path: str = "output/datasets"

@dataclass
class ConsensusConfig:
    """Consensus configuration"""
    validator_stake_minimum: int = 1000  # NOCK tokens
    competition_duration: int = 86400  # seconds (24 hours)
    proof_verification_timeout: int = 300  # seconds

class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.network = NetworkConfig()
        self.blockchain = BlockchainConfig()
        self.mlops = MLOpsConfig()
        self.consensus = ConsensusConfig()
        
    @classmethod
    def load(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
            
            config = cls()
            
            # Update configurations from file
            if 'network' in data:
                for key, value in data['network'].items():
                    if hasattr(config.network, key):
                        setattr(config.network, key, value)
                        
            if 'blockchain' in data:
                for key, value in data['blockchain'].items():
                    if hasattr(config.blockchain, key):
                        setattr(config.blockchain, key, value)
                        
            if 'mlops' in data:
                for key, value in data['mlops'].items():
                    if hasattr(config.mlops, key):
                        setattr(config.mlops, key, value)
                        
            if 'consensus' in data:
                for key, value in data['consensus'].items():
                    if hasattr(config.consensus, key):
                        setattr(config.consensus, key, value)
            
            return config
            
        except FileNotFoundError:
            # Return default config if file doesn't exist
            return cls() 