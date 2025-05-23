"""
WTM v0.8 - World Truth Model Main Entry Point
A decentralized MLOps blockchain with useful work consensus
Building the global knowledge base through collaborative ML
"""
import asyncio
import logging
from pathlib import Path
from config import Config
from blockchain import WTMChain
from consensus import ZKMLOpsConsensus
from mlops import MLOpsOrchestrator
from wtm import WorldTruthModel
from network import P2PNetwork

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/wtm.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class WTMNode:
    """Main WTM node implementation"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = Config.load(config_path)
        self.blockchain = WTMChain(self.config)
        self.consensus = ZKMLOpsConsensus(self.config)
        self.mlops = MLOpsOrchestrator(self.config)
        self.wtm = WorldTruthModel(self.config)
        self.network = P2PNetwork(self.config)
        
    async def start(self):
        """Start the WTM node"""
        logger.info("Starting WTM v0.8 node...")
        
        # Initialize components
        await self.blockchain.initialize()
        await self.consensus.initialize()
        await self.mlops.initialize()
        await self.wtm.initialize()
        await self.network.start()
        
        logger.info("WTM node started successfully")
        
        # Main event loop
        await self.run_main_loop()
        
    async def run_main_loop(self):
        """Main node event loop"""
        while True:
            try:
                # Process pending transactions
                await self.blockchain.process_pending_transactions()
                
                # Handle ML training competitions
                await self.mlops.process_training_competitions()
                
                # Update World Truth Model
                await self.wtm.process_updates()
                
                # Network maintenance
                await self.network.maintain_connections()
                
                # Brief pause
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5.0)

async def main():
    """Main entry point"""
    # Ensure output directory exists
    Path("output").mkdir(exist_ok=True)
    
    # Create and start node
    node = WTMNode()
    await node.start()

if __name__ == "__main__":
    asyncio.run(main()) 