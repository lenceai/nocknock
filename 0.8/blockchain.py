"""
WTM Blockchain Implementation
Solana-inspired high-performance architecture for World Truth Model
"""
import hashlib
import json
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class Transaction:
    """Transaction structure for WTM blockchain"""
    tx_id: str
    tx_type: str  # model_contribution, data_contribution, validation, knowledge_update, etc.
    sender: str
    recipient: str
    amount: int  # WTM tokens
    data: Dict
    timestamp: float
    signature: str

@dataclass
class Block:
    """Block structure for WTM blockchain"""
    block_number: int
    previous_hash: str
    timestamp: float
    transactions: List[Transaction]
    merkle_root: str
    validator: str
    ml_proof_hash: str  # Hash of ML work proof
    knowledge_hash: str  # Hash of knowledge contributions
    nonce: int
    hash: str

class WTMChain:
    """Main WTM blockchain implementation"""
    
    def __init__(self, config):
        self.config = config
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.utxo_set: Dict[str, int] = {}  # WTM token balances
        self.mempool: List[Transaction] = []
        
    async def initialize(self):
        """Initialize WTM blockchain"""
        logger.info("Initializing WTM blockchain...")
        
        # Create genesis block if chain is empty
        if not self.chain:
            genesis_block = self.create_genesis_block()
            self.chain.append(genesis_block)
            
        logger.info(f"WTM blockchain initialized with {len(self.chain)} blocks")
        
    def create_genesis_block(self) -> Block:
        """Create the genesis block for WTM"""
        genesis_tx = Transaction(
            tx_id="genesis",
            tx_type="genesis", 
            sender="system",
            recipient="system",
            amount=0,
            data={"message": "WTM Genesis Block - Building the World Truth Model begins"},
            timestamp=time.time(),
            signature="genesis_signature"
        )
        
        return Block(
            block_number=0,
            previous_hash="0" * 64,
            timestamp=time.time(),
            transactions=[genesis_tx],
            merkle_root=self.calculate_merkle_root([genesis_tx]),
            validator="genesis_validator",
            ml_proof_hash="genesis_ml_proof",
            knowledge_hash="genesis_knowledge",
            nonce=0,
            hash=self.calculate_block_hash(0, "0" * 64, time.time(), [genesis_tx], "genesis_validator", "genesis_ml_proof", "genesis_knowledge", 0)
        )
        
    def calculate_merkle_root(self, transactions: List[Transaction]) -> str:
        """Calculate Merkle root of transactions"""
        if not transactions:
            return "0" * 64
            
        # Simple implementation - hash all transaction IDs
        tx_hashes = [tx.tx_id for tx in transactions]
        combined = "".join(sorted(tx_hashes))
        return hashlib.sha256(combined.encode()).hexdigest()
        
    def calculate_block_hash(self, block_number: int, previous_hash: str, timestamp: float, 
                           transactions: List[Transaction], validator: str, ml_proof_hash: str, 
                           knowledge_hash: str, nonce: int) -> str:
        """Calculate block hash for WTM"""
        merkle_root = self.calculate_merkle_root(transactions)
        
        block_string = f"{block_number}{previous_hash}{timestamp}{merkle_root}{validator}{ml_proof_hash}{knowledge_hash}{nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()
        
    async def add_transaction(self, transaction: Transaction):
        """Add transaction to mempool"""
        # Validate transaction
        if self.validate_transaction(transaction):
            self.mempool.append(transaction)
            logger.info(f"Added transaction {transaction.tx_id} to mempool")
        else:
            logger.warning(f"Invalid transaction {transaction.tx_id} rejected")
            
    def validate_transaction(self, transaction: Transaction) -> bool:
        """Validate a WTM transaction"""
        # Basic validation
        if not transaction.tx_id or not transaction.sender:
            return False
            
        # Check WTM token availability for token transfers
        if transaction.tx_type in ["token_transfer", "model_reward", "knowledge_reward"]:
            sender_balance = self.utxo_set.get(transaction.sender, 0)
            if sender_balance < transaction.amount:
                return False
                
        return True
        
    async def process_pending_transactions(self):
        """Process pending transactions and create new blocks"""
        if len(self.mempool) >= 10:  # Create block when we have enough transactions
            await self.create_new_block()
            
    async def create_new_block(self):
        """Create a new block from pending transactions"""
        if not self.mempool:
            return
            
        # Get transactions for new block
        transactions = self.mempool[:100]  # Max 100 transactions per block
        self.mempool = self.mempool[100:]
        
        # Get previous block
        previous_block = self.chain[-1] if self.chain else None
        previous_hash = previous_block.hash if previous_block else "0" * 64
        
        # Create new block
        new_block = Block(
            block_number=len(self.chain),
            previous_hash=previous_hash,
            timestamp=time.time(),
            transactions=transactions,
            merkle_root=self.calculate_merkle_root(transactions),
            validator="temp_validator",  # Will be determined by consensus
            ml_proof_hash="temp_ml_proof",  # Will be provided by ML work
            knowledge_hash="temp_knowledge",  # Will be provided by knowledge contributions
            nonce=0,
            hash=""  # Will be calculated
        )
        
        # Calculate hash
        new_block.hash = self.calculate_block_hash(
            new_block.block_number,
            new_block.previous_hash,
            new_block.timestamp,
            new_block.transactions,
            new_block.validator,
            new_block.ml_proof_hash,
            new_block.knowledge_hash,
            new_block.nonce
        )
        
        # Add to chain
        self.chain.append(new_block)
        
        # Update UTXO set
        self.update_utxo_set(transactions)
        
        # Save block to output
        self.save_block_to_file(new_block)
        
        logger.info(f"Created new block #{new_block.block_number} with {len(transactions)} transactions")
        
    def update_utxo_set(self, transactions: List[Transaction]):
        """Update UTXO set with new transactions"""
        for tx in transactions:
            if tx.tx_type in ["token_transfer", "model_reward", "knowledge_reward"]:
                # Deduct from sender
                self.utxo_set[tx.sender] = self.utxo_set.get(tx.sender, 0) - tx.amount
                # Add to recipient
                self.utxo_set[tx.recipient] = self.utxo_set.get(tx.recipient, 0) + tx.amount
                
    def save_block_to_file(self, block: Block):
        """Save block to output file"""
        try:
            with open(f"output/block_{block.block_number}.json", 'w') as f:
                # Convert block to dict for JSON serialization
                block_dict = asdict(block)
                json.dump(block_dict, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving block to file: {e}")
            
    def get_latest_block(self) -> Optional[Block]:
        """Get the latest block"""
        return self.chain[-1] if self.chain else None
        
    def get_wtm_balance(self, address: str) -> int:
        """Get WTM token balance for an address"""
        return self.utxo_set.get(address, 0)
        
    def get_total_wtm_supply(self) -> int:
        """Get total WTM token supply"""
        return sum(self.utxo_set.values()) 