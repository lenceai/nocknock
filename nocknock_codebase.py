# File structure for nocknock/0.7/

# nocknock/0.7/main.py
"""
NockNock v0.7 - Main entry point
A decentralized MLOps blockchain with useful work consensus
"""
import asyncio
import logging
from pathlib import Path
from config import Config
from blockchain import NockNockChain
from consensus import ZKMLOpsConsensus
from mlops import MLOpsOrchestrator
from wtm import WorldTruthModel
from network import P2PNetwork

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/nocknock.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class NockNockNode:
    """Main NockNock node implementation"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = Config.load(config_path)
        self.blockchain = NockNockChain(self.config)
        self.consensus = ZKMLOpsConsensus(self.config)
        self.mlops = MLOpsOrchestrator(self.config)
        self.wtm = WorldTruthModel(self.config)
        self.network = P2PNetwork(self.config)
        
    async def start(self):
        """Start the NockNock node"""
        logger.info("Starting NockNock v0.7 node...")
        
        # Initialize components
        await self.blockchain.initialize()
        await self.consensus.initialize()
        await self.mlops.initialize()
        await self.wtm.initialize()
        await self.network.start()
        
        logger.info("NockNock node started successfully")
        
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
    node = NockNockNode()
    await node.start()

if __name__ == "__main__":
    asyncio.run(main())

# nocknock/0.7/config.py
"""
Configuration management for NockNock
"""
import yaml
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class NetworkConfig:
    """Network configuration"""
    node_id: str = "nocknock-node-1"
    listen_port: int = 3006
    bootstrap_peers: list = None
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
                        
            # ... similar for other configs
            
            return config
            
        except FileNotFoundError:
            # Return default config if file doesn't exist
            return cls()

# nocknock/0.7/blockchain.py
"""
NockNock blockchain implementation
Solana-inspired high-performance architecture
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
    """Transaction structure"""
    tx_id: str
    tx_type: str  # model_contribution, data_contribution, validation, etc.
    sender: str
    recipient: str
    amount: int
    data: Dict
    timestamp: float
    signature: str

@dataclass
class Block:
    """Block structure"""
    block_number: int
    previous_hash: str
    timestamp: float
    transactions: List[Transaction]
    merkle_root: str
    validator: str
    ml_proof_hash: str  # Hash of ML work proof
    nonce: int
    hash: str

class NockNockChain:
    """Main blockchain implementation"""
    
    def __init__(self, config):
        self.config = config
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.utxo_set: Dict[str, int] = {}  # UTXO model like Bitcoin
        self.mempool: List[Transaction] = []
        
    async def initialize(self):
        """Initialize blockchain"""
        logger.info("Initializing NockNock blockchain...")
        
        # Create genesis block if chain is empty
        if not self.chain:
            genesis_block = self.create_genesis_block()
            self.chain.append(genesis_block)
            
        logger.info(f"Blockchain initialized with {len(self.chain)} blocks")
        
    def create_genesis_block(self) -> Block:
        """Create the genesis block"""
        genesis_tx = Transaction(
            tx_id="genesis",
            tx_type="genesis", 
            sender="system",
            recipient="system",
            amount=0,
            data={"message": "NockNock Genesis Block - The age of useful work begins"},
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
            nonce=0,
            hash=self.calculate_block_hash(0, "0" * 64, time.time(), [genesis_tx], "genesis_validator", "genesis_ml_proof", 0)
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
                           transactions: List[Transaction], validator: str, ml_proof_hash: str, nonce: int) -> str:
        """Calculate block hash"""
        merkle_root = self.calculate_merkle_root(transactions)
        
        block_string = f"{block_number}{previous_hash}{timestamp}{merkle_root}{validator}{ml_proof_hash}{nonce}"
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
        """Validate a transaction"""
        # Basic validation
        if not transaction.tx_id or not transaction.sender:
            return False
            
        # Check UTXO availability for token transfers
        if transaction.tx_type in ["token_transfer", "model_reward"]:
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
            if tx.tx_type in ["token_transfer", "model_reward"]:
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
        
    def get_balance(self, address: str) -> int:
        """Get balance for an address"""
        return self.utxo_set.get(address, 0)

# nocknock/0.7/consensus.py
"""
zkMLOps Consensus Mechanism
Proof-of-useful-work through ML training competitions
"""
import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

@dataclass
class TrainingTask:
    """ML training task definition"""
    task_id: str
    proposer: str
    model_type: str  # classification, regression, etc.
    dataset_specs: Dict
    evaluation_criteria: Dict
    reward_pool: int
    deadline: datetime
    status: str  # proposed, active, completed

@dataclass
class TrainingSubmission:
    """Training submission from validator"""
    submission_id: str
    task_id: str
    validator: str
    model_hash: str
    performance_metrics: Dict
    resource_usage: Dict
    training_proof: str  # ZK proof of training
    timestamp: datetime

@dataclass
class CompetitionResult:
    """Competition result"""
    task_id: str
    winner: str
    winning_submission: TrainingSubmission
    all_submissions: List[TrainingSubmission]
    performance_scores: Dict[str, float]

class ZKMLOpsConsensus:
    """zkMLOps consensus mechanism implementation"""
    
    def __init__(self, config):
        self.config = config
        self.active_tasks: Dict[str, TrainingTask] = {}
        self.submissions: Dict[str, List[TrainingSubmission]] = {}
        self.validators: Dict[str, Dict] = {}  # Validator registry
        self.competition_results: List[CompetitionResult] = []
        
    async def initialize(self):
        """Initialize consensus system"""
        logger.info("Initializing zkMLOps consensus system...")
        
        # Load any existing state
        await self.load_state()
        
        logger.info("zkMLOps consensus system initialized")
        
    async def load_state(self):
        """Load consensus state from disk"""
        try:
            with open("output/consensus_state.json", 'r') as f:
                state = json.load(f)
                # Restore state from saved data
                logger.info("Loaded consensus state from disk")
        except FileNotFoundError:
            logger.info("No existing consensus state found, starting fresh")
            
    async def save_state(self):
        """Save consensus state to disk"""
        try:
            state = {
                "active_tasks": len(self.active_tasks),
                "total_submissions": sum(len(subs) for subs in self.submissions.values()),
                "validators": len(self.validators),
                "timestamp": datetime.now().isoformat()
            }
            
            with open("output/consensus_state.json", 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving consensus state: {e}")
            
    async def propose_training_task(self, task: TrainingTask) -> bool:
        """Propose a new ML training task"""
        logger.info(f"Proposing training task: {task.task_id}")
        
        # Validate task proposal
        if not self.validate_task_proposal(task):
            logger.warning(f"Invalid task proposal: {task.task_id}")
            return False
            
        # Add to active tasks
        self.active_tasks[task.task_id] = task
        self.submissions[task.task_id] = []
        
        # Save task to output
        await self.save_task_to_file(task)
        
        logger.info(f"Training task {task.task_id} is now active")
        return True
        
    def validate_task_proposal(self, task: TrainingTask) -> bool:
        """Validate a training task proposal"""
        # Check required fields
        if not task.task_id or not task.proposer:
            return False
            
        # Check reward pool
        if task.reward_pool <= 0:
            return False
            
        # Check deadline is in future
        if task.deadline <= datetime.now():
            return False
            
        return True
        
    async def submit_training_result(self, submission: TrainingSubmission) -> bool:
        """Submit ML training result"""
        logger.info(f"Received training submission: {submission.submission_id}")
        
        # Validate submission
        if not self.validate_submission(submission):
            logger.warning(f"Invalid submission: {submission.submission_id}")
            return False
            
        # Verify ZK proof
        if not await self.verify_training_proof(submission):
            logger.warning(f"Invalid training proof: {submission.submission_id}")
            return False
            
        # Add to submissions
        if submission.task_id in self.submissions:
            self.submissions[submission.task_id].append(submission)
            
            # Save submission to output
            await self.save_submission_to_file(submission)
            
            logger.info(f"Training submission {submission.submission_id} accepted")
            return True
            
        return False
        
    def validate_submission(self, submission: TrainingSubmission) -> bool:
        """Validate a training submission"""
        # Check if task exists and is active
        if submission.task_id not in self.active_tasks:
            return False
            
        task = self.active_tasks[submission.task_id]
        if task.status != "active":
            return False
            
        # Check deadline
        if datetime.now() > task.deadline:
            return False
            
        # Check validator is registered
        if submission.validator not in self.validators:
            return False
            
        return True
        
    async def verify_training_proof(self, submission: TrainingSubmission) -> bool:
        """Verify zero-knowledge proof of training"""
        # Placeholder for ZK proof verification
        # In a real implementation, this would verify cryptographic proofs
        
        logger.info(f"Verifying training proof for submission {submission.submission_id}")
        
        # Simulate proof verification
        await asyncio.sleep(0.1)
        
        # For now, assume all proofs are valid
        return True
        
    async def evaluate_competition(self, task_id: str) -> Optional[CompetitionResult]:
        """Evaluate training competition and select winner"""
        if task_id not in self.active_tasks:
            return None
            
        task = self.active_tasks[task_id]
        submissions = self.submissions.get(task_id, [])
        
        if not submissions:
            logger.warning(f"No submissions for task {task_id}")
            return None
            
        logger.info(f"Evaluating competition for task {task_id} with {len(submissions)} submissions")
        
        # Score all submissions
        scores = {}
        for submission in submissions:
            score = self.calculate_submission_score(submission, task)
            scores[submission.validator] = score
            
        # Find winner
        winner = max(scores.keys(), key=lambda x: scores[x])
        winning_submission = next(s for s in submissions if s.validator == winner)
        
        # Create result
        result = CompetitionResult(
            task_id=task_id,
            winner=winner,
            winning_submission=winning_submission,
            all_submissions=submissions,
            performance_scores=scores
        )
        
        # Update task status
        task.status = "completed"
        
        # Save result
        self.competition_results.append(result)
        await self.save_competition_result(result)
        
        logger.info(f"Competition {task_id} completed. Winner: {winner} with score: {scores[winner]}")
        
        return result
        
    def calculate_submission_score(self, submission: TrainingSubmission, task: TrainingTask) -> float:
        """Calculate performance score for submission"""
        # Get performance metrics
        metrics = submission.performance_metrics
        
        # Calculate base performance score
        if task.model_type == "classification":
            base_score = metrics.get("accuracy", 0.0)
        elif task.model_type == "regression":
            base_score = 1.0 - metrics.get("mse", 1.0)  # Lower MSE is better
        else:
            base_score = metrics.get("score", 0.0)
            
        # Apply efficiency multiplier
        resource_usage = submission.resource_usage
        gpu_hours = resource_usage.get("gpu_hours", 1.0)
        efficiency_multiplier = 1.0 / max(gpu_hours, 0.1)  # Reward efficiency
        
        # Apply innovation bonus (placeholder)
        innovation_bonus = 1.0
        
        final_score = base_score * efficiency_multiplier * innovation_bonus
        
        return final_score
        
    async def process_training_competitions(self):
        """Process active training competitions"""
        current_time = datetime.now()
        
        # Check for completed competitions
        completed_tasks = []
        for task_id, task in self.active_tasks.items():
            if task.status == "active" and current_time > task.deadline:
                completed_tasks.append(task_id)
                
        # Evaluate completed competitions
        for task_id in completed_tasks:
            await self.evaluate_competition(task_id)
            
        # Save state
        await self.save_state()
        
    async def save_task_to_file(self, task: TrainingTask):
        """Save training task to file"""
        try:
            task_dict = {
                "task_id": task.task_id,
                "proposer": task.proposer,
                "model_type": task.model_type,
                "dataset_specs": task.dataset_specs,
                "evaluation_criteria": task.evaluation_criteria,
                "reward_pool": task.reward_pool,
                "deadline": task.deadline.isoformat(),
                "status": task.status
            }
            
            with open(f"output/task_{task.task_id}.json", 'w') as f:
                json.dump(task_dict, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving task to file: {e}")
            
    async def save_submission_to_file(self, submission: TrainingSubmission):
        """Save training submission to file"""
        try:
            submission_dict = {
                "submission_id": submission.submission_id,
                "task_id": submission.task_id,
                "validator": submission.validator,
                "model_hash": submission.model_hash,
                "performance_metrics": submission.performance_metrics,
                "resource_usage": submission.resource_usage,
                "training_proof": submission.training_proof,
                "timestamp": submission.timestamp.isoformat()
            }
            
            with open(f"output/submission_{submission.submission_id}.json", 'w') as f:
                json.dump(submission_dict, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving submission to file: {e}")
            
    async def save_competition_result(self, result: CompetitionResult):
        """Save competition result to file"""
        try:
            result_dict = {
                "task_id": result.task_id,
                "winner": result.winner,
                "performance_scores": result.performance_scores,
                "total_submissions": len(result.all_submissions),
                "timestamp": datetime.now().isoformat()
            }
            
            with open(f"output/competition_result_{result.task_id}.json", 'w') as f:
                json.dump(result_dict, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving competition result to file: {e}")

# nocknock/0.7/mlops.py
"""
MLOps Orchestration System
Integration with existing MLOps tools and frameworks
"""
import asyncio
import logging
import hashlib
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class MLExperiment:
    """ML experiment definition"""
    experiment_id: str
    name: str
    owner: str
    model_type: str
    parameters: Dict[str, Any]
    dataset_hash: str
    status: str  # running, completed, failed
    created_at: datetime
    completed_at: Optional[datetime] = None

@dataclass
class ModelArtifact:
    """Model artifact metadata"""
    artifact_id: str
    experiment_id: str
    model_hash: str
    model_path: str
    performance_metrics: Dict[str, float]
    model_size_bytes: int
    created_at: datetime

@dataclass
class DatasetInfo:
    """Dataset information"""
    dataset_id: str
    name: str
    contributor: str
    dataset_hash: str
    file_path: str
    size_bytes: int
    quality_score: float
    metadata: Dict[str, Any]

class MLOpsOrchestrator:
    """MLOps orchestration system"""
    
    def __init__(self, config):
        self.config = config
        self.experiments: Dict[str, MLExperiment] = {}
        self.models: Dict[str, ModelArtifact] = {}
        self.datasets: Dict[str, DatasetInfo] = {}
        self.active_training_jobs: Dict[str, Dict] = {}
        
        # Ensure storage directories exist
        Path(config.mlops.model_storage_path).mkdir(parents=True, exist_ok=True)
        Path(config.mlops.dataset_storage_path).mkdir(parents=True, exist_ok=True)
        
    async def initialize(self):
        """Initialize MLOps system"""
        logger.info("Initializing MLOps orchestrator...")
        
        # Load existing experiments and models
        await self.load_mlops_state()
        
        logger.info(f"MLOps orchestrator initialized with {len(self.experiments)} experiments")
        
    async def load_mlops_state(self):
        """Load MLOps state from disk"""
        try:
            # Load experiments
            experiments_file = Path("output/experiments.json")
            if experiments_file.exists():
                with open(experiments_file, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded {len(data)} experiments from disk")
                    
            # Load models
            models_file = Path("output/models.json") 
            if models_file.exists():
                with open(models_file, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded {len(data)} models from disk")
                    
        except Exception as e:
            logger.error(f"Error loading MLOps state: {e}")
            
    async def create_experiment(self, name: str, owner: str, model_type: str, 
                              parameters: Dict[str, Any], dataset_hash: str) -> str:
        """Create a new ML experiment"""
        experiment_id = self.generate_experiment_id(name, owner)
        
        experiment = MLExperiment(
            experiment_id=experiment_id,
            name=name,
            owner=owner,
            model_type=model_type,
            parameters=parameters,
            dataset_hash=dataset_hash,
            status="created",
            created_at=datetime.now()
        )
        
        self.experiments[experiment_id] = experiment
        
        # Save to disk
        await self.save_experiment(experiment)
        
        logger.info(f"Created experiment {experiment_id}: {name}")
        return experiment_id
        
    def generate_experiment_id(self, name: str, owner: str) -> str:
        """Generate unique experiment ID"""
        timestamp = str(datetime.now().timestamp())
        combined = f"{name}_{owner}_{timestamp}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
        
    async def start_training(self, experiment_id: str, training_config: Dict[str, Any]) -> bool:
        """Start ML model training"""
        if experiment_id not in self.experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return False
            
        experiment = self.experiments[experiment_id]
        experiment.status = "running"
        
        # Create training job
        training_job = {
            "experiment_id": experiment_id,
            "config": training_config,
            "start_time": datetime.now(),
            "status": "running"
        }
        
        self.active_training_jobs[experiment_id] = training_job
        
        # Start training task
        asyncio.create_task(self.run_training_job(experiment_id, training_config))
        
        logger.info(f"Started training for experiment {experiment_id}")
        return True
        
    async def run_training_job(self, experiment_id: str, training_config: Dict[str, Any]):
        """Run ML training job"""
        try:
            logger.info(f"Running training job for experiment {experiment_id}")
            
            # Simulate training process
            await asyncio.sleep(5.0)  # Simulate training time
            
            # Generate mock results
            performance_metrics = {
                "accuracy": 0.85 + (hash(experiment_id) % 100) / 1000,  # Mock accuracy
                "loss": 0.15 - (hash(experiment_id) % 100) / 2000,     # Mock loss
                "training_time": 300.0,  # Mock training time
                "epochs": training_config.get("epochs", 10)
            }
            
            # Create model artifact
            model_artifact = await self.create_model_artifact(experiment_id, performance_metrics)
            
            # Update experiment
            experiment = self.experiments[experiment_id]
            experiment.status = "completed"
            experiment.completed_at = datetime.now()
            
            # Clean up training job
            if experiment_id in self.active_training_jobs:
                del self.active_training_jobs[experiment_id]
                
            logger.info(f"Training completed for experiment {experiment_id}")
            
        except Exception as e:
            logger.error(f"Training failed for experiment {experiment_id}: {e}")
            
            # Update experiment status
            if experiment_id in self.experiments:
                self.experiments[experiment_id].status = "failed"
                
            # Clean up training job
            if experiment_id in self.active_training_jobs:
                del self.active_training_jobs[experiment_id]
                
    async def create_model_artifact(self, experiment_id: str, performance_metrics: Dict[str, float]) -> ModelArtifact:
        """Create model artifact from training results"""
        model_hash = hashlib.sha256(f"{experiment_id}_{datetime.now()}".encode()).hexdigest()
        model_filename = f"model_{model_hash}.pkl"
        model_path = f"{self.config.mlops.model_storage_path}/{model_filename}"
        
        # Create mock model file
        with open(model_path, 'w') as f:
            f.write(f"Mock model for experiment {experiment_id}")
            
        artifact = ModelArtifact(
            artifact_id=model_hash,
            experiment_id=experiment_id,
            model_hash=model_hash,
            model_path=model_path,
            performance_metrics=performance_metrics,
            model_size_bytes=len(f"Mock model for experiment {experiment_id}"),
            created_at=datetime.now()
        )
        
        self.models[model_hash] = artifact
        
        # Save to disk
        await self.save_model_artifact(artifact)
        
        logger.info(f"Created model artifact {model_hash} for experiment {experiment_id}")
        return artifact
        
    async def register_dataset(self, name: str, contributor: str, file_path: str, 
                             metadata: Dict[str, Any]) -> str:
        """Register a new dataset"""
        # Calculate dataset hash
        dataset_hash = await self.calculate_file_hash(file_path)
        
        # Get file size
        file_size = Path(file_path).stat().st_size
        
        # Calculate quality score (placeholder)
        quality_score = 0.8  # Mock quality score
        
        dataset = DatasetInfo(
            dataset_id=dataset_hash,
            name=name,
            contributor=contributor,
            dataset_hash=dataset_hash,
            file_path=file_path,
            size_bytes=file_size,
            quality_score=quality_score,
            metadata=metadata
        )
        
        self.datasets[dataset_hash] = dataset
        
        # Save to disk
        await self.save_dataset_info(dataset)
        
        logger.info(f"Registered dataset {name} with hash {dataset_hash}")
        return dataset_hash
        
    async def calculate_file_hash(self, file_path: str) -> str:
        """Calculate hash of file"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash: {e}")
            return hashlib.sha256(file_path.encode()).hexdigest()  # Fallback
            
    async def process_training_competitions(self):
        """Process training competitions from consensus layer"""
        # Check for completed training jobs
        completed_jobs = []
        for job_id, job in self.active_training_jobs.items():
            if job["status"] == "completed":
                completed_jobs.append(job_id)
                
        # Process completed jobs
        for job_id in completed_jobs:
            await self.process_completed_job(job_id)
            
    async def process_completed_job(self, job_id: str):
        """Process a completed training job"""
        if job_id in self.active_training_jobs:
            job = self.active_training_jobs[job_id]
            logger.info(f"Processing completed job {job_id}")
            
            # Job processing would happen here
            # For now, just clean up
            del self.active_training_jobs[job_id]
            
    async def save_experiment(self, experiment: MLExperiment):
        """Save experiment to disk"""
        try:
            experiment_dict = {
                "experiment_id": experiment.experiment_id,
                "name": experiment.name,
                "owner": experiment.owner,
                "model_type": experiment.model_type,
                "parameters": experiment.parameters,
                "dataset_hash": experiment.dataset_hash,
                "status": experiment.status,
                "created_at": experiment.created_at.isoformat(),
                "completed_at": experiment.completed_at.isoformat() if experiment.completed_at else None
            }
            
            with open(f"output/experiment_{experiment.experiment_id}.json", 'w') as f:
                json.dump(experiment_dict, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving experiment: {e}")
            
    async def save_model_artifact(self, artifact: ModelArtifact):
        """Save model artifact to disk"""
        try:
            artifact_dict = {
                "artifact_id": artifact.artifact_id,
                "experiment_id": artifact.experiment_id,
                "model_hash": artifact.model_hash,
                "model_path": artifact.model_path,
                "performance_metrics": artifact.performance_metrics,
                "model_size_bytes": artifact.model_size_bytes,
                "created_at": artifact.created_at.isoformat()
            }
            
            with open(f"output/model_{artifact.artifact_id}.json", 'w') as f:
                json.dump(artifact_dict, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving model artifact: {e}")
            
    async def save_dataset_info(self, dataset: DatasetInfo):
        """Save dataset info to disk"""
        try:
            dataset_dict = {
                "dataset_id": dataset.dataset_id,
                "name": dataset.name,
                "contributor": dataset.contributor,
                "dataset_hash": dataset.dataset_hash,
                "file_path": dataset.file_path,
                "size_bytes": dataset.size_bytes,
                "quality_score": dataset.quality_score,
                "metadata": dataset.metadata
            }
            
            with open(f"output/dataset_{dataset.dataset_id}.json", 'w') as f:
                json.dump(dataset_dict, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving dataset info: {e}")

# nocknock/0.7/wtm.py
"""
World Truth Model (WTM) Implementation
Global knowledge base with verifiable facts
"""
import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeEntity:
    """Knowledge entity in the World Truth Model"""
    entity_id: str
    name: str
    entity_type: str  # person, place, concept, event, etc.
    confidence_score: float
    attributes: Dict[str, Any]
    sources: List[str]
    last_updated: datetime

@dataclass
class KnowledgeFact:
    """Verifiable fact in the World Truth Model"""
    fact_id: str
    statement: str
    subject_entity: str
    predicate: str
    object_entity: str
    confidence_score: float
    evidence: List[str]
    domain: str  # science, history, current_events, etc.
    created_at: datetime
    verified_by: List[str]

@dataclass
class KnowledgeDomain:
    """Knowledge domain in the WTM"""
    domain_id: str
    name: str
    description: str
    entities: List[str]
    facts: List[str]
    expert_validators: List[str]
    last_updated: datetime

class WorldTruthModel:
    """World Truth Model implementation"""
    
    def __init__(self, config):
        self.config = config
        self.entities: Dict[str, KnowledgeEntity] = {}
        self.facts: Dict[str, KnowledgeFact] = {}
        self.domains: Dict[str, KnowledgeDomain] = {}
        self.update_queue: List[Dict] = []
        
        # Initialize core domains
        self.initialize_core_domains()
        
    def initialize_core_domains(self):
        """Initialize core knowledge domains"""
        core_domains = [
            ("science", "Scientific Knowledge", "Verified scientific facts, theories, and discoveries"),
            ("history", "Historical Events", "Documented historical events and figures"),
            ("geography", "Geographic Information", "Geographic locations, boundaries, and features"),
            ("mathematics", "Mathematical Knowledge", "Mathematical theorems, formulas, and constants"),
            ("current_events", "Current Events", "Recent news and developments")
        ]
        
        for domain_id, name, description in core_domains:
            domain = KnowledgeDomain(
                domain_id=domain_id,
                name=name,
                description=description,
                entities=[],
                facts=[],
                expert_validators=[],
                last_updated=datetime.now()
            )
            self.domains[domain_id] = domain
            
    async def initialize(self):
        """Initialize World Truth Model"""
        logger.info("Initializing World Truth Model...")
        
        # Load existing knowledge base
        await self.load_knowledge_base()
        
        # Initialize with some basic facts
        await self.bootstrap_basic_knowledge()
        
        logger.info(f"WTM initialized with {len(self.entities)} entities and {len(self.facts)} facts")
        
    async def load_knowledge_base(self):
        """Load existing knowledge base from disk"""
        try:
            # Load entities
            entities_file = Path("output/wtm_entities.json")
            if entities_file.exists():
                with open(entities_file, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded {len(data)} entities from knowledge base")
                    
            # Load facts
            facts_file = Path("output/wtm_facts.json")
            if facts_file.exists():
                with open(facts_file, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded {len(data)} facts from knowledge base")
                    
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            
    async def bootstrap_basic_knowledge(self):
        """Bootstrap with basic knowledge"""
        # Add some basic scientific facts
        await self.add_entity(KnowledgeEntity(
            entity_id="speed_of_light",
            name="Speed of Light",
            entity_type="physical_constant",
            confidence_score=1.0,
            attributes={
                "value": 299792458,
                "units": "meters per second",
                "symbol": "c"
            },
            sources=["CODATA 2018"],
            last_updated=datetime.now()
        ))
        
        await self.add_fact(KnowledgeFact(
            fact_id="speed_of_light_value",
            statement="The speed of light in vacuum is 299,792,458 meters per second",
            subject_entity="speed_of_light",
            predicate="has_value",
            object_entity="299792458_mps",
            confidence_score=1.0,
            evidence=["CODATA 2018 fundamental physical constants"],
            domain="science",
            created_at=datetime.now(),
            verified_by=["physics_experts"]
        ))
        
        # Add basic mathematical facts
        await self.add_entity(KnowledgeEntity(
            entity_id="pi",
            name="Pi",
            entity_type="mathematical_constant",
            confidence_score=1.0,
            attributes={
                "value": 3.141592653589793,
                "symbol": "π",
                "definition": "ratio of circle circumference to diameter"
            },
            sources=["Mathematical definition"],
            last_updated=datetime.now()
        ))
        
        logger.info("Bootstrapped basic knowledge into WTM")
        
    async def add_entity(self, entity: KnowledgeEntity):
        """Add entity to knowledge base"""
        self.entities[entity.entity_id] = entity
        
        # Add to appropriate domain
        domain_id = self.determine_entity_domain(entity)
        if domain_id in self.domains:
            if entity.entity_id not in self.domains[domain_id].entities:
                self.domains[domain_id].entities.append(entity.entity_id)
                
        logger.info(f"Added entity: {entity.name}")
        
    async def add_fact(self, fact: KnowledgeFact):
        """Add fact to knowledge base"""
        self.facts[fact.fact_id] = fact
        
        # Add to appropriate domain
        if fact.domain in self.domains:
            if fact.fact_id not in self.domains[fact.domain].facts:
                self.domains[fact.domain].facts.append(fact.fact_id)
                
        logger.info(f"Added fact: {fact.statement}")
        
    def determine_entity_domain(self, entity: KnowledgeEntity) -> str:
        """Determine which domain an entity belongs to"""
        entity_type_mapping = {
            "physical_constant": "science",
            "mathematical_constant": "mathematics",
            "historical_figure": "history",
            "geographic_location": "geography",
            "current_event": "current_events"
        }
        
        return entity_type_mapping.get(entity.entity_type, "general")
        
    async def query_knowledge(self, query: str, domain: Optional[str] = None) -> List[Dict]:
        """Query the knowledge base"""
        results = []
        
        # Simple keyword-based search
        query_lower = query.lower()
        
        # Search entities
        for entity in self.entities.values():
            if (query_lower in entity.name.lower() or 
                any(query_lower in str(attr).lower() for attr in entity.attributes.values())):
                if not domain or self.determine_entity_domain(entity) == domain:
                    results.append({
                        "type": "entity",
                        "entity": entity,
                        "confidence": entity.confidence_score
                    })
                    
        # Search facts
        for fact in self.facts.values():
            if query_lower in fact.statement.lower():
                if not domain or fact.domain == domain:
                    results.append({
                        "type": "fact", 
                        "fact": fact,
                        "confidence": fact.confidence_score
                    })
                    
        # Sort by confidence
        results.sort(key=lambda x: x["confidence"], reverse=True)
        
        logger.info(f"Query '{query}' returned {len(results)} results")
        return results[:10]  # Return top 10 results
        
    async def verify_fact(self, statement: str, evidence: List[str]) -> float:
        """Verify a fact statement with evidence"""
        # Placeholder fact verification logic
        # In a real implementation, this would use NLP and fact-checking algorithms
        
        confidence_score = 0.8  # Mock confidence
        
        # Check against existing facts
        for existing_fact in self.facts.values():
            if statement.lower() in existing_fact.statement.lower():
                confidence_score = max(confidence_score, existing_fact.confidence_score)
                
        logger.info(f"Verified fact with confidence: {confidence_score}")
        return confidence_score
        
    async def update_entity_confidence(self, entity_id: str, new_evidence: List[str]):
        """Update entity confidence based on new evidence"""
        if entity_id in self.entities:
            entity = self.entities[entity_id]
            
            # Add new evidence to sources
            entity.sources.extend(new_evidence)
            entity.last_updated = datetime.now()
            
            # Recalculate confidence (placeholder logic)
            evidence_count = len(entity.sources)
            entity.confidence_score = min(1.0, 0.5 + (evidence_count * 0.1))
            
            logger.info(f"Updated confidence for {entity.name}: {entity.confidence_score}")
            
    async def process_updates(self):
        """Process pending updates to the knowledge base"""
        # Process items in update queue
        processed_updates = []
        
        for update in self.update_queue[:10]:  # Process up to 10 updates at a time
            try:
                await self.process_single_update(update)
                processed_updates.append(update)
            except Exception as e:
                logger.error(f"Error processing update: {e}")
                
        # Remove processed updates
        for update in processed_updates:
            self.update_queue.remove(update)
            
        # Save state periodically
        if len(processed_updates) > 0:
            await self.save_knowledge_base()
            
    async def process_single_update(self, update: Dict):
        """Process a single knowledge update"""
        update_type = update.get("type")
        
        if update_type == "new_entity":
            entity_data = update["data"]
            entity = KnowledgeEntity(**entity_data)
            await self.add_entity(entity)
            
        elif update_type == "new_fact":
            fact_data = update["data"]
            fact = KnowledgeFact(**fact_data)
            await self.add_fact(fact)
            
        elif update_type == "update_confidence":
            entity_id = update["entity_id"]
            evidence = update["evidence"]
            await self.update_entity_confidence(entity_id, evidence)
            
    async def save_knowledge_base(self):
        """Save knowledge base to disk"""
        try:
            # Save entities
            entities_data = {}
            for entity_id, entity in self.entities.items():
                entities_data[entity_id] = {
                    "entity_id": entity.entity_id,
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "confidence_score": entity.confidence_score,
                    "attributes": entity.attributes,
                    "sources": entity.sources,
                    "last_updated": entity.last_updated.isoformat()
                }
                
            with open("output/wtm_entities.json", 'w') as f:
                json.dump(entities_data, f, indent=2)
                
            # Save facts
            facts_data = {}
            for fact_id, fact in self.facts.items():
                facts_data[fact_id] = {
                    "fact_id": fact.fact_id,
                    "statement": fact.statement,
                    "subject_entity": fact.subject_entity,
                    "predicate": fact.predicate,
                    "object_entity": fact.object_entity,
                    "confidence_score": fact.confidence_score,
                    "evidence": fact.evidence,
                    "domain": fact.domain,
                    "created_at": fact.created_at.isoformat(),
                    "verified_by": fact.verified_by
                }
                
            with open("output/wtm_facts.json", 'w') as f:
                json.dump(facts_data, f, indent=2)
                
            # Save domain information
            domains_data = {}
            for domain_id, domain in self.domains.items():
                domains_data[domain_id] = {
                    "domain_id": domain.domain_id,
                    "name": domain.name,
                    "description": domain.description,
                    "entities": domain.entities,
                    "facts": domain.facts,
                    "expert_validators": domain.expert_validators,
                    "last_updated": domain.last_updated.isoformat()
                }
                
            with open("output/wtm_domains.json", 'w') as f:
                json.dump(domains_data, f, indent=2)
                
            logger.info("Saved knowledge base to disk")
            
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")

# nocknock/0.7/network.py
"""
P2P Network Layer
Handles peer-to-peer communication and networking
"""
import asyncio
import logging
import json
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class PeerInfo:
    """Information about a network peer"""
    peer_id: str
    address: str
    port: int
    last_seen: datetime
    reputation_score: float
    capabilities: List[str]  # validator, trainer, storage, etc.

@dataclass
class NetworkMessage:
    """Network message structure"""
    message_id: str
    message_type: str
    sender: str
    recipient: str  # Can be "broadcast" for broadcast messages
    payload: Dict
    timestamp: datetime

class P2PNetwork:
    """Peer-to-peer network implementation"""
    
    def __init__(self, config):
        self.config = config
        self.node_id = config.network.node_id
        self.listen_port = config.network.listen_port
        self.peers: Dict[str, PeerInfo] = {}
        self.connected_peers: Set[str] = set()
        self.message_handlers: Dict[str, callable] = {}
        self.server = None
        
        # Register default message handlers
        self.register_message_handlers()
        
    async def start(self):
        """Start the P2P network"""
        logger.info(f"Starting P2P network on port {self.listen_port}")
        
        # Start server to listen for incoming connections
        self.server = await asyncio.start_server(
            self.handle_client,
            '0.0.0.0',
            self.listen_port
        )
        
        # Connect to bootstrap peers
        await self.connect_to_bootstrap_peers()
        
        logger.info(f"P2P network started with node ID: {self.node_id}")
        
    async def connect_to_bootstrap_peers(self):
        """Connect to bootstrap peers"""
        if not self.config.network.bootstrap_peers:
            logger.info("No bootstrap peers configured")
            return
            
        for peer_address in self.config.network.bootstrap_peers:
            try:
                host, port = peer_address.split(':')
                await self.connect_to_peer(host, int(port))
            except Exception as e:
                logger.warning(f"Failed to connect to bootstrap peer {peer_address}: {e}")
                
    async def connect_to_peer(self, host: str, port: int) -> bool:
        """Connect to a peer"""
        try:
            logger.info(f"Connecting to peer {host}:{port}")
            
            reader, writer = await asyncio.open_connection(host, port)
            
            # Send handshake
            handshake = NetworkMessage(
                message_id=f"handshake_{self.node_id}_{datetime.now().timestamp()}",
                message_type="handshake",
                sender=self.node_id,
                recipient="peer",
                payload={
                    "node_id": self.node_id,
                    "capabilities": ["validator", "trainer"],
                    "protocol_version": "0.7"
                },
                timestamp=datetime.now()
            )
            
            await self.send_message_to_writer(writer, handshake)
            
            # Handle peer connection
            peer_id = f"{host}:{port}"
            self.connected_peers.add(peer_id)
            
            # Start message handling for this peer
            asyncio.create_task(self.handle_peer_messages(reader, writer, peer_id))
            
            logger.info(f"Connected to peer {peer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to peer {host}:{port}: {e}")
            return False
            
    async def handle_client(self, reader, writer):
        """Handle incoming client connection"""
        peer_address = writer.get_extra_info('peername')
        logger.info(f"New peer connection from {peer_address}")
        
        peer_id = f"{peer_address[0]}:{peer_address[1]}"
        self.connected_peers.add(peer_id)
        
        try:
            await self.handle_peer_messages(reader, writer, peer_id)
        except Exception as e:
            logger.error(f"Error handling peer {peer_id}: {e}")
        finally:
            self.connected_peers.discard(peer_id)
            writer.close()
            await writer.wait_closed()
            
    async def handle_peer_messages(self, reader, writer, peer_id: str):
        """Handle messages from a specific peer"""
        try:
            while True:
                # Read message length
                length_data = await reader.read(4)
                if not length_data:
                    break
                    
                message_length = int.from_bytes(length_data, byteorder='big')
                
                # Read message data
                message_data = await reader.read(message_length)
                if not message_data:
                    break
                    
                # Parse message
                try:
                    message_dict = json.loads(message_data.decode('utf-8'))
                    message = NetworkMessage(
                        message_id=message_dict["message_id"],
                        message_type=message_dict["message_type"],
                        sender=message_dict["sender"],
                        recipient=message_dict["recipient"],
                        payload=message_dict["payload"],
                        timestamp=datetime.fromisoformat(message_dict["timestamp"])
                    )
                    
                    # Handle message
                    await self.handle_message(message, peer_id)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message from {peer_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in peer message handling for {peer_id}: {e}")
            
    async def send_message_to_writer(self, writer, message: NetworkMessage):
        """Send message to a specific writer"""
        try:
            # Serialize message
            message_dict = {
                "message_id": message.message_id,
                "message_type": message.message_type,
                "sender": message.sender,
                "recipient": message.recipient,
                "payload": message.payload,
                "timestamp": message.timestamp.isoformat()
            }
            
            message_data = json.dumps(message_dict).encode('utf-8')
            message_length = len(message_data)
            
            # Send length prefix and message
            writer.write(message_length.to_bytes(4, byteorder='big'))
            writer.write(message_data)
            await writer.drain()
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            
    async def broadcast_message(self, message: NetworkMessage):
        """Broadcast message to all connected peers"""
        logger.info(f"Broadcasting message {message.message_type} to {len(self.connected_peers)} peers")
        
        # Save broadcast message to output
        await self.save_message_to_file(message, "broadcast")
        
        # In a real implementation, this would send to all connected peers
        # For now, just log the broadcast
        
    async def handle_message(self, message: NetworkMessage, sender_peer_id: str):
        """Handle received message"""
        logger.info(f"Received {message.message_type} message from {sender_peer_id}")
        
        # Save received message to output
        await self.save_message_to_file(message, "received")
        
        # Call appropriate message handler
        if message.message_type in self.message_handlers:
            handler = self.message_handlers[message.message_type]
            await handler(message, sender_peer_id)
        else:
            logger.warning(f"No handler for message type: {message.message_type}")
            
    def register_message_handlers(self):
        """Register message handlers"""
        self.message_handlers = {
            "handshake": self.handle_handshake,
            "peer_list": self.handle_peer_list,
            "block_announcement": self.handle_block_announcement,
            "transaction": self.handle_transaction,
            "training_task": self.handle_training_task,
            "training_submission": self.handle_training_submission,
            "wtm_update": self.handle_wtm_update
        }
        
    async def handle_handshake(self, message: NetworkMessage, sender_peer_id: str):
        """Handle handshake message"""
        payload = message.payload
        remote_node_id = payload.get("node_id")
        capabilities = payload.get("capabilities", [])
        
        logger.info(f"Handshake from {remote_node_id} with capabilities: {capabilities}")
        
        # Add peer to known peers
        peer_info = PeerInfo(
            peer_id=remote_node_id,
            address=sender_peer_id.split(':')[0],
            port=int(sender_peer_id.split(':')[1]),
            last_seen=datetime.now(),
            reputation_score=1.0,
            capabilities=capabilities
        )
        
        self.peers[remote_node_id] = peer_info
        
    async def handle_peer_list(self, message: NetworkMessage, sender_peer_id: str):
        """Handle peer list message"""
        peer_list = message.payload.get("peers", [])
        logger.info(f"Received peer list with {len(peer_list)} peers")
        
        # Process peer list and potentially connect to new peers
        for peer_data in peer_list:
            if peer_data["peer_id"] not in self.peers:
                # Consider connecting to this new peer
                pass
                
    async def handle_block_announcement(self, message: NetworkMessage, sender_peer_id: str):
        """Handle block announcement"""
        block_hash = message.payload.get("block_hash")
        block_number = message.payload.get("block_number")
        
        logger.info(f"Block announcement: #{block_number} ({block_hash})")
        
    async def handle_transaction(self, message: NetworkMessage, sender_peer_id: str):
        """Handle transaction message"""
        transaction_data = message.payload
        logger.info(f"Received transaction: {transaction_data.get('tx_id')}")
        
    async def handle_training_task(self, message: NetworkMessage, sender_peer_id: str):
        """Handle training task message"""
        task_data = message.payload
        logger.info(f"Received training task: {task_data.get('task_id')}")
        
    async def handle_training_submission(self, message: NetworkMessage, sender_peer_id: str):
        """Handle training submission message"""
        submission_data = message.payload
        logger.info(f"Received training submission: {submission_data.get('submission_id')}")
        
    async def handle_wtm_update(self, message: NetworkMessage, sender_peer_id: str):
        """Handle World Truth Model update"""
        update_data = message.payload
        logger.info(f"Received WTM update: {update_data.get('update_type')}")
        
    async def maintain_connections(self):
        """Maintain network connections"""
        # Check peer health
        current_time = datetime.now()
        disconnected_peers = []
        
        for peer_id, peer_info in self.peers.items():
            time_since_last_seen = current_time - peer_info.last_seen
            if time_since_last_seen.total_seconds() > 300:  # 5 minutes timeout
                disconnected_peers.append(peer_id)
                
        # Remove disconnected peers
        for peer_id in disconnected_peers:
            if peer_id in self.peers:
                del self.peers[peer_id]
                self.connected_peers.discard(peer_id)
                logger.info(f"Removed disconnected peer: {peer_id}")
                
        # Try to maintain minimum number of connections
        if len(self.connected_peers) < 3:  # Minimum 3 peers
            await self.discover_and_connect_peers()
            
    async def discover_and_connect_peers(self):
        """Discover and connect to new peers"""
        # In a real implementation, this would use peer discovery mechanisms
        logger.info("Attempting to discover new peers...")
        
    async def save_message_to_file(self, message: NetworkMessage, direction: str):
        """Save message to output file"""
        try:
            message_data = {
                "direction": direction,
                "message_id": message.message_id,
                "message_type": message.message_type,
                "sender": message.sender,
                "recipient": message.recipient,
                "payload": message.payload,
                "timestamp": message.timestamp.isoformat()
            }
            
            filename = f"output/network_message_{message.message_id}.json"
            with open(filename, 'w') as f:
                json.dump(message_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving message to file: {e}")

# nocknock/0.7/utils.py
"""
Utility functions and helpers
"""
import hashlib
import time
import uuid
from typing import Any, Dict, List
from datetime import datetime

def generate_unique_id(prefix: str = "") -> str:
    """Generate a unique identifier"""
    timestamp = str(int(time.time() * 1000))
    random_part = str(uuid.uuid4())[:8]
    return f"{prefix}{timestamp}_{random_part}" if prefix else f"{timestamp}_{random_part}"

def calculate_hash(data: Any) -> str:
    """Calculate SHA-256 hash of data"""
    if isinstance(data, str):
        return hashlib.sha256(data.encode()).hexdigest()
    elif isinstance(data, (dict, list)):
        import json
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()
    else:
        return hashlib.sha256(str(data).encode()).hexdigest()

def validate_address(address: str) -> bool:
    """Validate a NockNock address"""
    # Simple validation - in real implementation would be more sophisticated
    return len(address) == 64 and all(c in '0123456789abcdef' for c in address.lower())

def format_token_amount(amount: int) -> str:
    """Format token amount for display"""
    # NOCK tokens have 9 decimal places (like SOL)
    return f"{amount / 1_000_000_000:.9f} NOCK"

def current_timestamp() -> float:
    """Get current timestamp"""
    return time.time()

def datetime_to_timestamp(dt: datetime) -> float:
    """Convert datetime to timestamp"""
    return dt.timestamp()

def timestamp_to_datetime(timestamp: float) -> datetime:
    """Convert timestamp to datetime"""
    return datetime.fromtimestamp(timestamp)

class Logger:
    """Simple logging utility"""
    
    @staticmethod
    def log_to_file(filename: str, data: Dict):
        """Log data to file"""
        import json
        from pathlib import Path
        
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            
            log_file = Path(f"output/{filename}")
            
            # Append to existing log file
            if log_file.exists():
                with open(log_file, 'r') as f:
                    existing_logs = json.load(f)
                existing_logs.append(log_entry)
            else:
                existing_logs = [log_entry]
                
            with open(log_file, 'w') as f:
                json.dump(existing_logs, f, indent=2)
                
        except Exception as e:
            print(f"Error logging to file: {e}")

# nocknock/0.7/cli.py
"""
Command Line Interface for NockNock
"""
import asyncio
import click
import logging
from pathlib import Path
from main import NockNockNode
from blockchain import Transaction
from consensus import TrainingTask, TrainingSubmission
from mlops import MLOpsOrchestrator
from datetime import datetime, timedelta
from utils import generate_unique_id, current_timestamp

# Setup logging for CLI
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """NockNock v0.7 Command Line Interface"""
    pass

@cli.command()
@click.option('--config', default='config.yaml', help='Configuration file path')
def start(config):
    """Start a NockNock node"""
    click.echo("Starting NockNock v0.7 node...")
    
    # Ensure output directory exists
    Path("output").mkdir(exist_ok=True)
    
    async def run_node():
        node = NockNockNode(config)
        await node.start()
    
    try:
        asyncio.run(run_node())
    except KeyboardInterrupt:
        click.echo("\nShutting down NockNock node...")

@cli.command()
@click.option('--name', required=True, help='Experiment name')
@click.option('--model-type', default='classification', help='Model type (classification, regression)')
@click.option('--dataset', required=True, help='Dataset path or identifier')
def create_experiment(name, model_type, dataset):
    """Create a new ML experiment"""
    click.echo(f"Creating experiment: {name}")
    
    async def create_exp():
        from config import Config
        config = Config.load('config.yaml')
        mlops = MLOpsOrchestrator(config)
        await mlops.initialize()
        
        # Create experiment
        experiment_id = await mlops.create_experiment(
            name=name,
            owner="cli_user",
            model_type=model_type,
            parameters={"epochs": 10, "learning_rate": 0.001},
            dataset_hash=dataset
        )
        
        click.echo(f"Created experiment: {experiment_id}")
        
        # Start training
        training_config = {
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001
        }
        
        success = await mlops.start_training(experiment_id, training_config)
        if success:
            click.echo("Training started successfully")
        else:
            click.echo("Failed to start training")
    
    asyncio.run(create_exp())

@cli.command()
@click.option('--task-id', required=True, help='Training task ID')
@click.option('--model-type', default='classification', help='Model type')
@click.option('--reward', default=1000, help='Reward pool in NOCK tokens')
def propose_task(task_id, model_type, reward):
    """Propose a new training task"""
    click.echo(f"Proposing training task: {task_id}")
    
    async def propose():
        from config import Config
        from consensus import ZKMLOpsConsensus, TrainingTask
        
        config = Config.load('config.yaml')
        consensus = ZKMLOpsConsensus(config)
        await consensus.initialize()
        
        # Create training task
        task = TrainingTask(
            task_id=task_id,
            proposer="cli_user",
            model_type=model_type,
            dataset_specs={"type": "image_classification", "size": "10000_samples"},
            evaluation_criteria={"metric": "accuracy", "minimum": 0.8},
            reward_pool=reward,
            deadline=datetime.now() + timedelta(hours=24),
            status="proposed"
        )
        
        success = await consensus.propose_training_task(task)
        if success:
            click.echo(f"Task {task_id} proposed successfully")
        else:
            click.echo(f"Failed to propose task {task_id}")
    
    asyncio.run(propose())

@cli.command()
@click.option('--query', required=True, help='Knowledge query')
@click.option('--domain', help='Knowledge domain to search')
def query_wtm(query, domain):
    """Query the World Truth Model"""
    click.echo(f"Querying WTM: {query}")
    
    async def query():
        from config import Config
        from wtm import WorldTruthModel
        
        config = Config.load('config.yaml')
        wtm = WorldTruthModel(config)
        await wtm.initialize()
        
        results = await wtm.query_knowledge(query, domain)
        
        if results:
            click.echo(f"Found {len(results)} results:")
            for i, result in enumerate(results[:5], 1):
                if result["type"] == "entity":
                    entity = result["entity"]
                    click.echo(f"{i}. Entity: {entity.name} (confidence: {entity.confidence_score:.2f})")
                elif result["type"] == "fact":
                    fact = result["fact"]
                    click.echo(f"{i}. Fact: {fact.statement} (confidence: {fact.confidence_score:.2f})")
        else:
            click.echo("No results found")
    
    asyncio.run(query())

@cli.command()
def status():
    """Show node status"""
    click.echo("NockNock Node Status")
    click.echo("=" * 50)
    
    # Check if output directory exists and show file counts
    output_dir = Path("output")
    if output_dir.exists():
        files = list(output_dir.glob("*.json"))
        click.echo(f"Output files: {len(files)}")
        
        # Count different file types
        block_files = len(list(output_dir.glob("block_*.json")))
        experiment_files = len(list(output_dir.glob("experiment_*.json")))
        model_files = len(list(output_dir.glob("model_*.json")))
        
        click.echo(f"Blocks: {block_files}")
        click.echo(f"Experiments: {experiment_files}")
        click.echo(f"Models: {model_files}")
    else:
        click.echo("Output directory not found - node hasn't been started yet")

@cli.command()
def list_files():
    """List all output files"""
    output_dir = Path("output")
    if not output_dir.exists():
        click.echo("No output directory found")
        return
    
    files = sorted(output_dir.glob("*.json"))
    if files:
        click.echo("Output files:")
        for file in files:
            size = file.stat().st_size
            modified = datetime.fromtimestamp(file.stat().st_mtime)
            click.echo(f"  {file.name} ({size} bytes, modified: {modified.strftime('%Y-%m-%d %H:%M:%S')})")
    else:
        click.echo("No output files found")

@cli.command()
@click.option('--sender', default='cli_user', help='Transaction sender')
@click.option('--recipient', required=True, help='Transaction recipient')
@click.option('--amount', default=100, help='Transaction amount')
def send_transaction(sender, recipient, amount):
    """Send a transaction"""
    click.echo(f"Sending {amount} NOCK from {sender} to {recipient}")
    
    async def send_tx():
        from config import Config
        from blockchain import NockNockChain, Transaction
        
        config = Config.load('config.yaml')
        blockchain = NockNockChain(config)
        await blockchain.initialize()
        
        # Create transaction
        tx = Transaction(
            tx_id=generate_unique_id("tx_"),
            tx_type="token_transfer",
            sender=sender,
            recipient=recipient,
            amount=amount,
            data={"memo": "CLI transaction"},
            timestamp=current_timestamp(),
            signature="cli_signature"
        )
        
        await blockchain.add_transaction(tx)
        click.echo(f"Transaction {tx.tx_id} added to mempool")
    
    asyncio.run(send_tx())

if __name__ == '__main__':
    cli()

# nocknock/0.7/requirements.txt
# NockNock v0.7 Dependencies

# Core dependencies
asyncio-mqtt==0.13.0
click>=8.0.0
PyYAML>=6.0
pydantic>=1.10.0
dataclasses-json>=0.5.7

# Cryptography and blockchain
cryptography>=3.4.8
pycryptodome>=3.15.0
ecdsa>=0.18.0

# Networking
aiohttp>=3.8.0
websockets>=10.4

# ML and Data Science
numpy>=1.21.0
pandas>=1.5.0
scikit-learn>=1.1.0
torch>=1.12.0
transformers>=4.21.0

# MLOps Integration
mlflow>=2.0.0
# zenml>=0.20.0  # Optional - uncomment if needed

# Development and testing
pytest>=7.0.0
pytest-asyncio>=0.21.0
black>=22.0.0
flake8>=5.0.0

# nocknock/0.7/config.yaml
# NockNock v0.7 Configuration

# Network configuration
network:
  node_id: "nocknock-node-1"
  listen_port: 3006
  bootstrap_peers:
    - "127.0.0.1:3007"
    - "127.0.0.1:3008"
  max_peers: 50

# Blockchain configuration  
blockchain:
  genesis_hash: "0x000000000000000000000000000000000000000000000000000000000000000"
  block_time: 5  # seconds
  max_block_size: 1000000  # bytes
  difficulty_adjustment: 100  # blocks

# MLOps configuration
mlops:
  max_concurrent_training: 10
  training_timeout: 3600  # seconds
  model_storage_path: "output/models"
  dataset_storage_path: "output/datasets"

# Consensus configuration
consensus:
  validator_stake_minimum: 1000  # NOCK tokens
  competition_duration: 86400  # seconds (24 hours)
  proof_verification_timeout: 300  # seconds

# World Truth Model configuration
wtm:
  max_entities: 1000000
  max_facts: 10000000
  confidence_threshold: 0.7
  update_batch_size: 100

# Logging configuration
logging:
  level: "INFO"
  file: "output/nocknock.log"
  max_size: "100MB"
  backup_count: 5

# nocknock/0.7/README.md
# NockNock v0.7

A decentralized MLOps blockchain with useful work consensus, enabling global collaboration on machine learning while building the World Truth Model.

## Overview

NockNock transforms blockchain mining from wasteful computation into meaningful machine learning work. Instead of solving arbitrary puzzles, network participants compete to train the best ML models, with all work contributing to a global knowledge base.

## Features

- **zkMLOps Consensus**: Proof-of-useful-work through ML training competitions
- **World Truth Model**: Comprehensive, verifiable global knowledge base
- **MLOps Integration**: Native support for MLflow, ZenML, and other ML tools
- **High Performance**: Solana-inspired architecture for fast transactions
- **Privacy-Preserving**: Zero-knowledge proofs protect sensitive data and methods
- **Fair Distribution**: 100% of tokens earned through useful work (no pre-mine)

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running a Node

```bash
# Start a NockNock node
python main.py

# Or use the CLI
python cli.py start
```

### Creating ML Experiments

```bash
# Create a new experiment
python cli.py create-experiment --name "image-classifier" --model-type classification --dataset "imagenet-subset"

# Propose a training task
python cli.py propose-task --task-id "task-001" --model-type classification --reward 1000
```

### Querying the World Truth Model

```bash
# Query for knowledge
python cli.py query-wtm --query "speed of light" --domain science

# Check node status
python cli.py status
```

## Architecture

### Core Components

- **Blockchain Layer** (`blockchain.py`): High-performance transaction processing
- **Consensus Layer** (`consensus.py`): zkMLOps useful work consensus mechanism  
- **MLOps Layer** (`mlops.py`): ML experiment orchestration and model management
- **World Truth Model** (`wtm.py`): Global knowledge base with verifiable facts
- **Network Layer** (`network.py`): P2P networking and communication
- **CLI Interface** (`cli.py`): Command-line tools for interaction

### Key Innovations

1. **Useful Work Mining**: Every computation advances ML research
2. **Zero-Knowledge ML**: Train and verify models without exposing data
3. **Global Collaboration**: Researchers worldwide contribute to shared models
4. **Verifiable Knowledge**: All facts backed by cryptographic proofs

## Configuration

Edit `config.yaml` to customize:

- Network settings (ports, peers)
- Blockchain parameters (block time, size limits)
- MLOps configuration (storage paths, timeouts)
- Consensus rules (stake requirements, competition duration)

## Output Files

All blockchain data, experiments, and logs are stored in the `output/` directory:

- `block_*.json`: Blockchain blocks
- `experiment_*.json`: ML experiments
- `model_*.json`: Model artifacts
- `task_*.json`: Training tasks
- `wtm_*.json`: World Truth Model data
- `nocknock.log`: Node logs

## Development

### File Structure

```
nocknock/0.7/
├── main.py           # Main node entry point
├── blockchain.py     # Blockchain implementation
├── consensus.py      # zkMLOps consensus
├── mlops.py         # MLOps orchestration
├── wtm.py           # World Truth Model
├── network.py       # P2P networking
├── cli.py           # Command-line interface
├── config.py        # Configuration management
├── utils.py         # Utility functions
├── config.yaml      # Default configuration
├── requirements.txt # Python dependencies
└── output/          # Generated files and logs
```

### Testing

```bash
# Run basic tests
python -m pytest

# Test specific components
python -c "import asyncio; from main import NockNockNode; asyncio.run(NockNockNode().start())"
```

## Contributing

NockNock is open source under the MIT License. Contributions welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Vision

NockNock enables a future where:
- AI development is a global collaborative effort
- Computational resources serve beneficial purposes
- Knowledge is accessible to all
- Research is reproducible and verifiable
- Innovation is incentivized through fair rewards

Join us in building the infrastructure for humanity's greatest challenges.

---

**NockNock v0.7** - The age of useful work begins.
