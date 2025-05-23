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
import hashlib
import time

# Setup logging for CLI
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_unique_id(prefix: str = "") -> str:
    """Generate a unique identifier"""
    timestamp = str(int(time.time() * 1000))
    random_part = str(hash(timestamp))[-8:]
    return f"{prefix}{timestamp}_{random_part}" if prefix else f"{timestamp}_{random_part}"

def current_timestamp() -> float:
    """Get current timestamp"""
    return time.time()

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
    
    async def run_query():
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
    
    asyncio.run(run_query())

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