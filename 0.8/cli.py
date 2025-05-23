"""
Command Line Interface for WTM (World Truth Model)
Enhanced with knowledge-specific commands and operations
"""
import asyncio
import click
import logging
from pathlib import Path
from main import WTMNode
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
    """Generate a unique identifier for WTM"""
    timestamp = str(int(time.time() * 1000))
    random_part = str(hash(timestamp))[-8:]
    return f"{prefix}{timestamp}_{random_part}" if prefix else f"{timestamp}_{random_part}"

def current_timestamp() -> float:
    """Get current timestamp"""
    return time.time()

@click.group()
def cli():
    """WTM v0.8 Command Line Interface - World Truth Model"""
    pass

@cli.command()
@click.option('--config', default='config.yaml', help='Configuration file path')
def start(config):
    """Start a WTM node"""
    click.echo("Starting WTM v0.8 node...")
    click.echo("Building the World Truth Model through collaborative intelligence")
    
    # Ensure output directory exists
    Path("output").mkdir(exist_ok=True)
    
    async def run_node():
        node = WTMNode(config)
        await node.start()
    
    try:
        asyncio.run(run_node())
    except KeyboardInterrupt:
        click.echo("\nShutting down WTM node...")

@cli.command()
@click.option('--name', required=True, help='Experiment name')
@click.option('--model-type', default='knowledge_extraction', help='Model type (classification, regression, knowledge_extraction)')
@click.option('--dataset', required=True, help='Dataset path or identifier')
@click.option('--knowledge-target', default='general', help='Knowledge domain this experiment targets')
def create_experiment(name, model_type, dataset, knowledge_target):
    """Create a new ML experiment for WTM"""
    click.echo(f"Creating WTM experiment: {name}")
    click.echo(f"Target knowledge domain: {knowledge_target}")
    
    async def create_exp():
        from config import Config
        config = Config.load('config.yaml')
        mlops = MLOpsOrchestrator(config)
        await mlops.initialize()
        
        # Create experiment with knowledge target
        experiment_id = await mlops.create_experiment(
            name=name,
            owner="cli_user",
            model_type=model_type,
            parameters={"epochs": 10, "learning_rate": 0.001},
            dataset_hash=dataset,
            knowledge_target=knowledge_target
        )
        
        click.echo(f"Created WTM experiment: {experiment_id}")
        
        # Start training
        training_config = {
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001,
            "knowledge_extraction": True
        }
        
        success = await mlops.start_training(experiment_id, training_config)
        if success:
            click.echo("WTM training started successfully")
        else:
            click.echo("Failed to start WTM training")
    
    asyncio.run(create_exp())

@cli.command()
@click.option('--task-id', required=True, help='Training task ID')
@click.option('--model-type', default='knowledge_extraction', help='Model type')
@click.option('--reward', default=1000, help='Reward pool in WTM tokens')
@click.option('--knowledge-target', required=True, help='Knowledge domain this task targets')
def propose_task(task_id, model_type, reward, knowledge_target):
    """Propose a new training task for WTM"""
    click.echo(f"Proposing WTM training task: {task_id}")
    click.echo(f"Knowledge target: {knowledge_target}")
    
    async def propose():
        from config import Config
        from consensus import ZKMLOpsConsensus, TrainingTask
        
        config = Config.load('config.yaml')
        consensus = ZKMLOpsConsensus(config)
        await consensus.initialize()
        
        # Create training task with knowledge target
        task = TrainingTask(
            task_id=task_id,
            proposer="cli_user",
            model_type=model_type,
            dataset_specs={"type": "knowledge_dataset", "size": "10000_samples"},
            evaluation_criteria={"metric": "knowledge_score", "minimum": 0.8},
            reward_pool=reward,
            knowledge_target=knowledge_target,
            deadline=datetime.now() + timedelta(hours=24),
            status="proposed"
        )
        
        success = await consensus.propose_training_task(task)
        if success:
            click.echo(f"WTM task {task_id} proposed successfully")
        else:
            click.echo(f"Failed to propose WTM task {task_id}")
    
    asyncio.run(propose())

@cli.command()
@click.option('--query', required=True, help='Knowledge query')
@click.option('--domain', help='Knowledge domain to search')
def query_wtm(query, domain):
    """Query the World Truth Model knowledge base"""
    click.echo(f"Querying WTM: {query}")
    if domain:
        click.echo(f"Domain: {domain}")
    
    async def run_query():
        from config import Config
        from wtm import WorldTruthModel
        
        config = Config.load('config.yaml')
        wtm = WorldTruthModel(config)
        await wtm.initialize()
        
        results = await wtm.query_knowledge(query, domain)
        
        if results:
            click.echo(f"Found {len(results)} results:")
            for i, result in enumerate(results[:10], 1):
                if result["type"] == "entity":
                    entity = result["entity"]
                    click.echo(f"{i}. Entity: {entity.name}")
                    click.echo(f"   Type: {entity.entity_type}")
                    click.echo(f"   Confidence: {entity.confidence_score:.2f}")
                    click.echo(f"   Domain: {result.get('domain', 'unknown')}")
                    if entity.ml_contributions:
                        click.echo(f"   ML Contributors: {len(entity.ml_contributions)}")
                    click.echo()
                elif result["type"] == "fact":
                    fact = result["fact"]
                    click.echo(f"{i}. Fact: {fact.statement}")
                    click.echo(f"   Domain: {fact.domain}")
                    click.echo(f"   Confidence: {fact.confidence_score:.2f}")
                    if fact.ml_evidence:
                        click.echo(f"   ML Evidence: {len(fact.ml_evidence)} sources")
                    click.echo()
        else:
            click.echo("No results found in WTM")
    
    asyncio.run(run_query())

@cli.command()
def status():
    """Show WTM node status"""
    click.echo("WTM Node Status")
    click.echo("=" * 50)
    
    # Check if output directory exists and show file counts
    output_dir = Path("output")
    if output_dir.exists():
        files = list(output_dir.glob("*.json"))
        click.echo(f"Total output files: {len(files)}")
        
        # Count different WTM file types
        block_files = len(list(output_dir.glob("block_*.json")))
        experiment_files = len(list(output_dir.glob("wtm_experiment_*.json")))
        model_files = len(list(output_dir.glob("wtm_model_*.json")))
        task_files = len(list(output_dir.glob("wtm_task_*.json")))
        knowledge_files = len(list(output_dir.glob("wtm_*.json")))
        
        click.echo(f"Blockchain blocks: {block_files}")
        click.echo(f"ML experiments: {experiment_files}")
        click.echo(f"Model artifacts: {model_files}")
        click.echo(f"Training tasks: {task_files}")
        click.echo(f"Knowledge files: {knowledge_files}")
    else:
        click.echo("Output directory not found - WTM node hasn't been started yet")

@cli.command()
def domain_stats():
    """Show WTM knowledge domain statistics"""
    click.echo("WTM Knowledge Domain Statistics")
    click.echo("=" * 50)
    
    async def get_stats():
        from config import Config
        from wtm import WorldTruthModel
        
        config = Config.load('config.yaml')
        wtm = WorldTruthModel(config)
        await wtm.initialize()
        
        stats = await wtm.get_domain_statistics()
        
        if stats:
            for domain_id, domain_stats in stats.items():
                click.echo(f"Domain: {domain_stats['name']}")
                click.echo(f"  Entities: {domain_stats['entity_count']}")
                click.echo(f"  Facts: {domain_stats['fact_count']}")
                click.echo(f"  Completeness: {domain_stats['completeness']:.1%}")
                click.echo(f"  ML Contributors: {domain_stats['ml_contributors']}")
                click.echo(f"  Last Updated: {domain_stats['last_updated']}")
                click.echo()
        else:
            click.echo("No domain statistics available")
    
    asyncio.run(get_stats())

@cli.command()
def list_files():
    """List all WTM output files"""
    output_dir = Path("output")
    if not output_dir.exists():
        click.echo("No output directory found")
        return
    
    files = sorted(output_dir.glob("*.json"))
    if files:
        click.echo("WTM Output Files:")
        click.echo("-" * 40)
        
        for file in files:
            size = file.stat().st_size
            modified = datetime.fromtimestamp(file.stat().st_mtime)
            
            # Categorize file type
            if file.name.startswith("wtm_"):
                category = "WTM"
            elif file.name.startswith("block_"):
                category = "Blockchain"
            else:
                category = "Other"
                
            click.echo(f"[{category}] {file.name}")
            click.echo(f"  Size: {size} bytes")
            click.echo(f"  Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
            click.echo()
    else:
        click.echo("No WTM output files found")

@cli.command()
@click.option('--sender', default='cli_user', help='Transaction sender')
@click.option('--recipient', required=True, help='Transaction recipient')
@click.option('--amount', default=100, help='Transaction amount in WTM tokens')
@click.option('--tx-type', default='token_transfer', help='Transaction type')
def send_transaction(sender, recipient, amount, tx_type):
    """Send a WTM transaction"""
    click.echo(f"Sending {amount} WTM tokens from {sender} to {recipient}")
    
    async def send_tx():
        from config import Config
        from blockchain import WTMChain, Transaction
        
        config = Config.load('config.yaml')
        blockchain = WTMChain(config)
        await blockchain.initialize()
        
        # Create WTM transaction
        tx = Transaction(
            tx_id=generate_unique_id("wtm_tx_"),
            tx_type=tx_type,
            sender=sender,
            recipient=recipient,
            amount=amount,
            data={"memo": "WTM CLI transaction", "cli_version": "0.8"},
            timestamp=current_timestamp(),
            signature="wtm_cli_signature"
        )
        
        await blockchain.add_transaction(tx)
        click.echo(f"WTM transaction {tx.tx_id} added to mempool")
    
    asyncio.run(send_tx())

@cli.command()
@click.option('--entity-name', required=True, help='Entity name')
@click.option('--entity-type', required=True, help='Entity type')
@click.option('--domain', default='general', help='Knowledge domain')
def add_knowledge(entity_name, entity_type, domain):
    """Add knowledge entity to WTM"""
    click.echo(f"Adding knowledge entity: {entity_name}")
    click.echo(f"Type: {entity_type}, Domain: {domain}")
    
    async def add_entity():
        from config import Config
        from wtm import WorldTruthModel, KnowledgeEntity
        
        config = Config.load('config.yaml')
        wtm = WorldTruthModel(config)
        await wtm.initialize()
        
        # Create knowledge entity
        entity = KnowledgeEntity(
            entity_id=generate_unique_id("cli_entity_"),
            name=entity_name,
            entity_type=entity_type,
            confidence_score=0.8,  # CLI additions get medium confidence
            attributes={"source": "cli_user", "created_via": "wtm_cli"},
            sources=["CLI input"],
            ml_contributions=[],
            last_updated=datetime.now()
        )
        
        await wtm.add_entity(entity)
        click.echo(f"Knowledge entity added to WTM with ID: {entity.entity_id}")
    
    asyncio.run(add_entity())

@cli.command()
@click.option('--model-id', required=True, help='Model ID to extract knowledge from')
def extract_knowledge(model_id):
    """Extract knowledge from a trained model"""
    click.echo(f"Extracting knowledge from model: {model_id}")
    
    async def extract():
        from config import Config
        from mlops import MLOpsOrchestrator
        
        config = Config.load('config.yaml')
        mlops = MLOpsOrchestrator(config)
        await mlops.initialize()
        
        knowledge = await mlops.extract_knowledge_from_model(model_id)
        
        if knowledge:
            click.echo("Knowledge extracted:")
            click.echo(f"  Entities: {len(knowledge.get('entities', []))}")
            click.echo(f"  Relationships: {len(knowledge.get('relationships', []))}")
            click.echo(f"  Confidence: {knowledge.get('confidence', 0):.2f}")
        else:
            click.echo("No knowledge extracted or model not found")
    
    asyncio.run(extract())

if __name__ == '__main__':
    cli() 