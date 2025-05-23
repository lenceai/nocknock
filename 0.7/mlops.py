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