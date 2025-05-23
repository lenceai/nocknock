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