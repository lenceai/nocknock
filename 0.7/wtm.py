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