"""
World Truth Model (WTM) Implementation v0.8
The core global knowledge base with verifiable facts
Enhanced with ML-driven knowledge extraction and validation
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
    ml_contributions: List[str]  # ML models that contributed to this entity
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
    ml_evidence: List[str]  # Evidence from ML models
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
    ml_contributors: List[str]  # ML models contributing to this domain
    knowledge_completeness: float  # How complete our knowledge is in this domain
    last_updated: datetime

@dataclass
class KnowledgeUpdate:
    """Knowledge update proposal"""
    update_id: str
    update_type: str  # entity_update, fact_addition, fact_correction
    proposer: str
    content: Dict[str, Any]
    confidence: float
    ml_source: Optional[str]  # ML model that generated this update
    validation_status: str  # pending, approved, rejected
    created_at: datetime

class WorldTruthModel:
    """Enhanced World Truth Model implementation"""
    
    def __init__(self, config):
        self.config = config
        self.entities: Dict[str, KnowledgeEntity] = {}
        self.facts: Dict[str, KnowledgeFact] = {}
        self.domains: Dict[str, KnowledgeDomain] = {}
        self.update_queue: List[KnowledgeUpdate] = []
        self.ml_integration_active = True
        
        # Initialize core domains
        self.initialize_core_domains()
        
    def initialize_core_domains(self):
        """Initialize core knowledge domains for WTM"""
        core_domains = [
            ("science", "Scientific Knowledge", "Verified scientific facts, theories, and discoveries"),
            ("technology", "Technology & AI", "Technological advances, AI developments, and innovations"),
            ("history", "Historical Events", "Documented historical events and figures"),
            ("geography", "Geographic Information", "Geographic locations, boundaries, and features"),
            ("mathematics", "Mathematical Knowledge", "Mathematical theorems, formulas, and constants"),
            ("current_events", "Current Events", "Recent news and developments"),
            ("culture", "Cultural Knowledge", "Art, literature, traditions, and cultural phenomena"),
            ("economics", "Economic Data", "Economic indicators, markets, and financial systems"),
            ("medicine", "Medical Knowledge", "Medical facts, treatments, and health information"),
            ("environment", "Environmental Science", "Climate, ecology, and environmental data")
        ]
        
        for domain_id, name, description in core_domains:
            domain = KnowledgeDomain(
                domain_id=domain_id,
                name=name,
                description=description,
                entities=[],
                facts=[],
                expert_validators=[],
                ml_contributors=[],
                knowledge_completeness=0.1,  # Start with low completeness
                last_updated=datetime.now()
            )
            self.domains[domain_id] = domain
            
    async def initialize(self):
        """Initialize World Truth Model"""
        logger.info("Initializing World Truth Model v0.8...")
        
        # Load existing knowledge base
        await self.load_knowledge_base()
        
        # Initialize with enhanced basic facts
        await self.bootstrap_enhanced_knowledge()
        
        logger.info(f"WTM v0.8 initialized with {len(self.entities)} entities and {len(self.facts)} facts across {len(self.domains)} domains")
        
    async def load_knowledge_base(self):
        """Load existing knowledge base from disk"""
        try:
            # Load entities
            entities_file = Path("output/wtm_entities.json")
            if entities_file.exists():
                with open(entities_file, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded {len(data)} entities from WTM knowledge base")
                    
            # Load facts
            facts_file = Path("output/wtm_facts.json")
            if facts_file.exists():
                with open(facts_file, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded {len(data)} facts from WTM knowledge base")
                    
        except Exception as e:
            logger.error(f"Error loading WTM knowledge base: {e}")
            
    async def bootstrap_enhanced_knowledge(self):
        """Bootstrap with enhanced knowledge including AI/ML facts"""
        # Add fundamental physics constants
        await self.add_entity(KnowledgeEntity(
            entity_id="speed_of_light",
            name="Speed of Light",
            entity_type="physical_constant",
            confidence_score=1.0,
            attributes={
                "value": 299792458,
                "units": "meters per second",
                "symbol": "c",
                "discovered_year": 1676,
                "significance": "Universal speed limit, fundamental to relativity"
            },
            sources=["CODATA 2018", "Einstein 1905"],
            ml_contributions=[],
            last_updated=datetime.now()
        ))
        
        # Add AI/ML knowledge
        await self.add_entity(KnowledgeEntity(
            entity_id="machine_learning",
            name="Machine Learning",
            entity_type="field_of_study",
            confidence_score=1.0,
            attributes={
                "definition": "Field of AI that enables computers to learn without explicit programming",
                "founded_year": 1959,
                "key_concepts": ["supervised_learning", "unsupervised_learning", "neural_networks"],
                "applications": ["computer_vision", "nlp", "robotics", "data_analysis"]
            },
            sources=["Arthur Samuel 1959", "Academic literature"],
            ml_contributions=[],
            last_updated=datetime.now()
        ))
        
        # Add mathematical constants
        await self.add_entity(KnowledgeEntity(
            entity_id="pi",
            name="Pi",
            entity_type="mathematical_constant",
            confidence_score=1.0,
            attributes={
                "value": 3.141592653589793,
                "symbol": "π",
                "definition": "ratio of circle circumference to diameter",
                "decimal_places_computed": "100 trillion+",
                "irrationality_proven": 1761
            },
            sources=["Mathematical definition", "Lambert 1761"],
            ml_contributions=[],
            last_updated=datetime.now()
        ))
        
        # Add corresponding facts
        await self.add_fact(KnowledgeFact(
            fact_id="speed_of_light_value",
            statement="The speed of light in vacuum is exactly 299,792,458 meters per second",
            subject_entity="speed_of_light",
            predicate="has_exact_value",
            object_entity="299792458_mps",
            confidence_score=1.0,
            evidence=["CODATA 2018 fundamental physical constants", "Meter definition"],
            ml_evidence=[],
            domain="science",
            created_at=datetime.now(),
            verified_by=["physics_community", "NIST"]
        ))
        
        await self.add_fact(KnowledgeFact(
            fact_id="ml_learning_paradigm",
            statement="Machine learning enables systems to automatically improve performance through experience",
            subject_entity="machine_learning",
            predicate="enables",
            object_entity="automatic_performance_improvement",
            confidence_score=0.95,
            evidence=["Arthur Samuel 1959", "Tom Mitchell 1997", "Academic consensus"],
            ml_evidence=[],
            domain="technology",
            created_at=datetime.now(),
            verified_by=["ai_research_community"]
        ))
        
        logger.info("Bootstrapped enhanced WTM knowledge base")
        
    async def add_entity(self, entity: KnowledgeEntity):
        """Add entity to WTM knowledge base"""
        self.entities[entity.entity_id] = entity
        
        # Add to appropriate domain
        domain_id = self.determine_entity_domain(entity)
        if domain_id in self.domains:
            if entity.entity_id not in self.domains[domain_id].entities:
                self.domains[domain_id].entities.append(entity.entity_id)
                # Update domain completeness
                self.update_domain_completeness(domain_id)
                
        logger.info(f"Added WTM entity: {entity.name}")
        
    async def add_fact(self, fact: KnowledgeFact):
        """Add fact to WTM knowledge base"""
        self.facts[fact.fact_id] = fact
        
        # Add to appropriate domain
        if fact.domain in self.domains:
            if fact.fact_id not in self.domains[fact.domain].facts:
                self.domains[fact.domain].facts.append(fact.fact_id)
                # Update domain completeness
                self.update_domain_completeness(fact.domain)
                
        logger.info(f"Added WTM fact: {fact.statement[:100]}...")
        
    def determine_entity_domain(self, entity: KnowledgeEntity) -> str:
        """Determine which domain an entity belongs to"""
        entity_type_mapping = {
            "physical_constant": "science",
            "mathematical_constant": "mathematics",
            "historical_figure": "history",
            "geographic_location": "geography",
            "current_event": "current_events",
            "field_of_study": "science",
            "technology": "technology",
            "ai_model": "technology",
            "cultural_artifact": "culture",
            "economic_indicator": "economics",
            "medical_condition": "medicine",
            "environmental_factor": "environment"
        }
        
        return entity_type_mapping.get(entity.entity_type, "general")
        
    def update_domain_completeness(self, domain_id: str):
        """Update the completeness score for a domain"""
        if domain_id in self.domains:
            domain = self.domains[domain_id]
            entity_count = len(domain.entities)
            fact_count = len(domain.facts)
            
            # Simple completeness calculation - could be made more sophisticated
            completeness = min(1.0, (entity_count + fact_count) / 1000)
            domain.knowledge_completeness = completeness
            domain.last_updated = datetime.now()
        
    async def query_knowledge(self, query: str, domain: Optional[str] = None) -> List[Dict]:
        """Enhanced query system for WTM knowledge base"""
        results = []
        
        # Simple keyword-based search with enhanced scoring
        query_lower = query.lower()
        
        # Search entities with weighted scoring
        for entity in self.entities.values():
            relevance_score = 0.0
            
            # Name match (highest weight)
            if query_lower in entity.name.lower():
                relevance_score += 1.0
                
            # Attribute match (medium weight)
            for attr_value in entity.attributes.values():
                if isinstance(attr_value, str) and query_lower in attr_value.lower():
                    relevance_score += 0.5
                elif isinstance(attr_value, list):
                    for item in attr_value:
                        if isinstance(item, str) and query_lower in item.lower():
                            relevance_score += 0.3
            
            if relevance_score > 0 and (not domain or self.determine_entity_domain(entity) == domain):
                results.append({
                    "type": "entity",
                    "entity": entity,
                    "confidence": entity.confidence_score,
                    "relevance": relevance_score,
                    "domain": self.determine_entity_domain(entity)
                })
                    
        # Search facts with relevance scoring
        for fact in self.facts.values():
            relevance_score = 0.0
            
            if query_lower in fact.statement.lower():
                relevance_score += 1.0
            
            if relevance_score > 0 and (not domain or fact.domain == domain):
                results.append({
                    "type": "fact", 
                    "fact": fact,
                    "confidence": fact.confidence_score,
                    "relevance": relevance_score,
                    "domain": fact.domain
                })
                    
        # Sort by relevance and confidence
        results.sort(key=lambda x: (x["relevance"], x["confidence"]), reverse=True)
        
        logger.info(f"WTM query '{query}' returned {len(results)} results")
        return results[:20]  # Return top 20 results
        
    async def integrate_ml_knowledge(self, ml_model_id: str, knowledge_data: Dict[str, Any]) -> int:
        """Integrate knowledge from ML models into WTM"""
        updates_processed = 0
        
        logger.info(f"Integrating knowledge from ML model {ml_model_id}")
        
        # Process entity discoveries
        if "entities" in knowledge_data:
            for entity_data in knowledge_data["entities"]:
                update = KnowledgeUpdate(
                    update_id=f"ml_{ml_model_id}_{updates_processed}",
                    update_type="entity_addition",
                    proposer="ml_system",
                    content=entity_data,
                    confidence=entity_data.get("confidence", 0.5),
                    ml_source=ml_model_id,
                    validation_status="pending",
                    created_at=datetime.now()
                )
                self.update_queue.append(update)
                updates_processed += 1
        
        # Process fact discoveries
        if "facts" in knowledge_data:
            for fact_data in knowledge_data["facts"]:
                update = KnowledgeUpdate(
                    update_id=f"ml_{ml_model_id}_{updates_processed}",
                    update_type="fact_addition",
                    proposer="ml_system",
                    content=fact_data,
                    confidence=fact_data.get("confidence", 0.5),
                    ml_source=ml_model_id,
                    validation_status="pending",
                    created_at=datetime.now()
                )
                self.update_queue.append(update)
                updates_processed += 1
        
        logger.info(f"Queued {updates_processed} knowledge updates from ML model {ml_model_id}")
        return updates_processed
        
    async def validate_knowledge_update(self, update: KnowledgeUpdate) -> bool:
        """Validate a knowledge update proposal"""
        # ML-generated updates require higher confidence threshold
        if update.ml_source:
            threshold = 0.7
        else:
            threshold = 0.5
            
        if update.confidence < threshold:
            update.validation_status = "rejected"
            return False
            
        # Additional validation logic would go here
        # For now, approve updates above threshold
        update.validation_status = "approved"
        return True
        
    async def process_updates(self):
        """Process pending updates to the WTM knowledge base"""
        processed_updates = []
        
        for update in self.update_queue[:10]:  # Process up to 10 updates at a time
            try:
                if await self.validate_knowledge_update(update):
                    await self.apply_knowledge_update(update)
                processed_updates.append(update)
            except Exception as e:
                logger.error(f"Error processing WTM update: {e}")
                
        # Remove processed updates
        for update in processed_updates:
            self.update_queue.remove(update)
            
        # Save state periodically
        if len(processed_updates) > 0:
            await self.save_knowledge_base()
            
        return len(processed_updates)
        
    async def apply_knowledge_update(self, update: KnowledgeUpdate):
        """Apply an approved knowledge update to WTM"""
        if update.validation_status != "approved":
            return
            
        if update.update_type == "entity_addition":
            # Create new entity from ML data
            entity_data = update.content
            entity = KnowledgeEntity(
                entity_id=entity_data["entity_id"],
                name=entity_data["name"],
                entity_type=entity_data.get("type", "ml_discovered"),
                confidence_score=update.confidence,
                attributes=entity_data.get("attributes", {}),
                sources=entity_data.get("sources", []),
                ml_contributions=[update.ml_source] if update.ml_source else [],
                last_updated=datetime.now()
            )
            await self.add_entity(entity)
            
        elif update.update_type == "fact_addition":
            # Create new fact from ML data
            fact_data = update.content
            fact = KnowledgeFact(
                fact_id=fact_data["fact_id"],
                statement=fact_data["statement"],
                subject_entity=fact_data.get("subject", ""),
                predicate=fact_data.get("predicate", ""),
                object_entity=fact_data.get("object", ""),
                confidence_score=update.confidence,
                evidence=fact_data.get("evidence", []),
                ml_evidence=[update.ml_source] if update.ml_source else [],
                domain=fact_data.get("domain", "general"),
                created_at=datetime.now(),
                verified_by=["ml_system"]
            )
            await self.add_fact(fact)
            
    async def get_domain_statistics(self) -> Dict[str, Any]:
        """Get statistics about WTM knowledge domains"""
        stats = {}
        
        for domain_id, domain in self.domains.items():
            stats[domain_id] = {
                "name": domain.name,
                "entity_count": len(domain.entities),
                "fact_count": len(domain.facts),
                "completeness": domain.knowledge_completeness,
                "ml_contributors": len(domain.ml_contributors),
                "last_updated": domain.last_updated.isoformat()
            }
            
        return stats
        
    async def save_knowledge_base(self):
        """Save WTM knowledge base to disk"""
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
                    "ml_contributions": entity.ml_contributions,
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
                    "ml_evidence": fact.ml_evidence,
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
                    "ml_contributors": domain.ml_contributors,
                    "knowledge_completeness": domain.knowledge_completeness,
                    "last_updated": domain.last_updated.isoformat()
                }
                
            with open("output/wtm_domains.json", 'w') as f:
                json.dump(domains_data, f, indent=2)
                
            logger.info("Saved WTM knowledge base to disk")
            
        except Exception as e:
            logger.error(f"Error saving WTM knowledge base: {e}") 