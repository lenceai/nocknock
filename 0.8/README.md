# WTM v0.8 - World Truth Model

A decentralized MLOps blockchain focused on building the World Truth Model through collaborative intelligence. WTM transforms blockchain mining from wasteful computation into meaningful machine learning work that contributes to a global knowledge base.

## Overview

WTM (World Truth Model) is the evolution of decentralized AI, where every computation contributes to building humanity's most comprehensive and verifiable knowledge base. Instead of solving arbitrary puzzles, network participants compete to train the best ML models, with all work contributing to the global World Truth Model.

## Key Features

- **Knowledge-Centric Mining**: All computational work contributes to building the World Truth Model
- **zkMLOps Consensus**: Proof-of-useful-work through ML training competitions
- **Global Knowledge Base**: Comprehensive, verifiable facts across all domains of human knowledge
- **ML-Driven Discovery**: Automated knowledge extraction and validation from trained models
- **Domain Expertise**: Specialized knowledge domains (science, technology, history, etc.)
- **High Performance**: Solana-inspired architecture for fast transaction processing
- **Privacy-Preserving**: Zero-knowledge proofs protect sensitive data and methods

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running a WTM Node

```bash
# Start a WTM node
python main.py

# Or use the CLI
python cli.py start
```

### Interacting with the World Truth Model

```bash
# Query the knowledge base
python cli.py query-wtm --query "speed of light" --domain science

# Create a knowledge-focused experiment
python cli.py create-experiment --name "physics-discovery" --model-type knowledge_extraction --dataset "physics-papers" --knowledge-target science

# Propose a training task
python cli.py propose-task --task-id "task-001" --model-type knowledge_extraction --reward 1000 --knowledge-target technology

# Check domain statistics
python cli.py domain-stats

# Add knowledge manually
python cli.py add-knowledge --entity-name "Quantum Computing" --entity-type technology --domain technology
```

## Architecture

### Core Components

- **World Truth Model** (`wtm.py`): The central knowledge base with entities, facts, and domains
- **Blockchain Layer** (`blockchain.py`): High-performance transaction processing with knowledge hashes
- **zkMLOps Consensus** (`consensus.py`): Useful work consensus through ML training competitions  
- **MLOps Orchestration** (`mlops.py`): Knowledge-driven ML experiment management
- **P2P Network** (`network.py`): Knowledge synchronization and communication
- **CLI Interface** (`cli.py`): Command-line tools for knowledge interaction

### Knowledge Domains

WTM organizes knowledge into specialized domains:

- **Science**: Physics, chemistry, biology, and scientific discoveries
- **Technology**: AI, computing, engineering, and technological advances
- **Mathematics**: Theorems, formulas, constants, and mathematical knowledge
- **History**: Historical events, figures, and documented facts
- **Geography**: Locations, boundaries, and geographic information
- **Culture**: Art, literature, traditions, and cultural phenomena
- **Economics**: Markets, indicators, and economic systems
- **Medicine**: Medical knowledge, treatments, and health information
- **Environment**: Climate, ecology, and environmental science
- **Current Events**: Recent developments and news

### Key Innovations

1. **Knowledge-First Mining**: Every computation advances human knowledge
2. **ML-Driven Validation**: Automated fact-checking and knowledge extraction
3. **Domain Specialization**: Expert validation within knowledge domains
4. **Collaborative Intelligence**: Global cooperation on knowledge building
5. **Verifiable Facts**: All knowledge backed by cryptographic proofs

## Configuration

Edit `config.yaml` to customize:

- **Network settings**: Ports, peers, and connectivity
- **Blockchain parameters**: Block time, size limits, and consensus rules
- **MLOps configuration**: Storage paths, training timeouts, and concurrency
- **WTM settings**: Knowledge domains, confidence thresholds, and validation rules

## Output Files

All data is stored in the `output/` directory:

- `wtm_entities.json`: Knowledge entities in the World Truth Model
- `wtm_facts.json`: Verifiable facts and statements
- `wtm_domains.json`: Knowledge domain information and statistics
- `wtm_experiment_*.json`: ML experiments focused on knowledge extraction
- `wtm_model_*.json`: Model artifacts with extracted knowledge
- `wtm_task_*.json`: Training tasks for knowledge generation
- `block_*.json`: Blockchain blocks with knowledge hashes
- `wtm.log`: Node operation logs

## CLI Commands

### Knowledge Operations
- `query-wtm`: Search the World Truth Model
- `domain-stats`: View knowledge domain statistics
- `add-knowledge`: Manually add knowledge entities
- `extract-knowledge`: Extract knowledge from trained models

### ML Operations
- `create-experiment`: Create knowledge-focused ML experiments
- `propose-task`: Propose training tasks for knowledge generation

### Node Operations
- `start`: Start a WTM node
- `status`: Show node status and file counts
- `list-files`: List all output files
- `send-transaction`: Send WTM token transactions

## Development

### File Structure

```
wtm/0.8/
├── main.py           # WTM node entry point
├── wtm.py           # World Truth Model implementation
├── blockchain.py     # WTM blockchain with knowledge hashes
├── consensus.py      # zkMLOps consensus mechanism
├── mlops.py         # Knowledge-driven MLOps orchestration
├── network.py       # P2P networking with knowledge sync
├── cli.py           # Command-line interface
├── config.py        # Configuration management
├── config.yaml      # Default configuration
├── requirements.txt # Python dependencies
└── output/          # Generated files and knowledge base
```

### Testing

```bash
# Test basic functionality
python cli.py status

# Test knowledge queries
python cli.py query-wtm --query "machine learning"

# Test domain statistics
python cli.py domain-stats
```

## Contributing

WTM is open source under the MIT License. Contributions welcome!

1. Fork the repository
2. Create a feature branch
3. Focus on knowledge-related improvements
4. Add tests for knowledge validation
5. Submit a pull request

## Vision

WTM enables a future where:
- **Global Knowledge**: Humanity's knowledge is accessible to all
- **Collaborative Discovery**: AI development serves knowledge advancement
- **Verifiable Truth**: All facts are cryptographically verified
- **Useful Computation**: Every calculation contributes to human understanding
- **Domain Expertise**: Specialized knowledge communities drive validation
- **Open Science**: Research is transparent and reproducible

## License

MIT License - see LICENSE file for details.

---

**WTM v0.8** - Building the World Truth Model through collaborative intelligence.

*"In the age of information, truth becomes the most valuable currency."* 