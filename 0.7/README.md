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