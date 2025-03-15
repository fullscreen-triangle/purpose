# Domain-LLM

Domain-LLM is a framework for creating domain-specific language models by fine-tuning base models on specialized data. It provides a complete pipeline for data processing, model training, and inference.

## Features

- **Flexible Data Processing**: Process various data formats (CSV, JSON) into training-ready text
- **Streamlined Training**: Fine-tune LLMs with optimized parameters for domain adaptation
- **Inference Tools**: Generate domain-specific responses with fine-tuned models
- **Command-Line Interface**: Easy-to-use CLI for the entire workflow
- **Domain Examples**: Ready-to-use examples for specific domains (sports, etc.)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/domain-llm.git
cd domain-llm

# Install in development mode
pip install -e .
```

## Quick Start

### 1. Process Your Domain Data

```bash
domain-llm process --data-dir your_data_directory --output-dir data/processed
```

### 2. Train a Domain-Specific Model

```bash
domain-llm train --data-dir data/processed --model-dir models --model-name gpt2
```

### 3. Generate with Your Model

```bash
domain-llm generate --model-dir models/gpt2-domain --prompt "Your domain-specific question"
```

## Command-Line Interface

Domain-LLM provides a unified CLI with several commands:

### List Available Models

```bash
domain-llm list-models
```

### Process Domain Data

```bash
domain-llm process --data-dir INPUT_DIR --output-dir OUTPUT_DIR
```

Options:
- `--data-dir`: Directory containing domain data files
- `--output-dir`: Directory for processed data output
- `--log-level`: Logging level (default: INFO)

### Train a Model

```bash
domain-llm train --data-dir DATA_DIR --model-dir MODEL_DIR
```

Options:
- `--data-dir`: Directory with processed data
- `--model-dir`: Directory to save the trained model
- `--model-name`: Base model to fine-tune (default: gpt2)
- `--batch-size`: Training batch size (default: 4)
- `--learning-rate`: Learning rate (default: 5e-5)
- `--epochs`: Number of training epochs (default: 3)
- `--use-lora`: Use LoRA for parameter-efficient fine-tuning (default: False)
- `--force-cpu`: Force using CPU even if GPU is available (default: False)

### Generate Text

```bash
domain-llm generate --model-dir MODEL_DIR --prompt "Your prompt"
```

Options:
- `--model-dir`: Directory containing the fine-tuned model
- `--prompt`: Input prompt for generation
- `--max-length`: Maximum output length (default: 100)
- `--temperature`: Sampling temperature (default: 0.7)
- `--num-return`: Number of responses to generate (default: 1)
- `--interactive`: Start interactive mode (default: False)

## Example Domains

The package includes ready-to-use examples for specific domains:

- **Sprint Running**: Data processing and model training for sprint-related information
  - See [Sprint Example Documentation](domain_llm/examples/sprint/README.md)

To run a complete example:

```bash
# Navigate to the example directory
cd domain_llm/examples/sprint

# Generate sample data
./data/generate_sample_data.py

# Run the complete pipeline
./run_pipeline.py
```

## Creating Your Own Domain Implementation

1. Create a new directory for your domain:
   ```
   mkdir -p domain_llm/examples/your_domain/data
   ```

2. Create the necessary files:
   - Data processing script
   - Training configuration
   - Sample data generator (optional)
   - Pipeline runner (optional)

3. Use the CLI to process and train:
   ```
   domain-llm process --data-dir your_data_dir --output-dir data/processed
   domain-llm train --data-dir data/processed --model-dir models
   ```

## Package Architecture

- **domain_llm/processing/**: Data processing modules
- **domain_llm/training/**: Model training modules
- **domain_llm/inference/**: Text generation and inference
- **domain_llm/examples/**: Domain-specific implementations
- **domain_llm/cli/**: Command-line interface modules

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face Transformers
- PyTorch
- The open-source NLP community
