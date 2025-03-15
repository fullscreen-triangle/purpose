# Sprint Domain Example

This example demonstrates how to use Domain-LLM to create a domain-specific language model for track and field sprint information.

## Overview

The sprint domain example includes:

1. Sample data generation for sprint races, athlete profiles, and biomechanics
2. Data processing specific to sprint-related information
3. Model training configuration for sprint data
4. Example inference with sprint-specific questions

## Getting Started

### Generate Sample Data

First, generate sample sprint data:

```bash
./data/generate_sample_data.py
```

This will create a `content/` directory with CSV and JSON files containing sprint race results, athlete information, and biomechanical data.

### Processing the Data

Process the sprint data to prepare it for training:

```bash
domain-llm process --data-dir content --output-dir data/processed
```

This will:
- Extract sprint-specific information from the CSV and JSON files
- Create a text corpus formatted for language model training
- Save processed records in a structured format

### Training a Model

Train a language model on the processed sprint data:

```bash
domain-llm train --data-dir data/processed --model-dir models --model-name gpt2 --epochs 3
```

Options:
- `--model-name`: Base model to fine-tune (default: gpt2)
- `--batch-size`: Training batch size (default: 4)
- `--learning-rate`: Learning rate (default: 5e-5)
- `--epochs`: Number of training epochs (default: 3)
- `--use-lora`: Use LoRA for parameter-efficient fine-tuning (default: False)

### Using the Model

Generate responses to sprint-related questions:

```bash
domain-llm generate --model-dir models/gpt2-sprint --prompt "What's the average stride length for elite sprinters?"
```

### Running the Complete Pipeline

To run the entire process in one command:

```bash
./run_pipeline.py
```

Options:
- `--skip-processing`: Skip the data processing step
- `--skip-training`: Skip the model training step
- `--batch-size`: Training batch size
- `--epochs`: Number of training epochs

## Understanding the Sprint Domain

This example focuses on these key aspects of sprint running:

1. **Race Results**: Times, positions, and competition details
2. **Athlete Profiles**: Personal bests, physical attributes, and career stats
3. **Biomechanics**: Stride length, frequency, ground contact time, and other technical metrics
4. **Performance Analysis**: Reaction times, acceleration phases, and race breakdowns

## Customizing the Sprint Example

To adapt this example to your own sprint data:

1. Place your CSV and JSON files in the `content/` directory
2. Modify `process_content_data.py` if needed to handle your specific data formats
3. Adjust training parameters in `train_sprint_llm.py` based on your dataset size
4. Update prompts in inference examples to match your domain questions

For more information on customizing Domain-LLM, see the main documentation. 