# Sprint Domain Model Enhancement

This directory contains advanced techniques for improving the domain-specific sprint models using large language model APIs (OpenAI GPT-4, Claude) to generate high-quality training data.

## Overview

We provide two complementary approaches to enhance your sprint domain models:

1. **Knowledge Distillation**: Generate question-answer pairs using large LLMs and use them to train a smaller domain-specific model
2. **Model Optimization**: Focus on generating formal mathematical models of sprint phenomena rather than simple QA pairs
3. **Combined Approach**: Use both types of data for a more comprehensive domain model

## Knowledge Distillation

The knowledge distillation approach generates high-quality QA pairs using large models like GPT-4 or Claude, then trains your domain-specific model on this data.

### Key Features

- Generates synthetic questions covering diverse aspects of sprint science
- Queries large models for expert-quality answers
- Structures data for optimal learning
- Creates a training corpus automatically
- Integrates with your existing training pipeline

### How to Use

```bash
# Run the knowledge distillation pipeline with default settings
python -m purpose.examples.sprint.knowledge_distill --num-samples 50 --target-model gpt-4

# Set API keys via environment variables
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Or provide them directly (not recommended for security)
python -m purpose.examples.sprint.knowledge_distill --openai-key "your-key" --anthropic-key "your-key"
```

## Model Optimization

This approach focuses on generating rigorous mathematical models of sprint phenomena rather than simple Q&A pairs. This results in a model that understands the formal, mathematical representation of sprint concepts.

### Key Features

- Generates formal mathematical models of sprint phenomena
- Includes complete equations, parameter definitions, and boundary conditions
- Provides implementation code for each model
- Includes validation approaches and scientific references
- Structured as formal mathematical papers

### How to Use

```bash
# Run the model optimization pipeline with default settings
python -m purpose.examples.sprint.model_optimization --num-samples 30 --model-type mathematical

# Choose different model types
python -m purpose.examples.sprint.model_optimization --model-type statistical
python -m purpose.examples.sprint.model_optimization --model-type biomechanical
```

## Combined Enhanced Training

The enhanced training pipeline combines both approaches for a more comprehensive domain model.

### How to Use

```bash
# Run the complete enhanced training pipeline
python -m purpose.examples.sprint.enhanced_training --run-all --api-keys-from-env

# Run specific phases only
python -m purpose.examples.sprint.enhanced_training --run-knowledge-distill --create-combined --train-model
```

## Implementation Details

### Knowledge Distillation Implementation

The knowledge distillation approach is implemented in `knowledge_distill.py`. The main class is `KnowledgeDistiller`, which:

1. Generates synthetic questions about sprint science
2. Queries large models (GPT-4/Claude) to get expert-quality answers
3. Creates a training corpus from the QA pairs
4. Trains a domain-specific model on this corpus

The question generation uses carefully designed templates covering:
- Biomechanics of sprinting
- Physiological aspects
- Training methodology
- Performance analysis
- Equipment and technology

### Model Optimization Implementation

The model optimization approach is implemented in `model_optimization.py`. The main class is `ModelOptimizer`, which:

1. Generates prompts for creating mathematical models of sprint phenomena
2. Queries large models to get formal mathematical formulations
3. Creates a training corpus from these model formulations
4. Trains a model on this corpus of formal models

The model templates cover:
- Sprint velocity models
- Wind effect models
- Stride parameter models
- Ground reaction force models
- Energy expenditure models
- Race strategy optimization models
- And more...

Each model includes:
- Formal mathematical equations
- Parameter definitions with units
- Boundary conditions
- Validation approaches
- Implementation code
- Scientific references

## API Requirements

Both approaches require API access to large language models:

- OpenAI API (for GPT-4 or GPT-3.5-turbo)
- Anthropic API (for Claude models)

You can set your API keys as environment variables:
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## Output Files

The pipelines create several output directories:

- `data/distill_corpus/`: Contains the knowledge distillation corpus
- `data/model_corpus/`: Contains the model optimization corpus
- `data/combined_corpus/`: Contains the combined corpus
- `model_data/`: Contains individual model JSON files
- `models/`: Contains the trained models

## Advanced Customization

You can customize both approaches by:

- Adding new question templates to `KnowledgeDistiller._load_question_templates()`
- Adding new model templates to `ModelOptimizer._load_model_templates()`
- Modifying the system prompts for each approach
- Adjusting the parameters for model generation

## References

The implementation draws on research in:
- Knowledge distillation for language models
- Mathematical modeling in sports biomechanics
- Meta-learning and model-based representation 