# Domain-LLM Examples

This directory contains example implementations of Domain-LLM for different domains.

## Available Examples

- **Sprint**: An implementation for the track and field sprint domain, with data processing, model training, and inference for sprint-related data.

## Using the Examples

Each example directory contains:

1. Data processing scripts
2. Training scripts
3. Inference examples
4. Sample data generators

### Running an Example

To run a complete example:

1. Navigate to the example directory:
   ```
   cd domain_llm/examples/sprint
   ```

2. Generate sample data:
   ```
   ./data/generate_sample_data.py
   ```

3. Process the data:
   ```
   domain-llm process --data-dir content --output-dir data/processed
   ```

4. Train a model:
   ```
   domain-llm train --data-dir data/processed --model-dir models --model-name gpt2 --epochs 3
   ```

5. Use the model:
   ```
   domain-llm generate --model-dir models/gpt2-sprint --prompt "What was Usain Bolt's best time?"
   ```

6. Alternative: Run the entire pipeline:
   ```
   ./run_pipeline.py
   ```

## Creating Your Own Domain Implementation

To create a custom domain implementation:

1. Create a new directory for your domain:
   ```
   mkdir -p domain_llm/examples/your_domain_name/data
   ```

2. Create the necessary scripts:
   - Data processing script
   - Sample data generator
   - Pipeline runner

3. Implement the domain-specific processing logic

4. Train and test your model

For more detailed information, see the main [Domain-LLM documentation](../../README.md). 