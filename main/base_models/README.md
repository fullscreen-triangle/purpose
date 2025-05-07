# Purpose Models

This directory contains model implementations and configurations for the Purpose project.

## Specialized Models

The `specialized_models.py` module provides domain-specific model registrations for the Purpose ModelHub. These specialized models are optimized for specific domains and tasks:

- **Medical Models**: Clinical reasoning, medical knowledge extraction, patient-doctor dialogue
- **Legal Models**: Legal document analysis, case law understanding, compliance analysis
- **Financial Models**: Market analysis, financial sentiment, fiscal knowledge
- **Mathematical Models**: Theorem proving, mathematical reasoning, step-by-step explanations
- **Code Models**: Code generation, algorithm explanation, software development

## Usage

### Basic Usage

The specialized models are automatically registered when the ModelHub is initialized:

```python
from main.utils.model_hub import ModelHub

# Create ModelHub with specialized models
model_hub = ModelHub(load_specialized=True)

# Get recommended models for a task
models = model_hub.get_recommended_models("knowledge_extraction")
print(f"Recommended models: {models}")
```

### Domain-Specific Clients

You can create domain-specific clients that prioritize models from a particular domain:

```python
from main.utils.model_hub import PurposeAPIClient
from main.models.specialized_models import create_domain_specific_client

# Create medical domain-specific client
client = PurposeAPIClient(api_token="YOUR_API_TOKEN")
medical_config = create_domain_specific_client(client.api_token, "medical")
client.task_model_map.update(medical_config)

# Now client will prioritize medical models for each task
response = await client.process_task("knowledge_extraction", "Summarize the treatment options for hypertension.")
```

## Examples

For detailed examples of using specialized models, see:

- `main/examples/specialized_model_usage.py` - Demonstrates using models across different domains
- `docs/specialized.md` - Reference list of specialized models with their capabilities

## Adding New Models

To add new specialized models:

1. Update the `docs/specialized.md` file with model information
2. Add model registration to the appropriate function in `specialized_models.py`
3. Add the model to the task_model_map updates in `update_task_model_map_with_specialized`
4. Add domain-specific configuration in `create_domain_specific_client` 