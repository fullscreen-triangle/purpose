"""
Specialized Model Usage Example

This script demonstrates how to use the specialized domain-specific models
registered in the Purpose ModelHub for different domains and tasks.
"""

import asyncio
import os
from typing import Dict, List, Optional

from main.utils.model_hub import ModelHub, PurposeAPIClient, TaskType
from main.base_models.specialized_models import create_domain_specific_client

# Example queries for different domains
EXAMPLE_QUERIES = {
    "medical": [
        "Summarize the key treatment options for type 2 diabetes.",
        "Explain the mechanism of action for ACE inhibitors in hypertension treatment.",
        "Extract the main side effects mentioned in this clinical trial: 'Patients receiving the treatment reported nausea (12%), headache (8%), and fatigue (15%). No serious adverse events were observed.'"
    ],
    "legal": [
        "Summarize the key provisions in this contract section: 'The Parties agree that all disputes arising under this Agreement shall be resolved through binding arbitration in accordance with the rules of the American Arbitration Association.'",
        "Identify legal entities and their relationships in this text: 'Smith & Associates, LLP filed a motion on behalf of Acme Corporation against Blue Sky Enterprises in the Southern District Court of New York.'",
        "Explain the implications of the Fair Use doctrine in copyright law for educational institutions."
    ],
    "finance": [
        "Analyze the sentiment in this financial news: 'Company XYZ reported quarterly earnings that exceeded analyst expectations by 15%, leading to a 3% increase in share price during after-hours trading.'",
        "Summarize the key factors affecting interest rate decisions by central banks.",
        "Extract financial entities from this text: 'BlackRock increased its holdings in Apple Inc. by 5.3% to $13.2 billion, while reducing its stake in Microsoft to 2.1% of the portfolio.'"
    ],
    "code": [
        "Write a Python function to find the longest common subsequence of two strings.",
        "Explain how promises work in JavaScript and provide an example of chaining promises.",
        "Refactor this code to use modern Python features: 'def filter_data(data): result = []; for item in data: if item > 0: result.append(item); return result'"
    ],
    "math": [
        "Solve the differential equation dy/dx = 2xy with the initial condition y(0) = 1.",
        "Prove that the sum of the first n odd numbers equals n².",
        "Explain the concept of eigenvalues and eigenvectors with an example."
    ]
}

async def demonstrate_specialized_model(api_token: str, domain: str):
    """
    Demonstrate using specialized models for a specific domain.
    
    Args:
        api_token: The API token for authentication
        domain: The domain to demonstrate (medical, legal, finance, code, math)
    """
    print(f"\n===== Demonstrating {domain.upper()} domain models =====\n")
    
    # Create a domain-specific client
    client = PurposeAPIClient(api_token=api_token)
    domain_config = create_domain_specific_client(api_token, domain)
    client.task_model_map.update(domain_config)
    
    # Get example queries for the domain
    queries = EXAMPLE_QUERIES.get(domain, ["Generic query"])
    
    # Determine task types to use based on domain
    task_types = list(domain_config.keys())
    
    try:
        for i, query in enumerate(queries):
            if i >= len(task_types):
                # Use a default task if we run out of task types
                task_type = "response_generation" 
            else:
                task_type = task_types[i]
                
            print(f"Query {i+1}: {query}")
            print(f"Using task type: {task_type}")
            print(f"Selected models: {domain_config.get(task_type, ['No specialized models'])}")
            
            # We're just showing the setup, not actually making API calls
            # In a real scenario, you would use:
            # response = await client.process_task(task_type, query)
            # print(f"Response: {response}\n")
            
            print(f"[Response would be generated using {domain} specialized models]\n")
    finally:
        await client.close()


async def list_all_specialized_models():
    """List all specialized models registered in the ModelHub."""
    print("\n===== All Specialized Models in ModelHub =====\n")
    
    # Create a ModelHub with specialized models
    model_hub = ModelHub(load_specialized=True)
    
    # Group models by domain/category
    domains = {
        "Medical": ["meditron", "MedLLM", "BioMed", "Clinical", "Bio"],
        "Legal": ["legal", "Law", "Case"],
        "Finance": ["fin", "Finance", "financial"],
        "Math": ["Math", "math"],
        "Code": ["Code", "code", "Coder", "coder"],
        "Embedding": ["embed", "bge", "reranker"]
    }
    
    for domain, keywords in domains.items():
        print(f"\n----- {domain} Models -----")
        domain_models = []
        
        for model_id, model_info in model_hub.models.items():
            # Check if any keyword matches the model_id
            if any(keyword in model_id for keyword in keywords):
                specialties = ", ".join([s.value if isinstance(s, TaskType) else s 
                                        for s in model_info.specialties])
                domain_models.append(f"  - {model_id} ({model_info.size_params}, {specialties})")
        
        if domain_models:
            for model in sorted(domain_models):
                print(model)
        else:
            print("  No models found")


async def main():
    # Get API token from environment or use a placeholder
    api_token = os.environ.get("HUGGINGFACE_API_KEY", "YOUR_HF_TOKEN")
    
    # List all specialized models
    await list_all_specialized_models()
    
    # Demonstrate using specialized models for each domain
    domains = ["medical", "legal", "finance", "code", "math"]
    for domain in domains:
        await demonstrate_specialized_model(api_token, domain)


if __name__ == "__main__":
    asyncio.run(main()) 