"""
Specialized Models Registration Module

This module provides functions to register specialized domain-specific models
for medical, legal, financial, and technical domains in the Purpose ModelHub.
"""

from typing import Dict, List, Optional
from main.utils.model_hub import ModelHub, ModelInfo, ModelSource, TaskType

def register_medical_models(model_hub: ModelHub) -> None:
    """Register specialized medical domain models to the ModelHub."""
    # Meditron Models (Medical)
    model_hub.register_model(
        ModelInfo(
            model_id="epfl-llm/meditron-70b",
            source=ModelSource.HUGGINGFACE,
            strengths=["SOTA clinical reasoning", "Medical knowledge", "High diagnostic accuracy"],
            size_params="70B",
            context_window=8192,
            specialties=[TaskType.KNOWLEDGE_EXTRACTION, TaskType.RESPONSE_GENERATION],
        )
    )
    
    model_hub.register_model(
        ModelInfo(
            model_id="epfl-llm/meditron-7b",
            source=ModelSource.HUGGINGFACE,
            strengths=["Efficient medical reasoning", "Clinical knowledge", "Resource-friendly"],
            size_params="7B",
            context_window=4096,
            specialties=[TaskType.KNOWLEDGE_EXTRACTION, TaskType.DISTILLATION_TARGET],
        )
    )
    
    # DISC-MedLLM
    model_hub.register_model(
        ModelInfo(
            model_id="Flmc/DISC-MedLLM",
            source=ModelSource.HUGGINGFACE,
            strengths=["Patient-doctor dialogues", "Medical advice generation", "Clinical conversation"],
            size_params="13B",
            context_window=4096,
            specialties=[TaskType.RESPONSE_GENERATION, TaskType.INFERENCE],
        )
    )
    
    # BioMedLM
    model_hub.register_model(
        ModelInfo(
            model_id="stanford-crfm/BioMedLM-2.7B",
            source=ModelSource.HUGGINGFACE,
            strengths=["Lightweight medical model", "PubMed knowledge", "Efficient deployment"],
            size_params="2.7B",
            context_window=2048,
            specialties=[TaskType.DISTILLATION_TARGET, TaskType.INFERENCE],
        )
    )
    
    # Clinical ModernBERT
    model_hub.register_model(
        ModelInfo(
            model_id="Simonlee711/Clinical ModernBERT",
            source=ModelSource.HUGGINGFACE,
            strengths=["Medical entity recognition", "Clinical embeddings", "Fast inference"],
            size_params="110M",
            context_window=512,
            specialties=[TaskType.KNOWLEDGE_MAPPING, TaskType.TEXT_EMBEDDING],
        )
    )
    
    # BioGPT
    model_hub.register_model(
        ModelInfo(
            model_id="microsoft/BioGPT-Large",
            source=ModelSource.HUGGINGFACE,
            strengths=["Biomedical text generation", "Medical QA creation", "PubMed knowledge"],
            size_params="1.5B",
            context_window=1024,
            specialties=[TaskType.QUERY_GENERATION, TaskType.DATA_PROCESSING],
        )
    )
    
    # Medical Embeddings
    model_hub.register_model(
        ModelInfo(
            model_id="NeuML/pubmedbert-base-embeddings-matryoshka",
            source=ModelSource.HUGGINGFACE,
            strengths=["Dynamic dimension embeddings", "Biomedical corpus", "Efficient storage"],
            size_params="110M",
            context_window=512,
            specialties=[TaskType.TEXT_EMBEDDING, TaskType.KNOWLEDGE_MAPPING],
        )
    )


def register_math_models(model_hub: ModelHub) -> None:
    """Register specialized mathematics models to the ModelHub."""
    # MathCoder Models
    model_hub.register_model(
        ModelInfo(
            model_id="MathLLMs/MathCoder-L-13B",
            source=ModelSource.HUGGINGFACE,
            strengths=["Code-augmented math solving", "Step-by-step derivations", "Explanatory capability"],
            size_params="13B",
            context_window=4096,
            specialties=[TaskType.RESPONSE_GENERATION, TaskType.REASONING],
        )
    )
    
    model_hub.register_model(
        ModelInfo(
            model_id="MathLLMs/MathCoder-L-7B",
            source=ModelSource.HUGGINGFACE,
            strengths=["Efficient math reasoning", "Code integration", "Resource-friendly"],
            size_params="7B",
            context_window=4096,
            specialties=[TaskType.RESPONSE_GENERATION, TaskType.REASONING],
        )
    )
    
    model_hub.register_model(
        ModelInfo(
            model_id="MathLLMs/MathCoder-CL-34B",
            source=ModelSource.HUGGINGFACE,
            strengths=["Large context window", "Complex theorem handling", "Advanced mathematical reasoning"],
            size_params="34B",
            context_window=16384,
            specialties=[TaskType.KNOWLEDGE_EXTRACTION, TaskType.REASONING],
        )
    )


def register_legal_models(model_hub: ModelHub) -> None:
    """Register specialized legal domain models to the ModelHub."""
    # Legal BERT Models
    model_hub.register_model(
        ModelInfo(
            model_id="lexlms/legal-roberta-base",
            source=ModelSource.HUGGINGFACE,
            strengths=["Legal document classification", "Legal terminology understanding", "Case categorization"],
            size_params="125M",
            context_window=512,
            specialties=[TaskType.KNOWLEDGE_MAPPING, TaskType.TEXT_CLASSIFICATION],
        )
    )
    
    model_hub.register_model(
        ModelInfo(
            model_id="lexlms/legal-longformer-base",
            source=ModelSource.HUGGINGFACE,
            strengths=["Long legal document processing", "Contract analysis", "Extended context"],
            size_params="149M",
            context_window=4096,
            specialties=[TaskType.KNOWLEDGE_MAPPING, TaskType.TEXT_CLASSIFICATION],
        )
    )
    
    model_hub.register_model(
        ModelInfo(
            model_id="nile/legal-bert-base",
            source=ModelSource.HUGGINGFACE,
            strengths=["US case law specialization", "Legal entity recognition", "Citation linking"],
            size_params="110M",
            context_window=512,
            specialties=[TaskType.KNOWLEDGE_EXTRACTION, TaskType.TEXT_CLASSIFICATION],
        )
    )
    
    model_hub.register_model(
        ModelInfo(
            model_id="CaseLawBERT/CaseLawBERT",
            source=ModelSource.HUGGINGFACE,
            strengths=["Legal precedent identification", "Case law understanding", "Legal reasoning"],
            size_params="340M",
            context_window=512,
            specialties=[TaskType.KNOWLEDGE_EXTRACTION, TaskType.TEXT_CLASSIFICATION],
        )
    )
    
    model_hub.register_model(
        ModelInfo(
            model_id="IBM/Legal-Universe-Llama-2-7b",
            source=ModelSource.HUGGINGFACE,
            strengths=["Legal reasoning", "Compliance analysis", "Regulatory knowledge"],
            size_params="7B",
            context_window=4096,
            specialties=[TaskType.RESPONSE_GENERATION, TaskType.REASONING],
        )
    )


def register_finance_models(model_hub: ModelHub) -> None:
    """Register specialized finance domain models to the ModelHub."""
    # Finance BERT Models
    model_hub.register_model(
        ModelInfo(
            model_id="yiyanghkust/finbert-tone",
            source=ModelSource.HUGGINGFACE,
            strengths=["Financial sentiment analysis", "Market tone detection", "Earnings report analysis"],
            size_params="110M",
            context_window=512,
            specialties=[TaskType.KNOWLEDGE_MAPPING, TaskType.TEXT_CLASSIFICATION],
        )
    )
    
    model_hub.register_model(
        ModelInfo(
            model_id="ProsusAI/finbert",
            source=ModelSource.HUGGINGFACE,
            strengths=["Financial entity recognition", "Financial instrument classification", "Financial NLP"],
            size_params="110M",
            context_window=512,
            specialties=[TaskType.KNOWLEDGE_MAPPING, TaskType.TEXT_CLASSIFICATION],
        )
    )
    
    # FinGPT Models
    model_hub.register_model(
        ModelInfo(
            model_id="FinGPT/fingpt-mt_llama2-7b",
            source=ModelSource.HUGGINGFACE,
            strengths=["Multi-task financial reasoning", "Market analysis", "Financial forecasting"],
            size_params="7B",
            context_window=4096,
            specialties=[TaskType.RESPONSE_GENERATION, TaskType.REASONING],
        )
    )
    
    # Microsoft Finance Models
    model_hub.register_model(
        ModelInfo(
            model_id="microsoft/phi-2-finance",
            source=ModelSource.HUGGINGFACE,
            strengths=["Compact financial model", "Fiscal knowledge", "Efficient deployment"],
            size_params="2.7B",
            context_window=2048,
            specialties=[TaskType.DISTILLATION_TARGET, TaskType.INFERENCE],
        )
    )
    
    # NVIDIA Finance Models
    model_hub.register_model(
        ModelInfo(
            model_id="NVIDIA/NeMo-Megatron-Fin",
            source=ModelSource.CUSTOM,
            strengths=["Large-scale financial analysis", "Regulatory compliance", "Advanced financial reasoning"],
            size_params="20B",
            context_window=8192,
            specialties=[TaskType.KNOWLEDGE_EXTRACTION, TaskType.REASONING],
            url="https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/megatron_fin"
        )
    )


def register_code_models(model_hub: ModelHub) -> None:
    """Register specialized code and technical models to the ModelHub."""
    # Facebook Code Models
    model_hub.register_model(
        ModelInfo(
            model_id="facebook/incoder-6B",
            source=ModelSource.HUGGINGFACE,
            strengths=["Code infilling", "Code completion", "Developer assistance"],
            size_params="6B",
            context_window=2048,
            specialties=[TaskType.RESPONSE_GENERATION, TaskType.CODE_GENERATION],
        )
    )
    
    # Wizard Code Models
    model_hub.register_model(
        ModelInfo(
            model_id="WizardLM/WizardCoder-Python-34B",
            source=ModelSource.HUGGINGFACE,
            strengths=["Expert Python code generation", "Code explanation", "Advanced algorithms"],
            size_params="34B",
            context_window=8192,
            specialties=[TaskType.RESPONSE_GENERATION, TaskType.CODE_GENERATION],
        )
    )
    
    # CodeLlama Models
    model_hub.register_model(
        ModelInfo(
            model_id="codellama/CodeLlama-7b-hf",
            source=ModelSource.HUGGINGFACE,
            strengths=["Multi-language code generation", "Base for fine-tuning", "Versatile coding"],
            size_params="7B",
            context_window=8192,
            specialties=[TaskType.DISTILLATION_TARGET, TaskType.CODE_GENERATION],
        )
    )
    
    # StarCoder Models
    model_hub.register_model(
        ModelInfo(
            model_id="bigcode/starcoder2-15b",
            source=ModelSource.HUGGINGFACE,
            strengths=["Permissive license", "Enterprise integration", "Diverse programming languages"],
            size_params="15B",
            context_window=8192,
            specialties=[TaskType.RESPONSE_GENERATION, TaskType.CODE_GENERATION],
        )
    )


def register_embedding_models(model_hub: ModelHub) -> None:
    """Register specialized embedding models to the ModelHub."""
    # BGE Embedding Models
    model_hub.register_model(
        ModelInfo(
            model_id="BAAI/bge-large-en-v1.5",
            source=ModelSource.HUGGINGFACE,
            strengths=["SOTA retrieval performance", "Dense vector embeddings", "High quality retrieval"],
            size_params="335M",
            context_window=8192,
            specialties=[TaskType.TEXT_EMBEDDING, TaskType.KNOWLEDGE_MAPPING],
        )
    )
    
    model_hub.register_model(
        ModelInfo(
            model_id="BAAI/bge-m3",
            source=ModelSource.HUGGINGFACE,
            strengths=["Multi-function retrieval", "Multi-lingual support", "Dense & sparse capabilities"],
            size_params="568M",
            context_window=8192,
            specialties=[TaskType.TEXT_EMBEDDING, TaskType.KNOWLEDGE_MAPPING, TaskType.MULTILINGUAL],
        )
    )
    
    model_hub.register_model(
        ModelInfo(
            model_id="BAAI/bge-reranker-v2-m3",
            source=ModelSource.HUGGINGFACE,
            strengths=["Cross-encoder reranking", "Retrieval pipeline enhancement", "Precision improvement"],
            size_params="440M",
            context_window=4096,
            specialties=[TaskType.TEXT_CLASSIFICATION, TaskType.KNOWLEDGE_MAPPING],
        )
    )


def register_all_specialized_models(model_hub: ModelHub) -> None:
    """Register all specialized domain models to the ModelHub."""
    register_medical_models(model_hub)
    register_math_models(model_hub)
    register_legal_models(model_hub)
    register_finance_models(model_hub)
    register_code_models(model_hub)
    register_embedding_models(model_hub)


def update_task_model_map_with_specialized(model_hub: ModelHub) -> None:
    """
    Update the model hub's task model map with specialized domain models
    for each task type.
    """
    # Extend task model map with specialized models
    specialized_task_map = {
        # Medical domain tasks
        TaskType.KNOWLEDGE_EXTRACTION: [
            "epfl-llm/meditron-7b",
            "epfl-llm/meditron-70b",
            "CaseLawBERT/CaseLawBERT",  # Legal domain
            "NVIDIA/NeMo-Megatron-Fin"  # Finance domain
        ],
        TaskType.KNOWLEDGE_MAPPING: [
            "Simonlee711/Clinical ModernBERT",
            "BAAI/bge-m3",
            "lexlms/legal-longformer-base",  # Legal domain
            "ProsusAI/finbert"  # Finance domain
        ],
        TaskType.QUERY_GENERATION: [
            "microsoft/BioGPT-Large"
        ],
        TaskType.RESPONSE_GENERATION: [
            "Flmc/DISC-MedLLM",
            "MathLLMs/MathCoder-L-13B",  # Math domain
            "IBM/Legal-Universe-Llama-2-7b",  # Legal domain
            "FinGPT/fingpt-mt_llama2-7b",  # Finance domain
            "WizardLM/WizardCoder-Python-34B"  # Code domain
        ],
        TaskType.DISTILLATION_TARGET: [
            "stanford-crfm/BioMedLM-2.7B",
            "microsoft/phi-2-finance",  # Finance domain
            "codellama/CodeLlama-7b-hf"  # Code domain
        ],
        TaskType.TEXT_EMBEDDING: [
            "NeuML/pubmedbert-base-embeddings-matryoshka",
            "BAAI/bge-large-en-v1.5"
        ],
        TaskType.REASONING: [
            "MathLLMs/MathCoder-CL-34B"
        ],
        TaskType.CODE_GENERATION: [
            "facebook/incoder-6B",
            "bigcode/starcoder2-15b"
        ]
    }
    
    # Update the model hub's task model map
    for task_type, models in specialized_task_map.items():
        # Get current models for this task
        current_models = model_hub.task_model_map.get(task_type, [])
        
        # Add specialized models at the beginning of the list (higher priority)
        for model in reversed(models):
            if model not in current_models:
                current_models.insert(0, model)
        
        # Update the task model map
        model_hub.task_model_map[task_type] = current_models


def create_domain_specific_client(api_token: str, domain: str) -> dict:
    """
    Create a domain-specific configuration for the PurposeAPIClient.
    
    Args:
        api_token: The API token for authentication
        domain: The specific domain to focus on (e.g., 'medical', 'legal', 'finance', 'code', 'math')
    
    Returns:
        A dictionary containing task-to-model mappings specific to the domain
    """
    domain_configs = {
        "medical": {
            "knowledge_extraction": ["epfl-llm/meditron-7b", "epfl-llm/meditron-70b"],
            "knowledge_mapping": ["Simonlee711/Clinical ModernBERT", "BAAI/bge-large-en-v1.5"],
            "query_generation": ["microsoft/BioGPT-Large"],
            "response_generation": ["Flmc/DISC-MedLLM", "epfl-llm/meditron-70b"],
            "distillation_target": ["stanford-crfm/BioMedLM-2.7B"],
            "text_embedding": ["NeuML/pubmedbert-base-embeddings-matryoshka"]
        },
        "legal": {
            "knowledge_extraction": ["nile/legal-bert-base", "CaseLawBERT/CaseLawBERT"],
            "knowledge_mapping": ["lexlms/legal-roberta-base", "lexlms/legal-longformer-base"],
            "response_generation": ["IBM/Legal-Universe-Llama-2-7b"],
            "text_classification": ["lexlms/legal-roberta-base"]
        },
        "finance": {
            "knowledge_extraction": ["NVIDIA/NeMo-Megatron-Fin"],
            "knowledge_mapping": ["yiyanghkust/finbert-tone", "ProsusAI/finbert"],
            "response_generation": ["FinGPT/fingpt-mt_llama2-7b"],
            "distillation_target": ["microsoft/phi-2-finance"],
            "text_classification": ["yiyanghkust/finbert-tone"]
        },
        "code": {
            "code_generation": ["facebook/incoder-6B", "WizardLM/WizardCoder-Python-34B"],
            "response_generation": ["bigcode/starcoder2-15b"],
            "distillation_target": ["codellama/CodeLlama-7b-hf"]
        },
        "math": {
            "response_generation": ["MathLLMs/MathCoder-L-13B", "MathLLMs/MathCoder-L-7B"],
            "knowledge_extraction": ["MathLLMs/MathCoder-CL-34B"],
            "reasoning": ["MathLLMs/MathCoder-CL-34B"]
        }
    }
    
    if domain not in domain_configs:
        raise ValueError(f"Unsupported domain: {domain}. Choose from: {list(domain_configs.keys())}")
    
    return domain_configs[domain]


# Example usage
if __name__ == "__main__":
    from main.utils.model_hub import PurposeAPIClient
    
    # Create a client with medical domain specialization
    client = PurposeAPIClient(api_token="YOUR_HF_TOKEN")
    medical_config = create_domain_specific_client(client.api_token, "medical")
    client.task_model_map.update(medical_config)
    
    # Print the updated configuration
    for task, models in client.task_model_map.items():
        print(f"{task}: {models}") 