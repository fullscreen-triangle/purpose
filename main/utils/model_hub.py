"""
ModelHub: A central interface for accessing various specialized language models.

This module provides access to specialized models for different tasks in the 
Purpose pipeline, supporting both local models via Ollama and remote models 
via various APIs (Hugging Face, OpenAI, Anthropic, etc.).
"""

import os
import json
import asyncio
import aiohttp
import logging
from typing import Dict, Any, Optional, Union, List
from enum import Enum
from pathlib import Path

# Setup logging
logger = logging.getLogger("main.model_hub")

class ModelSource(Enum):
    """Enum for different model sources"""
    HUGGINGFACE = "huggingface"
    LOCAL_HUGGINGFACE = "local_huggingface"  # New source for local HF models
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    REPLICATE = "replicate"
    TOGETHER_AI = "together_ai"
    GOOGLE = "google"
    CUSTOM = "custom"

class TaskType(Enum):
    """Enum for different task types in the pipeline"""
    BASE_TRAINING = "base_training"
    DISTILLATION_TARGET = "distillation_target"
    DATA_PROCESSING = "data_processing"
    KNOWLEDGE_MAPPING = "knowledge_mapping"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    QUERY_GENERATION = "query_generation"
    RESPONSE_GENERATION = "response_generation"
    CURRICULUM_LEARNING = "curriculum_learning"
    INFERENCE = "inference"
    TEXT_EMBEDDING = "text_embedding"
    TEXT_CLASSIFICATION = "text_classification"
    REASONING = "reasoning"
    INSTRUCTION_FOLLOWING = "instruction_following"
    CODE_GENERATION = "code_generation"
    MULTILINGUAL = "multilingual"

class ModelInfo:
    """Class to store model information"""
    
    def __init__(
        self, 
        model_id: str, 
        source: ModelSource,
        strengths: List[str],
        size_params: str,
        context_window: int,
        specialties: List[TaskType] = None,
        endpoint: Optional[str] = None,
        url: Optional[str] = None,
        local_path: Optional[str] = None,
    ):
        self.model_id = model_id
        self.source = source
        self.strengths = strengths
        self.size_params = size_params
        self.context_window = context_window
        self.specialties = specialties or []
        self.endpoint = endpoint
        self.url = url
        self.local_path = local_path

class ModelHub:
    """
    Hub for accessing specialized models for different tasks.
    
    This class provides a unified interface for accessing models from various sources,
    including Hugging Face, Ollama, OpenAI, Anthropic, etc.
    """
    
    def __init__(self, config_path: Optional[str] = None, load_specialized: bool = True, local_models_dir: Optional[str] = None):
        """
        Initialize the ModelHub.
        
        Args:
            config_path: Path to a JSON config file with API keys and model configurations
            load_specialized: Whether to load specialized domain-specific models
            local_models_dir: Directory containing locally downloaded models
        """
        # Load API keys from config or environment
        self.api_keys = {}
        self.load_api_keys(config_path)
        
        # Set local models directory
        self.local_models_dir = local_models_dir or os.environ.get("LOCAL_MODELS_DIR") or "./models"
        
        # Initialize model registry
        self.models: Dict[str, ModelInfo] = {}
        self.register_default_models()
        
        # Map tasks to recommended models
        self.task_model_map = {
            # Training & Distillation Models
            TaskType.BASE_TRAINING: ["meta-llama/llama-3-8b", "mistralai/mistral-7b-v0.1", "microsoft/phi-3-medium-4k-instruct"],
            TaskType.DISTILLATION_TARGET: ["microsoft/phi-3-mini-4k-instruct", "google/gemma-2-2b-it", "TinyLlama/TinyLlama-1.1B"],
            
            # Data Processing Models
            TaskType.DATA_PROCESSING: ["mistralai/mistral-7b-instruct-v0.2", "databricks/dolly-v2-3b", "google/flan-t5-xl"],
            TaskType.KNOWLEDGE_MAPPING: ["Qwen/Qwen1.5-7B", "allenai/OLMo-7B", "bigscience/bloomz-7b1"],
            TaskType.KNOWLEDGE_EXTRACTION: ["meta-llama/llama-3-8b", "allenai/tulu-2-7b", "HuggingFaceH4/zephyr-7b-beta"],
            
            # Query & Response Generation
            TaskType.QUERY_GENERATION: ["mistralai/Mixtral-8x7B-Instruct-v0.1", "bigscience/T0_3B", "databricks/dolly-v2-12b"],
            TaskType.RESPONSE_GENERATION: ["01-ai/Yi-34B", "meta-llama/llama-3-70b", "mistralai/Mixtral-8x22B-v0.1"],
            
            # Specialized Tasks
            TaskType.CURRICULUM_LEARNING: ["meta-llama/llama-3-8b", "togethercomputer/RedPajama-INCITE-7B-Instruct", "mistralai/mistral-7b-instruct-v0.2"],
            TaskType.INFERENCE: ["microsoft/phi-3-small-4k-instruct", "google/gemma-2-9b-it", "mistralai/mistral-7b-instruct-v0.2"],
            TaskType.TEXT_EMBEDDING: ["intfloat/e5-large-v2", "BAAI/bge-large-en-v1.5", "sentence-transformers/all-mpnet-base-v2"],
            TaskType.TEXT_CLASSIFICATION: ["facebook/bart-large-mnli", "cross-encoder/nli-roberta-base", "distilbert-base-uncased-finetuned-sst-2-english"],
            TaskType.REASONING: ["allenai/tulu-2-dpo-70b", "google/gemma-2-27b-it", "meta-llama/llama-3-70b"],
            TaskType.INSTRUCTION_FOLLOWING: ["microsoft/phi-3-medium-4k-instruct", "mistralai/mistral-7b-instruct-v0.2", "databricks/dolly-v2-7b"],
            TaskType.CODE_GENERATION: ["Salesforce/codegen25-7b-instruct", "bigcode/starcoder2-15b", "replit/replit-code-v1.5-3b"],
            TaskType.MULTILINGUAL: ["google/mT5-base", "facebook/mbart-large-50", "facebook/xglm-7.5B"],
        }
        
        # Load specialized domain-specific models if requested
        if load_specialized:
            try:
                # Import needs to be here to avoid circular imports
                from main.base_models.specialized_models import register_all_specialized_models, update_task_model_map_with_specialized
                
                # Register specialized models and update task model map
                register_all_specialized_models(self)
                update_task_model_map_with_specialized(self)
                logger.info("Loaded specialized domain-specific models successfully")
            except ImportError as e:
                logger.warning(f"Failed to load specialized models: {e}")
        
        # Register local models from the models directory
        self.register_local_models()
        
        # Initialize session
        self.session = None
    
    def load_api_keys(self, config_path: Optional[str] = None) -> None:
        """
        Load API keys from config file or environment variables.
        
        Args:
            config_path: Path to config file
        """
        # Try to load from config file
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.api_keys = config.get('api_keys', {})
        
        # Override/supplement with environment variables
        for source in ModelSource:
            env_var = f"{source.value.upper()}_API_KEY"
            if env_var in os.environ:
                self.api_keys[source.value] = os.environ[env_var]
    
    def register_default_models(self) -> None:
        """Register the default set of models in the hub."""
        # LLaMA Models
        self.register_model(
            ModelInfo(
                model_id="meta-llama/llama-3-8b",
                source=ModelSource.HUGGINGFACE,
                strengths=["Strong general capability", "Efficient", "Good instruction following"],
                size_params="8B",
                context_window=8192,
                specialties=[TaskType.BASE_TRAINING, TaskType.KNOWLEDGE_EXTRACTION, TaskType.CURRICULUM_LEARNING],
            )
        )
        
        self.register_model(
            ModelInfo(
                model_id="meta-llama/llama-3-70b",
                source=ModelSource.HUGGINGFACE,
                strengths=["Advanced reasoning", "Strong knowledge", "High accuracy"],
                size_params="70B",
                context_window=8192,
                specialties=[TaskType.RESPONSE_GENERATION, TaskType.REASONING],
            )
        )
        
        # Mistral Models
        self.register_model(
            ModelInfo(
                model_id="mistralai/mistral-7b-instruct-v0.2",
                source=ModelSource.HUGGINGFACE,
                strengths=["Efficient instruction following", "Strong reasoning", "Good knowledge"],
                size_params="7B",
                context_window=8192,
                specialties=[TaskType.DATA_PROCESSING, TaskType.INFERENCE, TaskType.INSTRUCTION_FOLLOWING],
            )
        )
        
        self.register_model(
            ModelInfo(
                model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                source=ModelSource.HUGGINGFACE,
                strengths=["Mixture of experts", "Strong multi-domain performance", "Good for complex queries"],
                size_params="8x7B (MoE)",
                context_window=32768,
                specialties=[TaskType.QUERY_GENERATION, TaskType.REASONING],
            )
        )
        
        # Microsoft Phi Models
        self.register_model(
            ModelInfo(
                model_id="microsoft/phi-3-mini-4k-instruct",
                source=ModelSource.HUGGINGFACE,
                strengths=["Highly efficient", "Strong performance for size", "Good instruction following"],
                size_params="3.8B",
                context_window=4096,
                specialties=[TaskType.DISTILLATION_TARGET],
            )
        )
        
        self.register_model(
            ModelInfo(
                model_id="microsoft/phi-3-small-4k-instruct",
                source=ModelSource.HUGGINGFACE,
                strengths=["Efficient", "Strong reasoning for size", "Good for lightweight deployment"],
                size_params="7B",
                context_window=4096,
                specialties=[TaskType.INFERENCE],
            )
        )
        
        self.register_model(
            ModelInfo(
                model_id="microsoft/phi-3-medium-4k-instruct",
                source=ModelSource.HUGGINGFACE,
                strengths=["Good balance of performance and efficiency", "Strong instruction following"],
                size_params="14B",
                context_window=4096,
                specialties=[TaskType.BASE_TRAINING, TaskType.INSTRUCTION_FOLLOWING],
            )
        )
        
        # Google Models
        self.register_model(
            ModelInfo(
                model_id="google/gemma-2-2b-it",
                source=ModelSource.HUGGINGFACE,
                strengths=["Highly efficient", "Strong performance for size", "Good for edge deployment"],
                size_params="2B",
                context_window=8192,
                specialties=[TaskType.DISTILLATION_TARGET],
            )
        )
        
        self.register_model(
            ModelInfo(
                model_id="google/gemma-2-9b-it",
                source=ModelSource.HUGGINGFACE,
                strengths=["Efficient", "Strong instruction following", "Good for on-device applications"],
                size_params="9B",
                context_window=8192,
                specialties=[TaskType.INFERENCE],
            )
        )
        
        self.register_model(
            ModelInfo(
                model_id="google/gemma-2-27b-it",
                source=ModelSource.HUGGINGFACE,
                strengths=["Advanced reasoning", "Strong knowledge", "Good for complex tasks"],
                size_params="27B",
                context_window=8192,
                specialties=[TaskType.REASONING],
            )
        )
        
        # Yi Models
        self.register_model(
            ModelInfo(
                model_id="01-ai/Yi-34B",
                source=ModelSource.HUGGINGFACE,
                strengths=["Strong knowledge", "Good multilingual support", "Advanced reasoning"],
                size_params="34B",
                context_window=4096,
                specialties=[TaskType.RESPONSE_GENERATION],
            )
        )
        
        # Qwen Models
        self.register_model(
            ModelInfo(
                model_id="Qwen/Qwen1.5-7B",
                source=ModelSource.HUGGINGFACE,
                strengths=["Strong knowledge organization", "Good for structured data", "Efficient"],
                size_params="7B",
                context_window=8192,
                specialties=[TaskType.KNOWLEDGE_MAPPING],
            )
        )
        
        # Code Models
        self.register_model(
            ModelInfo(
                model_id="Salesforce/codegen25-7b-instruct",
                source=ModelSource.HUGGINGFACE,
                strengths=["Strong code generation", "Good documentation", "Debugging capabilities"],
                size_params="7B",
                context_window=8192,
                specialties=[TaskType.CODE_GENERATION],
            )
        )
        
        self.register_model(
            ModelInfo(
                model_id="bigcode/starcoder2-15b",
                source=ModelSource.HUGGINGFACE,
                strengths=["Advanced code generation", "Multiple languages", "Reasoning about code"],
                size_params="15B",
                context_window=16384,
                specialties=[TaskType.CODE_GENERATION],
            )
        )
        
        # Embedding Models
        self.register_model(
            ModelInfo(
                model_id="intfloat/e5-large-v2",
                source=ModelSource.HUGGINGFACE,
                strengths=["Strong text embeddings", "Good for semantic search", "Efficient"],
                size_params="335M",
                context_window=512,
                specialties=[TaskType.TEXT_EMBEDDING],
            )
        )
        
        self.register_model(
            ModelInfo(
                model_id="BAAI/bge-large-en-v1.5",
                source=ModelSource.HUGGINGFACE,
                strengths=["State-of-the-art embeddings", "Good for retrieval", "Strong semantic understanding"],
                size_params="335M",
                context_window=512,
                specialties=[TaskType.TEXT_EMBEDDING],
            )
        )
        
        # Add more models...
    
    def register_model(self, model_info: ModelInfo) -> None:
        """
        Register a model in the hub.
        
        Args:
            model_info: ModelInfo object with model details
        """
        self.models[model_info.model_id] = model_info
        logger.debug(f"Registered model: {model_info.model_id}")
    
    def get_recommended_models(self, task_type: Union[TaskType, str]) -> List[str]:
        """
        Get recommended models for a specific task.
        
        Args:
            task_type: Task type enum or string
            
        Returns:
            List of recommended model IDs
        """
        if isinstance(task_type, str):
            task_type = TaskType(task_type)
        
        return self.task_model_map.get(task_type, [])
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get information about a specific model.
        
        Args:
            model_id: Model ID
            
        Returns:
            ModelInfo object or None if not found
        """
        return self.models.get(model_id)
    
    async def _ensure_session(self) -> None:
        """Ensure aiohttp session is initialized."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def _call_huggingface_api(self, model_id: str, input_text: str, **kwargs) -> Any:
        """
        Call Hugging Face Inference API.
        
        Args:
            model_id: Model ID
            input_text: Input text
            **kwargs: Additional parameters
            
        Returns:
            API response
        """
        await self._ensure_session()
        
        API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
        
        parameters = kwargs.get("parameters", {})
        if "temperature" not in parameters:
            parameters["temperature"] = 0.7
        if "max_new_tokens" not in parameters:
            parameters["max_new_tokens"] = 512
            
        payload = {
            "inputs": input_text,
            "parameters": parameters
        }
        
        headers = {"Authorization": f"Bearer {self.api_keys.get('huggingface')}"}
        
        async with self.session.post(API_URL, headers=headers, json=payload) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"HuggingFace API error ({response.status}): {error_text}")
    
    async def _call_ollama_api(self, model_id: str, input_text: str, **kwargs) -> Any:
        """
        Call Ollama API.
        
        Args:
            model_id: Model ID
            input_text: Input text
            **kwargs: Additional parameters
            
        Returns:
            API response
        """
        await self._ensure_session()
        
        API_URL = "http://localhost:11434/api/generate"
        
        parameters = kwargs.get("parameters", {})
        
        payload = {
            "model": model_id,
            "prompt": input_text,
            "stream": False,
            "temperature": parameters.get("temperature", 0.7),
            "num_predict": parameters.get("max_new_tokens", 512),
        }
        
        async with self.session.post(API_URL, json=payload) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Ollama API error ({response.status}): {error_text}")
    
    async def _call_openai_api(self, model_id: str, input_text: str, **kwargs) -> Any:
        """
        Call OpenAI API.
        
        Args:
            model_id: Model ID
            input_text: Input text
            **kwargs: Additional parameters
            
        Returns:
            API response
        """
        await self._ensure_session()
        
        API_URL = "https://api.openai.com/v1/chat/completions"
        
        parameters = kwargs.get("parameters", {})
        
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": input_text}],
            "temperature": parameters.get("temperature", 0.7),
            "max_tokens": parameters.get("max_new_tokens", 512),
        }
        
        headers = {"Authorization": f"Bearer {self.api_keys.get('openai')}"}
        
        async with self.session.post(API_URL, headers=headers, json=payload) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"OpenAI API error ({response.status}): {error_text}")
    
    async def _call_anthropic_api(self, model_id: str, input_text: str, **kwargs) -> Any:
        """
        Call Anthropic API.
        
        Args:
            model_id: Model ID
            input_text: Input text
            **kwargs: Additional parameters
            
        Returns:
            API response
        """
        await self._ensure_session()
        
        API_URL = "https://api.anthropic.com/v1/messages"
        
        parameters = kwargs.get("parameters", {})
        
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": input_text}],
            "temperature": parameters.get("temperature", 0.7),
            "max_tokens": parameters.get("max_new_tokens", 512),
        }
        
        headers = {
            "x-api-key": self.api_keys.get('anthropic'),
            "anthropic-version": "2023-06-01"
        }
        
        async with self.session.post(API_URL, headers=headers, json=payload) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Anthropic API error ({response.status}): {error_text}")
    
    async def _call_replicate_api(self, model_id: str, input_text: str, **kwargs) -> Any:
        """
        Call Replicate API.
        
        Args:
            model_id: Model ID
            input_text: Input text
            **kwargs: Additional parameters
            
        Returns:
            API response
        """
        await self._ensure_session()
        
        API_URL = "https://api.replicate.com/v1/predictions"
        
        parameters = kwargs.get("parameters", {})
        
        payload = {
            "version": model_id,
            "input": {
                "prompt": input_text,
                "temperature": parameters.get("temperature", 0.7),
                "max_new_tokens": parameters.get("max_new_tokens", 512),
            }
        }
        
        headers = {"Authorization": f"Token {self.api_keys.get('replicate')}"}
        
        async with self.session.post(API_URL, headers=headers, json=payload) as response:
            if response.status == 201:
                prediction = await response.json()
                # Poll for completion
                get_url = prediction.get("urls", {}).get("get")
                if get_url:
                    while True:
                        async with self.session.get(get_url, headers=headers) as get_response:
                            status = await get_response.json()
                            if status["status"] == "succeeded":
                                return status
                            elif status["status"] == "failed":
                                raise Exception(f"Replicate prediction failed: {status.get('error')}")
                            await asyncio.sleep(1)
                return prediction
            else:
                error_text = await response.text()
                raise Exception(f"Replicate API error ({response.status}): {error_text}")
    
    async def _call_together_ai_api(self, model_id: str, input_text: str, **kwargs) -> Any:
        """
        Call Together AI API.
        
        Args:
            model_id: Model ID
            input_text: Input text
            **kwargs: Additional parameters
            
        Returns:
            API response
        """
        await self._ensure_session()
        
        API_URL = "https://api.together.xyz/v1/completions"
        
        parameters = kwargs.get("parameters", {})
        
        payload = {
            "model": model_id,
            "prompt": input_text,
            "temperature": parameters.get("temperature", 0.7),
            "max_tokens": parameters.get("max_new_tokens", 512),
        }
        
        headers = {"Authorization": f"Bearer {self.api_keys.get('together_ai')}"}
        
        async with self.session.post(API_URL, headers=headers, json=payload) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Together AI API error ({response.status}): {error_text}")
    
    async def process_task(
        self, 
        task_type: Union[TaskType, str], 
        input_text: str, 
        model_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Process a task using an appropriate model.
        
        Args:
            task_type: Task type enum or string
            input_text: Input text
            model_id: Specific model ID to use (optional)
            **kwargs: Additional parameters
            
        Returns:
            Model response
        """
        # Convert string task type to enum if needed
        if isinstance(task_type, str):
            try:
                task_type = TaskType(task_type)
            except ValueError:
                raise ValueError(f"Unknown task type: {task_type}")
        
        # Get model ID if not specified
        if model_id is None:
            recommended_models = self.get_recommended_models(task_type)
            if not recommended_models:
                raise ValueError(f"No recommended models for task type: {task_type.value}")
            model_id = recommended_models[0]
        
        # Get model info
        model_info = self.get_model_info(model_id)
        if not model_info:
            raise ValueError(f"Unknown model: {model_id}")
        
        # Call appropriate API based on model source
        if model_info.source == ModelSource.HUGGINGFACE:
            return await self._call_huggingface_api(model_id, input_text, **kwargs)
        elif model_info.source == ModelSource.OLLAMA:
            return await self._call_ollama_api(model_id, input_text, **kwargs)
        elif model_info.source == ModelSource.OPENAI:
            return await self._call_openai_api(model_id, input_text, **kwargs)
        elif model_info.source == ModelSource.ANTHROPIC:
            return await self._call_anthropic_api(model_id, input_text, **kwargs)
        elif model_info.source == ModelSource.REPLICATE:
            return await self._call_replicate_api(model_id, input_text, **kwargs)
        elif model_info.source == ModelSource.TOGETHER_AI:
            return await self._call_together_ai_api(model_id, input_text, **kwargs)
        else:
            raise ValueError(f"Unsupported model source: {model_info.source.value}")
    
    async def close(self) -> None:
        """Close aiohttp session."""
        if self.session is not None:
            await self.session.close()
            self.session = None

    def register_local_models(self) -> None:
        """Register locally downloaded models from the models directory."""
        if not os.path.exists(self.local_models_dir):
            logger.warning(f"Local models directory not found: {self.local_models_dir}")
            return
        
        try:
            # Scan for model directories
            for model_dir in Path(self.local_models_dir).iterdir():
                if model_dir.is_dir():
                    # Extract the original model ID from the directory name
                    # Directory names are created with '/' replaced by '_'
                    model_id = model_dir.name.replace('_', '/', 1)  # Replace only the first underscore
                    
                    # Check if this is a valid model directory by looking for config.json
                    if (model_dir / "config.json").exists():
                        # Register as a local model
                        local_model_info = ModelInfo(
                            model_id=model_id,
                            source=ModelSource.LOCAL_HUGGINGFACE,
                            strengths=["Local deployment", "Low latency", "No API required"],
                            local_path=str(model_dir),
                            # Other fields can be set based on config if needed
                        )
                        
                        self.register_model(local_model_info)
                        logger.info(f"Registered local model: {model_id} from {model_dir}")
        
        except Exception as e:
            logger.warning(f"Error registering local models: {e}")

    async def get_model(self, model_id: str, input_text: str, **kwargs) -> Any:
        """
        Get a response from a model.
        
        This method will first check if the model is available locally before
        attempting to use any remote APIs.
        
        Args:
            model_id: Model ID
            input_text: Input text
            **kwargs: Additional parameters
            
        Returns:
            Model response
        """
        model_info = self.get_model_info(model_id)
        
        if not model_info:
            logger.warning(f"Model not found: {model_id}")
            # Try to find if we have a local version of this model
            local_model_id = self._find_local_version(model_id)
            if local_model_id:
                logger.info(f"Using local version of {model_id}: {local_model_id}")
                model_info = self.get_model_info(local_model_id)
            else:
                # If no local version, try as a regular HF model
                logger.info(f"Using regular HF model: {model_id}")
                return await self._call_huggingface_api(model_id, input_text, **kwargs)
        
        # Use the appropriate method based on model source
        if model_info.source == ModelSource.LOCAL_HUGGINGFACE:
            return self._call_local_model(model_info.local_path, input_text, **kwargs)
        elif model_info.source == ModelSource.HUGGINGFACE:
            return await self._call_huggingface_api(model_id, input_text, **kwargs)
        elif model_info.source == ModelSource.OLLAMA:
            return await self._call_ollama_api(model_id, input_text, **kwargs)
        # ... other model sources ...
        
        raise NotImplementedError(f"Model source not implemented: {model_info.source}")
    
    def _find_local_version(self, model_id: str) -> Optional[str]:
        """
        Find if we have a local version of a remote model.
        
        Args:
            model_id: Remote model ID
            
        Returns:
            Local model ID if found, None otherwise
        """
        # Check if we have the exact model
        for local_model_id, model_info in self.models.items():
            if model_info.source == ModelSource.LOCAL_HUGGINGFACE:
                # Extract remote model ID from local model ID
                remote_id_from_local = local_model_id.split('/')[-1]
                remote_id_parts = model_id.split('/')
                
                # Check if the model name matches (ignoring organization)
                if remote_id_parts[-1] == remote_id_from_local:
                    return local_model_id
        
        return None
    
    def _call_local_model(self, model_path: str, input_text: str, **kwargs) -> Any:
        """
        Call a local Hugging Face model.
        
        Args:
            model_path: Path to the local model
            input_text: Input text
            **kwargs: Additional parameters
            
        Returns:
            Model output
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        
        try:
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            
            # Create generation pipeline
            generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
            
            # Set generation parameters
            parameters = kwargs.get("parameters", {})
            if "temperature" not in parameters:
                parameters["temperature"] = 0.7
            if "max_new_tokens" not in parameters:
                parameters["max_new_tokens"] = 512
                
            # Generate text
            result = generator(input_text, **parameters)
            
            return result[0]["generated_text"]
            
        except Exception as e:
            logger.error(f"Error using local model: {e}")
            # Fall back to Hugging Face API
            return asyncio.run(self._call_huggingface_api(model_path, input_text, **kwargs))


class PurposeAPIClient:
    """
    Client for the Purpose API.
    
    This is a wrapper around ModelHub that provides a simpler interface for
    common tasks in the Purpose pipeline.
    """
    
    def __init__(self, api_token: str, config_path: Optional[str] = None):
        """
        Initialize the Purpose API client.
        
        Args:
            api_token: API token for authentication
            config_path: Path to config file (optional)
        """
        self.api_token = api_token
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
        
        # Initialize model hub
        self.model_hub = ModelHub(config_path)
        
        # Map task types to appropriate models with fallbacks
        self.task_model_map = {
            "base_training": ["meta-llama/llama-3-8b", "mistralai/mistral-7b-v0.1", "microsoft/phi-3-medium-4k-instruct"],
            "distillation_target": ["microsoft/phi-3-mini-4k-instruct", "google/gemma-2-2b-it", "TinyLlama/TinyLlama-1.1B"],
            "data_processing": ["mistralai/mistral-7b-instruct-v0.2", "databricks/dolly-v2-3b", "google/flan-t5-xl"],
            "knowledge_mapping": ["Qwen/Qwen1.5-7B", "allenai/OLMo-7B", "bigscience/bloomz-7b1"],
            "knowledge_extraction": ["meta-llama/llama-3-8b", "allenai/tulu-2-7b", "HuggingFaceH4/zephyr-7b-beta"],
            "query_generation": ["mistralai/Mixtral-8x7B-Instruct-v0.1", "bigscience/T0_3B", "databricks/dolly-v2-12b"],
            "response_generation": ["01-ai/Yi-34B", "meta-llama/llama-3-70b", "mistralai/Mixtral-8x22B-v0.1"],
            "curriculum_learning": ["meta-llama/llama-3-8b", "togethercomputer/RedPajama-INCITE-7B-Instruct", "mistralai/mistral-7b-instruct-v0.2"],
            "inference": ["microsoft/phi-3-small-4k-instruct", "google/gemma-2-9b-it", "mistralai/mistral-7b-instruct-v0.2"],
            "text_embedding": ["intfloat/e5-large-v2", "BAAI/bge-large-en-v1.5", "sentence-transformers/all-mpnet-base-v2"],
            "text_classification": ["facebook/bart-large-mnli", "cross-encoder/nli-roberta-base", "distilbert-base-uncased-finetuned-sst-2-english"],
            "reasoning": ["allenai/tulu-2-dpo-70b", "google/gemma-2-27b-it", "meta-llama/llama-3-70b"],
            "code_generation": ["Salesforce/codegen25-7b-instruct", "bigcode/starcoder2-15b", "replit/replit-code-v1.5-3b"],
        }
    
    async def process_task(self, task_type: str, input_text: str, **kwargs) -> Any:
        """
        Process a task using the appropriate model via API.
        
        Args:
            task_type: Task type
            input_text: Input text
            **kwargs: Additional parameters
            
        Returns:
            API response
        """
        task_enum = TaskType(task_type) if task_type in [t.value for t in TaskType] else None
        
        # Get model candidates
        model_candidates = self.task_model_map.get(task_type, [])
        if not model_candidates:
            if task_enum:
                model_candidates = self.model_hub.get_recommended_models(task_enum)
            
            if not model_candidates:
                raise ValueError(f"Unknown task type: {task_type}")
        
        # Try models in order until one succeeds
        last_error = None
        for model_id in model_candidates:
            try:
                return await self.model_hub.process_task(task_type, input_text, model_id=model_id, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to use model {model_id} for task {task_type}: {e}")
                continue
        
        # If all models failed, raise the last error
        if last_error:
            raise last_error
        else:
            raise ValueError(f"No available models for task type: {task_type}")
    
    async def close(self) -> None:
        """Close the client."""
        await self.model_hub.close()


# Example usage
async def example_usage():
    """Example of how to use the Purpose API client."""
    # Initialize client
    client = PurposeAPIClient(api_token="your_token_here")
    
    try:
        # Process a knowledge mapping task
        result = await client.process_task(
            task_type="knowledge_mapping",
            input_text="Map the relationships between quantum mechanics and classical physics",
            parameters={"temperature": 0.5, "max_new_tokens": 1024},
        )
        print(result)
        
        # Process a code generation task with a specific model
        result = await client.process_task(
            task_type="code_generation",
            input_text="Write a Python function to implement the QuickSort algorithm",
            model_id="Salesforce/codegen25-7b-instruct",
            parameters={"temperature": 0.2},
        )
        print(result)
        
    finally:
        # Close client
        await client.close()

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage()) 