"""
Example demonstrating how to use specialized models in a Purpose pipeline.

This example shows how to construct a pipeline that uses different specialized
models for different stages of processing.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List

from purpose.pipelines.base import BasePipeline, FunctionStage
from purpose.utils.model_hub import PurposeAPIClient, TaskType
from purpose.knowledge.structure import KnowledgeGraph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("specialized_pipeline_example")

class ModelStage(FunctionStage):
    """Pipeline stage that uses a specialized model for a specific task."""
    
    def __init__(self, task_type: str, api_client: PurposeAPIClient, name: str = None):
        """
        Initialize a model stage.
        
        Args:
            task_type: Type of task for this stage
            api_client: PurposeAPIClient instance
            name: Name of the stage (optional)
        """
        self.task_type = task_type
        self.api_client = api_client
        super().__init__(self._process_with_model, name or f"{task_type}_stage")
    
    async def _process_with_model(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data using the appropriate specialized model.
        
        Args:
            input_data: Input data containing 'text' and optional 'parameters'
            
        Returns:
            Processed data with model outputs
        """
        text = input_data.get("text", "")
        parameters = input_data.get("parameters", {})
        
        try:
            # Process the text using the appropriate model for this task
            result = await self.api_client.process_task(
                task_type=self.task_type,
                input_text=text,
                **parameters
            )
            
            # Add the result to the output
            output = input_data.copy()
            output["model_output"] = result
            return output
            
        except Exception as e:
            logger.error(f"Error in model stage {self.name}: {str(e)}")
            raise

class SpecializedPipeline:
    """
    Pipeline that uses specialized models for different stages.
    
    This pipeline demonstrates how to use different specialized models
    for different stages of a knowledge processing pipeline.
    """
    
    def __init__(self, api_token: str = None, config_path: str = None):
        """
        Initialize the specialized pipeline.
        
        Args:
            api_token: API token for authentication (defaults to env var)
            config_path: Path to config file with API keys
        """
        # Use environment variable if token not provided
        if api_token is None:
            api_token = os.environ.get("HUGGINGFACE_API_KEY", "")
        
        # Initialize the API client
        self.api_client = PurposeAPIClient(api_token=api_token, config_path=config_path)
        
        # Create the pipeline
        self.pipeline = BasePipeline(name="specialized_knowledge_pipeline")
        self._build_pipeline()
    
    def _build_pipeline(self):
        """Build the pipeline with specialized model stages."""
        # Stage 1: Extract knowledge from input text
        self.pipeline.add_stage(ModelStage(
            task_type=TaskType.KNOWLEDGE_EXTRACTION.value,
            api_client=self.api_client,
            name="knowledge_extraction"
        ))
        
        # Stage 2: Map knowledge into structured format
        self.pipeline.add_function(self._organize_knowledge, "organize_knowledge")
        
        # Stage 3: Use specialized model for knowledge mapping
        self.pipeline.add_stage(ModelStage(
            task_type=TaskType.KNOWLEDGE_MAPPING.value,
            api_client=self.api_client,
            name="knowledge_mapping"
        ))
        
        # Stage 4: Convert to knowledge graph
        self.pipeline.add_function(self._create_knowledge_graph, "create_knowledge_graph")
        
        # Stage 5: Generate queries based on the knowledge graph
        self.pipeline.add_stage(ModelStage(
            task_type=TaskType.QUERY_GENERATION.value,
            api_client=self.api_client,
            name="query_generation"
        ))
        
        # Stage 6: Generate detailed responses to queries
        self.pipeline.add_stage(ModelStage(
            task_type=TaskType.RESPONSE_GENERATION.value,
            api_client=self.api_client,
            name="response_generation"
        ))
    
    def _organize_knowledge(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Organize extracted knowledge into a structured format.
        
        Args:
            input_data: Input data from previous stage
            
        Returns:
            Structured knowledge data
        """
        # Get the knowledge extraction output
        model_output = input_data.get("model_output", {})
        
        # Parse the model output to extract entities and relationships
        # This is a simplified example - in practice, you would have more complex parsing
        extracted_knowledge = self._parse_model_output(model_output)
        
        # Create prompt for the knowledge mapping model
        mapping_prompt = self._create_mapping_prompt(extracted_knowledge)
        
        # Update the output
        output = input_data.copy()
        output["text"] = mapping_prompt
        output["extracted_knowledge"] = extracted_knowledge
        return output
    
    def _parse_model_output(self, model_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse model output to extract structured knowledge.
        
        Args:
            model_output: Output from the knowledge extraction model
            
        Returns:
            Structured knowledge
        """
        # In a real implementation, this would parse the model's output format
        # This is a simplified example
        if isinstance(model_output, dict) and "generated_text" in model_output:
            text = model_output["generated_text"]
        elif isinstance(model_output, list) and len(model_output) > 0:
            text = model_output[0].get("generated_text", "")
        else:
            text = str(model_output)
        
        # Very basic entity extraction - in practice, use more sophisticated NLP
        entities = []
        relationships = []
        lines = text.split("\n")
        
        for line in lines:
            line = line.strip()
            if line.startswith("Entity:"):
                entities.append(line[7:].strip())
            elif line.startswith("Relationship:"):
                relationships.append(line[13:].strip())
        
        return {
            "entities": entities,
            "relationships": relationships,
            "raw_text": text
        }
    
    def _create_mapping_prompt(self, extracted_knowledge: Dict[str, Any]) -> str:
        """
        Create a prompt for the knowledge mapping model.
        
        Args:
            extracted_knowledge: Extracted knowledge from previous stage
            
        Returns:
            Prompt for knowledge mapping model
        """
        entities = extracted_knowledge.get("entities", [])
        relationships = extracted_knowledge.get("relationships", [])
        
        prompt = "Map the following knowledge into a structured format:\n\n"
        
        if entities:
            prompt += "Entities:\n"
            for entity in entities:
                prompt += f"- {entity}\n"
        
        if relationships:
            prompt += "\nRelationships:\n"
            for relationship in relationships:
                prompt += f"- {relationship}\n"
        
        prompt += "\nProvide a structured representation of this knowledge in JSON format with nodes, edges, and attributes."
        return prompt
    
    def _create_knowledge_graph(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert mapped knowledge into a knowledge graph.
        
        Args:
            input_data: Input data from previous stage
            
        Returns:
            Data with knowledge graph
        """
        model_output = input_data.get("model_output", {})
        
        # Parse the model output to extract structured knowledge
        if isinstance(model_output, dict) and "generated_text" in model_output:
            text = model_output["generated_text"]
        elif isinstance(model_output, list) and len(model_output) > 0:
            text = model_output[0].get("generated_text", "")
        else:
            text = str(model_output)
        
        # In a real implementation, parse JSON or other structured format
        # Here, we'll create a simple knowledge graph
        try:
            # Attempt to create a knowledge graph
            # In a real implementation, parse the model's structured output
            graph = KnowledgeGraph("Example Graph")
            
            # Add some example nodes and edges based on text
            # This is highly simplified - real implementation would parse the actual structure
            for i, line in enumerate(text.split("\n")):
                if "node" in line.lower() or "entity" in line.lower():
                    graph.add_node(f"node_{i}", {"label": line.strip()})
                elif "edge" in line.lower() or "relation" in line.lower():
                    graph.add_edge(f"node_{i-1}", f"node_{i}", {"label": line.strip()})
            
            # Create prompt for query generation
            query_prompt = f"Based on the following knowledge graph, generate 3 insightful questions:\n\n{text}"
            
            # Update output
            output = input_data.copy()
            output["knowledge_graph"] = graph
            output["text"] = query_prompt
            return output
            
        except Exception as e:
            logger.error(f"Error creating knowledge graph: {str(e)}")
            # Fall back to a simpler approach
            output = input_data.copy()
            output["text"] = f"Generate 3 questions about the following information:\n\n{text}"
            return output
    
    async def process(self, input_text: str) -> Dict[str, Any]:
        """
        Process input text through the specialized pipeline.
        
        Args:
            input_text: Input text to process
            
        Returns:
            Processed data with results from all stages
        """
        try:
            # Prepare initial input
            input_data = {
                "text": input_text,
                "parameters": {
                    "parameters": {
                        "temperature": 0.7,
                        "max_new_tokens": 512
                    }
                }
            }
            
            # Run the pipeline synchronously by awaiting async stages
            current_data = input_data
            for i, stage in enumerate(self.pipeline.stages):
                stage_name = getattr(stage, 'name', f"stage_{i}")
                logger.info(f"Running stage {i+1}/{len(self.pipeline.stages)}: {stage_name}")
                
                if isinstance(stage, ModelStage):
                    # For ModelStage, we need to await the processing
                    current_data = await stage._process_with_model(current_data)
                else:
                    # For regular stages, use normal processing
                    current_data = stage.process(current_data)
            
            return current_data
            
        except Exception as e:
            logger.error(f"Error in pipeline: {str(e)}")
            raise
        finally:
            # Ensure the API client is closed
            await self.api_client.close()

async def run_example():
    """Run the specialized pipeline example."""
    # Sample text about quantum physics
    sample_text = """
    Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles. It is the foundation of all quantum physics including quantum chemistry, quantum field theory, quantum technology, and quantum information science.

    Classical physics, the description of physics that existed before the theory of relativity and quantum mechanics, describes many aspects of nature at an ordinary (macroscopic) scale, while quantum mechanics explains the aspects of nature at small (atomic and subatomic) scales.

    Most theories in classical physics can be derived from quantum mechanics as an approximation valid at large (macroscopic) scale. Quantum mechanics differs from classical physics in that energy, momentum, angular momentum, and other quantities of a bound system are restricted to discrete values (quantization), objects have characteristics of both particles and waves (wave-particle duality), and there are limits to how accurately the value of a physical quantity can be predicted prior to its measurement, given a complete set of initial conditions (the uncertainty principle).
    """
    
    # Initialize the pipeline
    pipeline = SpecializedPipeline()
    
    # Process the input text
    logger.info("Starting processing of sample text...")
    result = await pipeline.process(sample_text)
    
    # Print the final result
    if "model_output" in result:
        if isinstance(result["model_output"], dict) and "generated_text" in result["model_output"]:
            print("\nFinal output:")
            print(result["model_output"]["generated_text"])
        elif isinstance(result["model_output"], list) and len(result["model_output"]) > 0:
            print("\nFinal output:")
            print(result["model_output"][0].get("generated_text", ""))
        else:
            print("\nFinal output:")
            print(result["model_output"])

if __name__ == "__main__":
    # Run the example
    asyncio.run(run_example()) 