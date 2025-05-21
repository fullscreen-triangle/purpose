"""
Base pipeline implementation for the Purpose project.

This module provides the core pipeline functionality for chaining operations.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Callable, TypeVar

from main.utils.interfaces import PipelineStage, Pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("main.log"),
    ]
)
logger = logging.getLogger("main")

T = TypeVar('T')
U = TypeVar('U')

class FunctionStage(PipelineStage):
    """Pipeline stage that wraps a function."""
    
    def __init__(self, func: Callable[[T], U], name: Optional[str] = None):
        """
        Initialize a function stage.
        
        Args:
            func: Function to wrap
            name: Name of the stage (optional)
        """
        self.func = func
        self.name = name or func.__name__
    
    def process(self, input_data: T) -> U:
        """
        Process input data using the wrapped function.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Function output
        """
        logger.debug(f"Running pipeline stage: {self.name}")
        return self.func(input_data)

class BasePipeline(Pipeline):
    """Base pipeline implementation for chaining operations."""
    
    def __init__(self, name: str = "pipeline"):
        """
        Initialize an empty pipeline.
        
        Args:
            name: Name of the pipeline
        """
        self.stages: List[PipelineStage] = []
        self.name = name
        self.logger = logger
    
    def add_stage(self, stage: PipelineStage) -> None:
        """
        Add a stage to the pipeline.
        
        Args:
            stage: Pipeline stage to add
        """
        self.stages.append(stage)
        self.logger.debug(f"Added stage to pipeline '{self.name}': {getattr(stage, 'name', str(stage))}")
    
    def add_function(self, func: Callable[[Any], Any], name: Optional[str] = None) -> None:
        """
        Add a function as a stage to the pipeline.
        
        Args:
            func: Function to add
            name: Name of the stage (optional)
        """
        stage = FunctionStage(func, name)
        self.add_stage(stage)
    
    def run(self, input_data: Any) -> Any:
        """
        Run the pipeline on input data.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Pipeline output
        """
        self.logger.info(f"Running pipeline '{self.name}' with {len(self.stages)} stages")
        
        current_data = input_data
        
        for i, stage in enumerate(self.stages):
            stage_name = getattr(stage, 'name', f"stage_{i}")
            self.logger.debug(f"Running stage {i+1}/{len(self.stages)}: {stage_name}")
            
            try:
                current_data = stage.process(current_data)
            except Exception as e:
                self.logger.error(f"Error in pipeline stage {stage_name}: {str(e)}")
                raise
        
        self.logger.info(f"Pipeline '{self.name}' completed successfully")
        return current_data

class ConditionalPipeline(BasePipeline):
    """Pipeline with conditional branching."""
    
    def __init__(self, name: str = "conditional_pipeline"):
        """
        Initialize a conditional pipeline.
        
        Args:
            name: Name of the pipeline
        """
        super().__init__(name)
        self.conditions: Dict[int, Callable[[Any], bool]] = {}
        self.branches: Dict[int, BasePipeline] = {}
    
    def add_conditional_branch(self, condition: Callable[[Any], bool], branch_pipeline: BasePipeline, position: int) -> None:
        """
        Add a conditional branch to the pipeline.
        
        Args:
            condition: Condition function that returns True if the branch should be taken
            branch_pipeline: Pipeline to run if the condition is True
            position: Position in the main pipeline to insert the branch
        """
        self.conditions[position] = condition
        self.branches[position] = branch_pipeline
        self.logger.debug(f"Added conditional branch at position {position} to pipeline '{self.name}'")
    
    def run(self, input_data: Any) -> Any:
        """
        Run the pipeline with conditional branching.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Pipeline output
        """
        self.logger.info(f"Running conditional pipeline '{self.name}' with {len(self.stages)} stages and {len(self.branches)} branches")
        
        current_data = input_data
        
        for i in range(len(self.stages) + 1):  # +1 to allow branches after all stages
            # Check for conditional branch at this position
            if i in self.conditions:
                condition = self.conditions[i]
                branch = self.branches[i]
                
                if condition(current_data):
                    self.logger.debug(f"Taking conditional branch at position {i}")
                    current_data = branch.run(current_data)
            
            # Run the main pipeline stage if we're not past the end
            if i < len(self.stages):
                stage = self.stages[i]
                stage_name = getattr(stage, 'name', f"stage_{i}")
                self.logger.debug(f"Running stage {i+1}/{len(self.stages)}: {stage_name}")
                
                try:
                    current_data = stage.process(current_data)
                except Exception as e:
                    self.logger.error(f"Error in pipeline stage {stage_name}: {str(e)}")
                    raise
        
        self.logger.info(f"Conditional pipeline '{self.name}' completed successfully")
        return current_data 