"""
Interactive interfaces for using domain-specific language models.
"""

import sys
from typing import Optional

from domain_llm.inference.model import ModelInference

class InteractiveInterface:
    """
    Interactive command-line interface for domain-specific LLMs.
    """
    
    def __init__(
        self,
        model_inference: ModelInference,
        mode: str = "chat",
        welcome_message: Optional[str] = None,
        exit_commands: Optional[list] = None
    ):
        """
        Initialize the interactive interface.
        
        Args:
            model_inference: ModelInference instance
            mode: Interface mode ('chat' or 'qa')
            welcome_message: Custom welcome message
            exit_commands: List of commands to exit the interface
        """
        self.model = model_inference
        self.mode = mode
        
        # Set default welcome message based on mode
        self.welcome_message = welcome_message
        if self.welcome_message is None:
            if self.mode == "chat":
                self.welcome_message = "Domain-Specific LLM Chat\nType 'exit', 'quit', or 'q' to exit."
            else:
                self.welcome_message = "Domain-Specific LLM Q&A\nType 'exit', 'quit', or 'q' to exit."
        
        # Set exit commands
        self.exit_commands = exit_commands or ["exit", "quit", "q"]
        
        # Store conversation history
        self.history = []
    
    def run(self):
        """
        Run the interactive interface.
        """
        # Print welcome message
        print("\n" + self.welcome_message)
        print("=" * len(self.welcome_message.split("\n")[0]))
        
        # Interactive loop
        while True:
            try:
                # Get user input
                if self.mode == "chat":
                    user_input = input("\nYou: ")
                else:
                    user_input = input("\nEnter your question: ")
                
                # Check if user wants to exit
                if user_input.lower() in self.exit_commands:
                    print("Exiting interactive mode.")
                    break
                
                # Generate response
                print("\nGenerating response...")
                
                if self.mode == "chat":
                    response = self.model.generate(
                        prompt=user_input,
                        max_length=512,
                        temperature=0.7
                    )
                    print(f"\nAssistant: {response}")
                else:
                    answer = self.model.answer_question(
                        question=user_input,
                        temperature=0.7
                    )
                    print(f"\nAnswer: {answer}")
                
                # Store in history
                self.history.append({
                    "input": user_input,
                    "output": response if self.mode == "chat" else answer
                })
                
            except KeyboardInterrupt:
                print("\nExiting interactive mode.")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")


class QAInterface:
    """
    Simple question-answering interface for domain-specific LLMs.
    """
    
    def __init__(
        self,
        model_inference: ModelInference
    ):
        """
        Initialize the QA interface.
        
        Args:
            model_inference: ModelInference instance
        """
        self.model = model_inference
    
    def answer_question(
        self,
        question: str,
        context: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Answer a question using the model.
        
        Args:
            question: Question to answer
            context: Optional context for the question
            **kwargs: Additional generation parameters
            
        Returns:
            Answer to the question
        """
        return self.model.answer_question(
            question=question,
            context=context,
            **kwargs
        )
    
    def answer_from_file(
        self,
        question: str,
        context_file: str,
        **kwargs
    ) -> str:
        """
        Answer a question using context from a file.
        
        Args:
            question: Question to answer
            context_file: Path to a file containing context
            **kwargs: Additional generation parameters
            
        Returns:
            Answer to the question
        """
        # Read context from file
        with open(context_file, 'r', encoding='utf-8') as f:
            context = f.read()
        
        return self.answer_question(
            question=question,
            context=context,
            **kwargs
        ) 