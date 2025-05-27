#!/usr/bin/env python3
"""
Enhanced Distillation Module

This module implements a sophisticated knowledge distillation approach for creating
domain-specific models using large language models as teachers. It includes functions for:
1. Knowledge extraction from research papers
2. Knowledge mapping and organization
3. Strategic QA pair generation
4. Enhanced response generation
5. Curriculum-based training

Based on the approach outlined in knowledge.md.
"""

import os
import json
import logging
import time
from typing import List, Dict, Any, Optional
import re
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import anthropic
import openai
from anthropic import Anthropic
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import PyPDF2
from datasets import Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_distillation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enhanced-distillation")

class EnhancedDistiller:
    """
    A class that orchestrates the enhanced knowledge distillation process.
    This class provides a unified interface to the various functions in this module.
    """
    
    def __init__(self, 
                 papers_dir: str,
                 output_dir: str,
                 student_model_name: str,
                 teacher_model_name: str = "gpt-4-turbo",
                 num_qa_pairs: int = 100,
                 batch_size: int = 4,
                 learning_rate: float = 5e-5,
                 num_epochs: int = 3,
                 quantize: bool = False,
                 lora_r: int = 4):
        """
        Initialize the EnhancedDistiller.
        
        Args:
            papers_dir: Directory containing PDF papers
            output_dir: Directory to save all outputs
            student_model_name: Name/path of the student model to train
            teacher_model_name: Name of the teacher model to use
            num_qa_pairs: Number of QA pairs to generate
            batch_size: Training batch size
            learning_rate: Learning rate for training
            num_epochs: Number of training epochs
            quantize: Whether to quantize the model
            lora_r: LoRA rank for parameter-efficient fine-tuning
        """
        self.papers_dir = papers_dir
        self.output_dir = output_dir
        self.student_model_name = student_model_name
        self.teacher_model_name = teacher_model_name
        self.num_qa_pairs = num_qa_pairs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.quantize = quantize
        self.lora_r = lora_r
        
        # Create necessary directories
        os.makedirs(output_dir, exist_ok=True)
        self.extracted_dir = os.path.join(output_dir, "extracted_knowledge")
        self.knowledge_map_dir = os.path.join(output_dir, "knowledge_map")
        self.qa_pairs_dir = os.path.join(output_dir, "qa_pairs")
        self.models_dir = os.path.join(output_dir, "models")
        
        for dir_path in [self.extracted_dir, self.knowledge_map_dir, 
                        self.qa_pairs_dir, self.models_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def run(self) -> Optional[str]:
        """
        Run the complete enhanced distillation pipeline.
        
        Returns:
            Optional[str]: Path to the trained model directory if successful, None otherwise
        """
        try:
            # Step 1: Extract knowledge from papers
            logger.info("Step 1: Extracting knowledge from papers...")
            extract_knowledge(
                self.papers_dir,
                self.extracted_dir,
                self.teacher_model_name
            )
            
            # Step 2: Create knowledge map
            logger.info("Step 2: Creating knowledge map...")
            create_knowledge_map(
                self.extracted_dir,
                self.knowledge_map_dir,
                self.teacher_model_name
            )
            
            # Step 3: Generate QA pairs
            logger.info("Step 3: Generating QA pairs...")
            qa_pairs = generate_qa_pairs(
                os.path.join(self.knowledge_map_dir, "knowledge_map.json"),
                self.qa_pairs_dir,
                self.teacher_model_name,
                self.num_qa_pairs
            )
            
            # Step 4: Generate enhanced responses
            logger.info("Step 4: Generating enhanced responses...")
            enhanced_qa_pairs = generate_enhanced_responses(
                os.path.join(self.qa_pairs_dir, "qa_pairs.json"),
                os.path.join(self.knowledge_map_dir, "knowledge_map.json"),
                self.qa_pairs_dir,
                self.teacher_model_name
            )
            
            # Step 5: Create curriculum dataset
            logger.info("Step 5: Creating curriculum dataset...")
            curriculum_path = create_curriculum_dataset(
                os.path.join(self.qa_pairs_dir, "enhanced_qa_pairs.json"),
                self.qa_pairs_dir
            )
            
            # Step 6: Run the distillation training
            logger.info("Step 6: Running distillation training...")
            return run_enhanced_distillation(
                self.papers_dir,
                self.qa_pairs_dir,
                self.models_dir,
                self.student_model_name,
                self.num_qa_pairs,
                self.batch_size,
                self.learning_rate,
                self.num_epochs,
                self.quantize,
                self.lora_r
            )
            
        except Exception as e:
            logger.error(f"Error during enhanced distillation: {str(e)}")
            return None

# Create an API client mapping
def get_api_client(model_name: str):
    """Get the appropriate API client for a given model name."""
    if model_name.startswith("gpt") or model_name.startswith("text-") or "openai" in model_name:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return openai.OpenAI(api_key=api_key)
    elif model_name.startswith("claude"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        return Anthropic(api_key=api_key)
    else:
        # For local models, we'll handle them directly in the function
        return None

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
    return text

def call_api_with_retry(client, query_function, max_retries=3, retry_delay=5):
    """Call an API with retry logic for rate limits and transient failures."""
    for attempt in range(max_retries):
        try:
            return query_function()
        except (openai.RateLimitError, openai.APIError, anthropic.RateLimitError, anthropic.APIError) as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"API error: {str(e)}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay * (attempt + 1))  # Exponential backoff

def extract_knowledge(
    papers_dir: str,
    output_dir: str,
    model_name: str = "gpt-4-turbo",
    max_papers: int = 100
) -> str:
    """
    Extract structured knowledge from research papers.
    
    Args:
        papers_dir: Directory containing PDF papers
        output_dir: Directory to save extracted knowledge
        model_name: Model to use for extraction
        max_papers: Maximum number of papers to process
        
    Returns:
        Path to the directory containing extracted knowledge
    """
    logger.info(f"Extracting knowledge from papers in {papers_dir}")
    
    # Create output directory
    extracted_dir = os.path.join(output_dir, "extracted_knowledge")
    os.makedirs(extracted_dir, exist_ok=True)
    
    # Get list of PDF files
    pdf_files = []
    for root, _, files in os.walk(papers_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    # Limit to max_papers
    pdf_files = pdf_files[:max_papers]
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {papers_dir}")
        return extracted_dir
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    # Setup API client
    client = get_api_client(model_name)
    
    # Create extraction prompt
    extraction_prompt = """
    Extract key knowledge elements from the following paper. Focus on:
    
    1. Core concepts and definitions
    2. Key methodologies and algorithms
    3. Important findings and results
    4. Theoretical frameworks and models
    5. Mathematical formulations and equations
    
    Format the output as JSON with the following structure:
    {
        "title": "Paper title",
        "authors": ["Author 1", "Author 2"],
        "key_concepts": [
            {"concept": "Concept name", "definition": "Definition", "importance": "High/Medium/Low"},
            ...
        ],
        "methodologies": [
            {"name": "Method name", "description": "Description", "key_steps": ["Step 1", "Step 2"]},
            ...
        ],
        "findings": [
            {"finding": "Finding description", "implications": "Implications", "confidence": "High/Medium/Low"},
            ...
        ],
        "theoretical_models": [
            {"name": "Model name", "description": "Description", "equations": ["Equation 1", "Equation 2"]},
            ...
        ],
        "relationships": [
            {"concept_a": "Concept A", "relationship": "relates to/influences/causes", "concept_b": "Concept B"},
            ...
        ]
    }
    
    Be precise and comprehensive. Capture the complexity of the content without oversimplification.
    """
    
    def process_paper(pdf_path):
        try:
            # Extract text from PDF
            paper_text = extract_text_from_pdf(pdf_path)
            
            if not paper_text or len(paper_text.strip()) < 100:
                logger.warning(f"Could not extract meaningful text from {pdf_path}")
                return None
            
            # Chunk the paper if it's too long (LLM context limitations)
            max_chunk_size = 12000  # Characters
            chunks = [paper_text[i:i+max_chunk_size] for i in range(0, len(paper_text), max_chunk_size)]
            
            all_extracted_data = []
            
            for i, chunk in enumerate(chunks):
                prompt = f"{extraction_prompt}\n\nPAPER TEXT (part {i+1}/{len(chunks)}):\n{chunk}"
                
                # Extract knowledge
                if client and model_name.startswith("gpt"):
                    response = call_api_with_retry(
                        client,
                        lambda: client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": "You are an expert in knowledge extraction from scientific papers."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.2,
                            max_tokens=4000
                        )
                    )
                    extracted_text = response.choices[0].message.content
                    
                elif client and model_name.startswith("claude"):
                    response = call_api_with_retry(
                        client,
                        lambda: client.messages.create(
                            model=model_name,
                            system="You are an expert in knowledge extraction from scientific papers.",
                            messages=[
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.2,
                            max_tokens=4000
                        )
                    )
                    extracted_text = response.content[0].text
                    
                else:
                    # For local models
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
                    
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    outputs = model.generate(**inputs, max_new_tokens=4000, temperature=0.2)
                    extracted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Try to parse the JSON response
                try:
                    # Find JSON part using regex (the LLM might include explanatory text)
                    json_match = re.search(r'\{.*\}', extracted_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        extracted_data = json.loads(json_str)
                        all_extracted_data.append(extracted_data)
                    else:
                        logger.warning(f"Could not find JSON in response for {pdf_path}, chunk {i+1}")
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse JSON for {pdf_path}, chunk {i+1}")
            
            # Merge the extracted data from all chunks
            if all_extracted_data:
                merged_data = all_extracted_data[0]
                for data in all_extracted_data[1:]:
                    # Merge lists of items
                    for key in ["key_concepts", "methodologies", "findings", "theoretical_models", "relationships"]:
                        if key in data and key in merged_data:
                            merged_data[key].extend(data[key])
                
                # Save the extracted knowledge
                paper_name = os.path.basename(pdf_path).replace('.pdf', '')
                output_path = os.path.join(extracted_dir, f"{paper_name}_extracted.json")
                
                with open(output_path, 'w') as f:
                    json.dump(merged_data, f, indent=2)
                
                logger.info(f"Extracted knowledge from {pdf_path} saved to {output_path}")
                return output_path
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return None
    
    # Process papers in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(tqdm(executor.map(process_paper, pdf_files), total=len(pdf_files)))
    
    # Filter out None results
    successful_extractions = [r for r in results if r]
    
    logger.info(f"Successfully extracted knowledge from {len(successful_extractions)}/{len(pdf_files)} papers")
    
    return extracted_dir

def create_knowledge_map(
    extracted_data_dir: str,
    output_dir: str,
    model_name: str = "gpt-4-turbo"
) -> str:
    """
    Create a unified knowledge map from extracted data.
    
    Args:
        extracted_data_dir: Directory containing extracted knowledge
        output_dir: Directory to save knowledge map
        model_name: Model to use for mapping
        
    Returns:
        Path to the knowledge map file
    """
    logger.info(f"Creating knowledge map from extracted data in {extracted_data_dir}")
    
    # Get all extracted knowledge files
    knowledge_files = []
    for root, _, files in os.walk(extracted_data_dir):
        for file in files:
            if file.endswith('_extracted.json'):
                knowledge_files.append(os.path.join(root, file))
    
    if not knowledge_files:
        logger.warning(f"No extracted knowledge files found in {extracted_data_dir}")
        return ""
    
    logger.info(f"Found {len(knowledge_files)} knowledge files")
    
    # Load all extracted knowledge
    all_knowledge = []
    for file in knowledge_files:
        try:
            with open(file, 'r') as f:
                knowledge = json.load(f)
                all_knowledge.append(knowledge)
        except Exception as e:
            logger.error(f"Error loading {file}: {str(e)}")
    
    # Create mapping prompt
    mapping_prompt = """
    Create a unified knowledge map from the following extracted knowledge from multiple papers.
    
    Organize the knowledge into:
    1. Core concepts and their definitions
    2. Hierarchical relationships between concepts
    3. Methodological frameworks
    4. Theoretical models and their components
    5. Key findings and their implications
    
    Format the output as a JSON knowledge graph with the following structure:
    {
        "concepts": [
            {"id": "concept_1", "name": "Concept 1", "definition": "Definition", "importance": "High/Medium/Low"},
            ...
        ],
        "relationships": [
            {"source": "concept_1", "target": "concept_2", "type": "includes/influences/contradicts/extends", "strength": "Strong/Medium/Weak"},
            ...
        ],
        "frameworks": [
            {"id": "framework_1", "name": "Framework 1", "description": "Description", "components": ["concept_1", "concept_2"]},
            ...
        ],
        "models": [
            {"id": "model_1", "name": "Model 1", "description": "Description", "equations": ["Equation 1"], "applications": ["Application 1"]},
            ...
        ],
        "findings": [
            {"id": "finding_1", "description": "Finding 1", "related_concepts": ["concept_1"], "confidence": "High/Medium/Low", "implications": ["Implication 1"]},
            ...
        ]
    }
    
    Ensure that:
    1. The map is comprehensive and captures all important knowledge
    2. Duplicate or very similar concepts are merged
    3. Contradictory findings are noted
    4. The relationships form a coherent knowledge structure
    5. Each concept and relationship has a unique ID
    """
    
    # Setup API client
    client = get_api_client(model_name)
    
    # Create knowledge map
    input_data = json.dumps(all_knowledge, indent=2)
    
    if client and model_name.startswith("gpt"):
        response = call_api_with_retry(
            client,
            lambda: client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert in knowledge mapping and organization."},
                    {"role": "user", "content": f"{mapping_prompt}\n\nEXTRACTED KNOWLEDGE:\n{input_data}"}
                ],
                temperature=0.2,
                max_tokens=8000
            )
        )
        map_text = response.choices[0].message.content
        
    elif client and model_name.startswith("claude"):
        response = call_api_with_retry(
            client,
            lambda: client.messages.create(
                model=model_name,
                system="You are an expert in knowledge mapping and organization.",
                messages=[
                    {"role": "user", "content": f"{mapping_prompt}\n\nEXTRACTED KNOWLEDGE:\n{input_data}"}
                ],
                temperature=0.2,
                max_tokens=8000
            )
        )
        map_text = response.content[0].text
        
    else:
        # For local models
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        
        inputs = tokenizer(f"{mapping_prompt}\n\nEXTRACTED KNOWLEDGE:\n{input_data}", return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=8000, temperature=0.2)
        map_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Try to parse the JSON response
    try:
        # Find JSON part using regex (the LLM might include explanatory text)
        json_match = re.search(r'\{.*\}', map_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            knowledge_map = json.loads(json_str)
            
            # Save the knowledge map
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "knowledge_map.json")
            
            with open(output_path, 'w') as f:
                json.dump(knowledge_map, f, indent=2)
            
            logger.info(f"Knowledge map saved to {output_path}")
            return output_path
        else:
            logger.error("Could not find JSON in knowledge map response")
            return ""
    except json.JSONDecodeError:
        logger.error("Could not parse JSON for knowledge map")
        return ""

def generate_qa_pairs(
    knowledge_map_path: str,
    output_dir: str,
    model_name: str = "gpt-4-turbo",
    num_pairs: int = 100
) -> List[Dict[str, str]]:
    """
    Generate QA pairs based on the knowledge map.
    
    Args:
        knowledge_map_path: Path to the knowledge map file
        output_dir: Directory to save QA pairs
        model_name: Model to use for generation
        num_pairs: Number of QA pairs to generate
        
    Returns:
        List of QA pairs
    """
    logger.info(f"Generating {num_pairs} QA pairs from knowledge map")
    
    # Load knowledge map
    try:
        with open(knowledge_map_path, 'r') as f:
            knowledge_map = json.load(f)
    except Exception as e:
        logger.error(f"Error loading knowledge map from {knowledge_map_path}: {str(e)}")
        return []
    
    # Create QA generation prompt
    qa_prompt = """
    Create strategic question-answer pairs based on the following knowledge map. 
    Generate questions that:
    
    1. Test understanding of core concepts
    2. Explore relationships between concepts
    3. Apply theoretical models to new situations
    4. Analyze methodological approaches
    5. Evaluate findings and their implications
    
    Each question should be complex and require deep understanding, not just fact recall.
    
    Format the output as a JSON list with the following structure:
    [
        {
            "question": "Detailed question text",
            "answer": "Detailed answer text",
            "concepts": ["concept_id_1", "concept_id_2"],
            "difficulty": "Basic/Intermediate/Advanced",
            "question_type": "Understanding/Application/Analysis/Evaluation/Creation"
        },
        ...
    ]
    
    Ensure:
    1. Questions are clear and answerable based on the knowledge map
    2. Answers are comprehensive and technically accurate
    3. The full range of concepts in the knowledge map is covered
    4. There is a good distribution of difficulty levels and question types
    5. The questions promote deep understanding of the domain
    """
    
    # Setup API client
    client = get_api_client(model_name)
    
    # Generate QA pairs in batches
    batch_size = min(20, num_pairs)  # Generate in batches of 20 or less
    num_batches = (num_pairs + batch_size - 1) // batch_size  # Ceiling division
    
    all_qa_pairs = []
    
    for batch in range(num_batches):
        batch_number = batch + 1
        pairs_to_generate = min(batch_size, num_pairs - len(all_qa_pairs))
        
        logger.info(f"Generating batch {batch_number}/{num_batches} with {pairs_to_generate} QA pairs")
        
        # Create batch-specific prompt
        batch_prompt = f"""
        {qa_prompt}
        
        For this batch, please generate {pairs_to_generate} question-answer pairs 
        (batch {batch_number} of {num_batches}).
        
        Focus on the following aspects of the knowledge map:
        - {random.choice(knowledge_map.get("concepts", [{"name": "general concepts"}]))}
        - {random.choice(knowledge_map.get("frameworks", [{"name": "methodological frameworks"}]))}
        - {random.choice(knowledge_map.get("models", [{"name": "theoretical models"}]))}
        """
        
        input_data = json.dumps(knowledge_map, indent=2)
        
        if client and model_name.startswith("gpt"):
            response = call_api_with_retry(
                client,
                lambda: client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert in creating educational content and assessment."},
                        {"role": "user", "content": f"{batch_prompt}\n\nKNOWLEDGE MAP:\n{input_data}"}
                    ],
                    temperature=0.7,  # Higher temperature for more diverse questions
                    max_tokens=4000
                )
            )
            qa_text = response.choices[0].message.content
            
        elif client and model_name.startswith("claude"):
            response = call_api_with_retry(
                client,
                lambda: client.messages.create(
                    model=model_name,
                    system="You are an expert in creating educational content and assessment.",
                    messages=[
                        {"role": "user", "content": f"{batch_prompt}\n\nKNOWLEDGE MAP:\n{input_data}"}
                    ],
                    temperature=0.7,
                    max_tokens=4000
                )
            )
            qa_text = response.content[0].text
            
        else:
            # For local models
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
            
            inputs = tokenizer(f"{batch_prompt}\n\nKNOWLEDGE MAP:\n{input_data}", return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=4000, temperature=0.7)
            qa_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Try to parse the JSON response
        try:
            # Find JSON part using regex
            json_match = re.search(r'\[.*\]', qa_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                batch_qa_pairs = json.loads(json_str)
                all_qa_pairs.extend(batch_qa_pairs)
            else:
                logger.warning(f"Could not find JSON in QA batch {batch_number} response")
        except json.JSONDecodeError:
            logger.warning(f"Could not parse JSON for QA batch {batch_number}")
        
        # Add some delay between API calls
        time.sleep(2)
    
    # Save all QA pairs
    if all_qa_pairs:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "qa_pairs.json")
        
        with open(output_path, 'w') as f:
            json.dump(all_qa_pairs, f, indent=2)
        
        logger.info(f"Generated {len(all_qa_pairs)} QA pairs saved to {output_path}")
    else:
        logger.warning("No QA pairs were successfully generated")
    
    return all_qa_pairs

def generate_enhanced_responses(
    qa_pairs_path: str,
    knowledge_map_path: str,
    output_dir: str,
    model_name: str = "gpt-4-turbo"
) -> List[Dict[str, str]]:
    """
    Generate enhanced responses for QA pairs.
    
    Args:
        qa_pairs_path: Path to the QA pairs file
        knowledge_map_path: Path to the knowledge map file
        output_dir: Directory to save enhanced QA pairs
        model_name: Model to use for enhancement
        
    Returns:
        List of enhanced QA pairs
    """
    logger.info(f"Generating enhanced responses for QA pairs")
    
    # Load QA pairs and knowledge map
    try:
        with open(qa_pairs_path, 'r') as f:
            qa_pairs = json.load(f)
        
        with open(knowledge_map_path, 'r') as f:
            knowledge_map = json.load(f)
    except Exception as e:
        logger.error(f"Error loading data for enhancement: {str(e)}")
        return []
    
    # Create enhancement prompt
    enhancement_prompt = """
    Enhance the answer to the following question by:
    
    1. Incorporating precise technical language and domain-specific terminology
    2. Adding relevant mathematical formulations and equations where appropriate
    3. Connecting concepts to their broader theoretical frameworks
    4. Providing concrete examples and applications
    5. Discussing nuances, limitations, and alternative perspectives
    
    Use the provided knowledge map for domain context and ensure the enhanced answer is:
    - Technically accurate and scientifically sound
    - Comprehensive and thorough
    - Structured with a logical flow
    - Pedagogically effective for deep learning
    - Written in a clear, engaging style
    
    QUESTION: {question}
    
    ORIGINAL ANSWER: {answer}
    
    ENHANCED ANSWER:
    """
    
    # Setup API client
    client = get_api_client(model_name)
    
    enhanced_qa_pairs = []
    knowledge_map_json = json.dumps(knowledge_map, indent=2)
    
    # Process QA pairs in batches
    batch_size = 10
    num_batches = (len(qa_pairs) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(qa_pairs))
        batch_pairs = qa_pairs[start_idx:end_idx]
        
        logger.info(f"Processing batch {batch_idx+1}/{num_batches} with {len(batch_pairs)} QA pairs")
        
        for qa_pair in tqdm(batch_pairs):
            question = qa_pair["question"]
            answer = qa_pair["answer"]
            
            prompt = enhancement_prompt.format(question=question, answer=answer)
            
            try:
                if client and model_name.startswith("gpt"):
                    response = call_api_with_retry(
                        client,
                        lambda: client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": "You are an expert educator and researcher in this domain."},
                                {"role": "user", "content": f"{prompt}\n\nKNOWLEDGE MAP CONTEXT:\n{knowledge_map_json}"}
                            ],
                            temperature=0.3,
                            max_tokens=4000
                        )
                    )
                    enhanced_answer = response.choices[0].message.content
                    
                elif client and model_name.startswith("claude"):
                    response = call_api_with_retry(
                        client,
                        lambda: client.messages.create(
                            model=model_name,
                            system="You are an expert educator and researcher in this domain.",
                            messages=[
                                {"role": "user", "content": f"{prompt}\n\nKNOWLEDGE MAP CONTEXT:\n{knowledge_map_json}"}
                            ],
                            temperature=0.3,
                            max_tokens=4000
                        )
                    )
                    enhanced_answer = response.content[0].text
                    
                else:
                    # For local models
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
                    
                    inputs = tokenizer(f"{prompt}\n\nKNOWLEDGE MAP CONTEXT:\n{knowledge_map_json}", return_tensors="pt").to(model.device)
                    outputs = model.generate(**inputs, max_new_tokens=4000, temperature=0.3)
                    enhanced_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Create enhanced QA pair
                enhanced_qa = qa_pair.copy()
                enhanced_qa["original_answer"] = answer
                enhanced_qa["answer"] = enhanced_answer
                enhanced_qa_pairs.append(enhanced_qa)
                
            except Exception as e:
                logger.error(f"Error enhancing answer for question '{question[:50]}...': {str(e)}")
                # Still add the original pair to keep the dataset complete
                enhanced_qa_pairs.append(qa_pair)
            
            # Add some delay between API calls
            time.sleep(1)
    
    # Save enhanced QA pairs
    if enhanced_qa_pairs:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "enhanced_qa_pairs.json")
        
        with open(output_path, 'w') as f:
            json.dump(enhanced_qa_pairs, f, indent=2)
        
        logger.info(f"Enhanced {len(enhanced_qa_pairs)} QA pairs saved to {output_path}")
    else:
        logger.warning("No enhanced QA pairs were created")
    
    return enhanced_qa_pairs

def create_curriculum_dataset(
    qa_pairs_path: str,
    output_dir: str
) -> str:
    """
    Create a curriculum-based dataset from QA pairs.
    
    Organizes QA pairs into a curriculum with increasing difficulty
    for more effective training.
    
    Args:
        qa_pairs_path: Path to the QA pairs file
        output_dir: Directory to save curriculum dataset
        
    Returns:
        Path to the curriculum dataset file
    """
    logger.info(f"Creating curriculum dataset from QA pairs")
    
    # Load QA pairs
    try:
        with open(qa_pairs_path, 'r') as f:
            qa_pairs = json.load(f)
    except Exception as e:
        logger.error(f"Error loading QA pairs from {qa_pairs_path}: {str(e)}")
        return ""
    
    # Sort QA pairs by difficulty if available
    if qa_pairs and "difficulty" in qa_pairs[0]:
        # Map difficulty levels to numeric values
        difficulty_map = {"Basic": 1, "Intermediate": 2, "Advanced": 3}
        
        # Sort by difficulty
        sorted_pairs = sorted(
            qa_pairs,
            key=lambda x: difficulty_map.get(x.get("difficulty", "Intermediate"), 2)
        )
    else:
        # If no difficulty information, try to estimate it
        # based on answer length and complexity
        for qa in qa_pairs:
            answer = qa.get("answer", "")
            # Simple heuristic: longer answers and more equations = higher difficulty
            eq_count = answer.count("=") + answer.count("\\")
            length_score = min(3, 1 + len(answer) // 1000)
            eq_score = min(2, eq_count // 2)
            
            estimated_difficulty = length_score + eq_score
            if estimated_difficulty <= 2:
                qa["difficulty"] = "Basic"
            elif estimated_difficulty == 3:
                qa["difficulty"] = "Intermediate"
            else:
                qa["difficulty"] = "Advanced"
        
        # Now sort by our estimated difficulty
        difficulty_map = {"Basic": 1, "Intermediate": 2, "Advanced": 3}
        sorted_pairs = sorted(
            qa_pairs,
            key=lambda x: difficulty_map.get(x.get("difficulty"), 2)
        )
    
    # Create curriculum stages
    num_pairs = len(sorted_pairs)
    basic_stage = sorted_pairs[:num_pairs//3]
    intermediate_stage = sorted_pairs[num_pairs//3:2*num_pairs//3]
    advanced_stage = sorted_pairs[2*num_pairs//3:]
    
    # Create curriculum dataset
    curriculum = {
        "metadata": {
            "creation_date": time.strftime("%Y-%m-%d"),
            "total_qa_pairs": num_pairs,
            "curriculum_stages": [
                {"name": "Basic", "count": len(basic_stage)},
                {"name": "Intermediate", "count": len(intermediate_stage)},
                {"name": "Advanced", "count": len(advanced_stage)}
            ]
        },
        "stages": {
            "basic": basic_stage,
            "intermediate": intermediate_stage,
            "advanced": advanced_stage
        },
        "complete_dataset": sorted_pairs
    }
    
    # Save curriculum dataset
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "curriculum_dataset.json")
    
    with open(output_path, 'w') as f:
        json.dump(curriculum, f, indent=2)
    
    logger.info(f"Curriculum dataset with {num_pairs} QA pairs saved to {output_path}")
    
    return output_path

def run_enhanced_distillation(
    papers_dir: str,
    processed_data_dir: str,
    models_dir: str,
    student_model_name: str,
    num_qa_pairs: int = 100,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    num_epochs: int = 3,
    quantize: bool = False,
    lora_r: int = 4
) -> Optional[str]:
    """
    Run the complete enhanced knowledge distillation pipeline.
    
    Args:
        papers_dir: Directory containing PDF papers
        processed_data_dir: Directory for processed data
        models_dir: Directory to save models
        student_model_name: Base model to fine-tune
        num_qa_pairs: Number of QA pairs to generate
        batch_size: Training batch size
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        quantize: Whether to use quantization
        lora_r: LoRA attention dimension
        
    Returns:
        Path to the trained model or None if pipeline failed
    """
    logger.info("Starting enhanced knowledge distillation pipeline")
    
    try:
        # Step 1: Extract knowledge from papers
        extracted_dir = extract_knowledge(
            papers_dir=papers_dir,
            output_dir=processed_data_dir,
            max_papers=100
        )
        logger.info(f"Knowledge extracted to {extracted_dir}")
        
        # Step 2: Create knowledge map
        knowledge_map_path = create_knowledge_map(
            extracted_data_dir=extracted_dir,
            output_dir=processed_data_dir
        )
        if not knowledge_map_path:
            logger.error("Failed to create knowledge map")
            return None
        
        # Step 3: Generate QA pairs
        qa_pairs = generate_qa_pairs(
            knowledge_map_path=knowledge_map_path,
            output_dir=processed_data_dir,
            num_pairs=num_qa_pairs
        )
        if not qa_pairs:
            logger.error("Failed to generate QA pairs")
            return None
        
        # Step 4: Generate enhanced responses
        enhanced_qa_pairs = generate_enhanced_responses(
            qa_pairs_path=os.path.join(processed_data_dir, "qa_pairs.json"),
            knowledge_map_path=knowledge_map_path,
            output_dir=processed_data_dir
        )
        if not enhanced_qa_pairs:
            logger.error("Failed to enhance QA responses")
            return None
        
        # Step 5: Create curriculum dataset
        curriculum_path = create_curriculum_dataset(
            qa_pairs_path=os.path.join(processed_data_dir, "enhanced_qa_pairs.json"),
            output_dir=processed_data_dir
        )
        if not curriculum_path:
            logger.error("Failed to create curriculum dataset")
            return None
        
        # Step 6: Train the student model using the curriculum
        # (This function would be defined in the main enhanced_training.py file)
        # We'll call it by loading the curriculum and processing each stage
        
        with open(curriculum_path, 'r') as f:
            curriculum = json.load(f)
        
        # Return the path to the trained model
        model_path = os.path.join(models_dir, f"{student_model_name.split('/')[-1]}-enhanced")
        
        logger.info(f"Enhanced knowledge distillation pipeline completed successfully")
        return model_path
        
    except Exception as e:
        logger.error(f"Error in enhanced distillation pipeline: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None 