"""
Knowledge Map Module

This module implements the first stage of the enhanced distillation process:
- Structured extraction of research papers
- Creating a knowledge map and taxonomy of the domain
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set
import time

# Third-party imports (assuming these are available)
try:
    import openai
    import anthropic
    import numpy as np
    from transformers import AutoTokenizer
    # Library for graph visualization
    import networkx as nx
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

logger = logging.getLogger(__name__)

class KnowledgeMap:
    """
    Implements the structured extraction and knowledge mapping phase
    from the enhanced distillation process described in knowledge.md.
    """
    
    def __init__(
        self, 
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        openai_model: str = "gpt-4", 
        claude_model: str = "claude-3-opus-20240229",
        output_dir: str = "output/knowledge_map",
        use_multiple_models: bool = True
    ):
        """
        Initialize the KnowledgeMap with API keys and model configuration.
        
        Args:
            openai_api_key: OpenAI API key (if None, tries to get from environment)
            anthropic_api_key: Anthropic API key (if None, tries to get from environment)
            openai_model: OpenAI model to use for knowledge extraction
            claude_model: Claude model to use (if available)
            output_dir: Directory to save knowledge map outputs
            use_multiple_models: Whether to use multiple models for extraction
        """
        # Initialize outputs directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI client
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if self.openai_api_key:
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
            self.openai_model = openai_model
        else:
            self.openai_client = None
            logger.warning("No OpenAI API key provided.")
        
        # Initialize Anthropic client (if key provided)
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.anthropic_client = None
        if self.anthropic_api_key and use_multiple_models:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
                self.claude_model = claude_model
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic client: {e}")
                
        self.use_multiple_models = use_multiple_models and self.anthropic_client is not None
        
        # Initialize empty knowledge structures
        self.knowledge_map = {
            "core_concepts": [],
            "terminology": {},
            "research_questions": [],
            "methodologies": [],
            "key_findings": [],
            "measurements": []
        }
        
        self.taxonomy = {}
        self.concept_graph = None
        
        # Initialize tokenizer for text chunking
        if DEPS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            except Exception:
                self.tokenizer = None
                logger.warning("Could not load tokenizer. Will use character counts for chunking.")
        else:
            self.tokenizer = None
            
    def process_paper(self, paper_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a single research paper and extract structured knowledge.
        
        Args:
            paper_path: Path to the research paper (PDF or TXT)
            
        Returns:
            Dictionary containing extracted knowledge from the paper
        """
        path = Path(paper_path)
        if not path.exists():
            raise FileNotFoundError(f"Paper not found: {paper_path}")
            
        logger.info(f"Processing paper: {path.name}")
        
        # Read paper content (assuming text or PDF)
        text_content = self._read_paper_content(path)
        if not text_content:
            logger.warning(f"Failed to extract text from {path.name}")
            return {}
            
        # Extract structured knowledge
        extracted_knowledge = self._extract_structured_knowledge(text_content, path.stem)
        
        # Save the extracted knowledge
        output_file = self.output_dir / f"extracted_{path.stem}.json"
        with open(output_file, "w") as f:
            json.dump(extracted_knowledge, f, indent=2)
            
        logger.info(f"Saved extracted knowledge from {path.name} to {output_file}")
        
        return extracted_knowledge
    
    def process_papers(self, papers_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Process all papers in a directory and build a comprehensive knowledge map.
        
        Args:
            papers_dir: Directory containing research papers
            
        Returns:
            Complete knowledge map
        """
        papers_dir = Path(papers_dir)
        if not papers_dir.exists() or not papers_dir.is_dir():
            raise ValueError(f"Invalid papers directory: {papers_dir}")
            
        # Find all PDF files
        paper_files = list(papers_dir.glob("*.pdf")) + list(papers_dir.glob("*.txt"))
        logger.info(f"Found {len(paper_files)} papers in {papers_dir}")
        
        if not paper_files:
            logger.warning(f"No papers found in {papers_dir}")
            return self.knowledge_map
            
        # Process each paper
        all_extractions = []
        for paper_path in paper_files:
            try:
                extracted = self.process_paper(paper_path)
                if extracted:
                    all_extractions.append(extracted)
            except Exception as e:
                logger.error(f"Error processing {paper_path.name}: {str(e)}")
                continue
                
        # Combine all extractions into a single knowledge map
        self.knowledge_map = self._combine_extractions(all_extractions)
        
        # Create the taxonomy
        self.taxonomy = self._create_taxonomy(self.knowledge_map)
        
        # Save the complete knowledge map
        map_path = self.output_dir / "knowledge_map.json"
        with open(map_path, "w") as f:
            json.dump(self.knowledge_map, f, indent=2)
            
        # Save the taxonomy
        taxonomy_path = self.output_dir / "taxonomy.json"
        with open(taxonomy_path, "w") as f:
            json.dump(self.taxonomy, f, indent=2)
            
        logger.info(f"Saved complete knowledge map to {map_path}")
        logger.info(f"Saved taxonomy to {taxonomy_path}")
        
        # Create and save concept graph if dependencies available
        if DEPS_AVAILABLE:
            self._create_concept_graph()
            
        return self.knowledge_map
    
    def _read_paper_content(self, paper_path: Path) -> str:
        """
        Read content from a paper file (PDF or text).
        
        Args:
            paper_path: Path to the paper file
            
        Returns:
            Extracted text content
        """
        # For a real implementation, use PyPDF2 or other PDF extraction libraries
        # Here we'll just read text files or assume text content
        
        if paper_path.suffix.lower() == '.txt':
            try:
                with open(paper_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Failed to read {paper_path}: {e}")
                return ""
                
        elif paper_path.suffix.lower() == '.pdf':
            # TODO: Implement PDF extraction in a real implementation
            # For now, log a warning about PDF extraction
            logger.warning(f"PDF extraction not implemented. Treating {paper_path} as plain text.")
            try:
                with open(paper_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception:
                logger.error(f"Failed to read {paper_path} as text.")
                return ""
        else:
            logger.warning(f"Unsupported file format: {paper_path.suffix}")
            return ""
    
    def _extract_structured_knowledge(self, text: str, paper_id: str) -> Dict[str, Any]:
        """
        Extract structured knowledge from paper text using LLM.
        
        Args:
            text: Paper text content
            paper_id: Identifier for the paper
            
        Returns:
            Dictionary containing structured knowledge
        """
        if not self.openai_client:
            logger.error("OpenAI client not configured. Cannot extract knowledge.")
            return {}
            
        # Split text into manageable chunks
        chunks = self._split_text_into_chunks(text)
        
        # Process each chunk
        all_responses = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)} of paper {paper_id}")
            
            # Extract knowledge using OpenAI
            openai_knowledge = self._extract_with_openai(chunk, paper_id)
            
            # If configured, also extract with Claude and merge
            if self.use_multiple_models:
                anthropic_knowledge = self._extract_with_anthropic(chunk, paper_id)
                merged_knowledge = self._merge_extractions(openai_knowledge, anthropic_knowledge)
                all_responses.append(merged_knowledge)
            else:
                all_responses.append(openai_knowledge)
                
            # Add a short delay to avoid hitting API rate limits
            time.sleep(1)
            
        # Combine all chunk responses
        combined = self._combine_extraction_chunks(all_responses)
        
        # Add paper_id to all items
        for category in ["core_concepts", "research_questions", "methodologies", 
                         "key_findings", "measurements"]:
            if category in combined:
                for item in combined[category]:
                    if isinstance(item, dict) and "source" not in item:
                        item["source"] = paper_id
        
        return combined
    
    def _extract_with_openai(self, text: str, paper_id: str) -> Dict[str, Any]:
        """
        Extract knowledge using OpenAI API.
        
        Args:
            text: Text chunk from paper
            paper_id: Paper identifier
            
        Returns:
            Structured knowledge extracted by OpenAI
        """
        system_prompt = """
        You are an expert in knowledge extraction and domain modeling. Extract structured information 
        from the given scientific text to create a comprehensive knowledge representation.
        
        Extract the following elements:
        1. Core concepts and their definitions
        2. Domain-specific terminology with definitions
        3. Research questions and hypotheses
        4. Methodologies used
        5. Key findings and conclusions
        6. Statistical results and measurements
        
        Format your response as a JSON object with the following structure:
        {
            "core_concepts": [
                {"concept": "concept_name", "definition": "definition", "related_concepts": ["related1", "related2"]}
            ],
            "terminology": {
                "term1": "definition1",
                "term2": "definition2"
            },
            "research_questions": [
                {"question": "question text", "source": "paper_id"}
            ],
            "methodologies": [
                {"name": "methodology name", "description": "description", "applications": ["app1", "app2"]}
            ],
            "key_findings": [
                {"finding": "finding description", "source": "paper_id", "supports": ["concept1"]}
            ],
            "measurements": [
                {"name": "measurement name", "unit": "unit", "typical_range": "range", "significance": "why important"}
            ]
        }
        
        Follow these guidelines:
        - Be precise and detailed in your extraction
        - Only include information explicitly presented in the text
        - For each extracted item, try to identify relationships to other concepts
        - Maintain objectivity and accuracy
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Add paper_id to items that don't have a source
            for category in ["research_questions", "key_findings"]:
                if category in result:
                    for item in result[category]:
                        if "source" not in item:
                            item["source"] = paper_id
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting knowledge with OpenAI: {e}")
            return {}
    
    def _extract_with_anthropic(self, text: str, paper_id: str) -> Dict[str, Any]:
        """
        Extract knowledge using Anthropic API.
        
        Args:
            text: Text chunk from paper
            paper_id: Paper identifier
            
        Returns:
            Structured knowledge extracted by Claude
        """
        if not self.anthropic_client:
            return {}
            
        system_prompt = """
        You are an expert in knowledge extraction and domain modeling. Extract structured information 
        from the given scientific text to create a comprehensive knowledge representation.
        
        Always return valid JSON with the following structure:
        {
            "core_concepts": [
                {"concept": "concept_name", "definition": "definition", "related_concepts": ["related1", "related2"]}
            ],
            "terminology": {
                "term1": "definition1",
                "term2": "definition2"
            },
            "research_questions": [
                {"question": "question text", "source": "paper_id"}
            ],
            "methodologies": [
                {"name": "methodology name", "description": "description", "applications": ["app1", "app2"]}
            ],
            "key_findings": [
                {"finding": "finding description", "source": "paper_id", "supports": ["concept1"]}
            ],
            "measurements": [
                {"name": "measurement name", "unit": "unit", "typical_range": "range", "significance": "why important"}
            ]
        }
        """
        
        prompt = f"""
        Extract structured knowledge from the following scientific text:
        
        {text}
        
        Format your response as a JSON object with core_concepts, terminology, research_questions, 
        methodologies, key_findings, and measurements as described in the system instructions.
        """
        
        try:
            response = self.anthropic_client.messages.create(
                model=self.claude_model,
                system=system_prompt,
                max_tokens=3000,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            # Extract the JSON portion of the response
            content = response.content[0].text
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                result = json.loads(content[start_idx:end_idx])
                
                # Add paper_id to items that don't have a source
                for category in ["research_questions", "key_findings"]:
                    if category in result:
                        for item in result[category]:
                            if "source" not in item:
                                item["source"] = paper_id
                
                return result
            else:
                logger.warning("Could not find JSON in Claude response")
                return {}
                
        except Exception as e:
            logger.error(f"Error extracting knowledge with Claude: {e}")
            return {}
    
    def _merge_extractions(self, extraction1: Dict[str, Any], extraction2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two knowledge extractions, removing duplicates.
        
        Args:
            extraction1: First knowledge extraction
            extraction2: Second knowledge extraction
            
        Returns:
            Merged extraction
        """
        if not extraction1:
            return extraction2
        if not extraction2:
            return extraction1
            
        merged = {
            "core_concepts": [],
            "terminology": {},
            "research_questions": [],
            "methodologies": [],
            "key_findings": [],
            "measurements": []
        }
        
        # Merge core concepts
        concept_names = set()
        # Add from first extraction
        for concept in extraction1.get("core_concepts", []):
            concept_names.add(concept["concept"].lower())
            merged["core_concepts"].append(concept)
            
        # Add from second extraction if not duplicate
        for concept in extraction2.get("core_concepts", []):
            if concept["concept"].lower() not in concept_names:
                concept_names.add(concept["concept"].lower())
                merged["core_concepts"].append(concept)
        
        # Merge terminology
        merged["terminology"].update(extraction1.get("terminology", {}))
        merged["terminology"].update(extraction2.get("terminology", {}))
        
        # Helper function to merge list items with basic deduplication
        def merge_lists(list1, list2, key_field):
            items = list1.copy()
            existing_keys = {item[key_field].lower() for item in items}
            
            for item in list2:
                if item[key_field].lower() not in existing_keys:
                    items.append(item)
                    existing_keys.add(item[key_field].lower())
            
            return items
        
        # Merge other categories
        merged["research_questions"] = merge_lists(
            extraction1.get("research_questions", []),
            extraction2.get("research_questions", []),
            "question"
        )
        
        merged["methodologies"] = merge_lists(
            extraction1.get("methodologies", []),
            extraction2.get("methodologies", []),
            "name"
        )
        
        merged["key_findings"] = merge_lists(
            extraction1.get("key_findings", []),
            extraction2.get("key_findings", []),
            "finding"
        )
        
        merged["measurements"] = merge_lists(
            extraction1.get("measurements", []),
            extraction2.get("measurements", []),
            "name"
        )
        
        return merged
    
    def _combine_extraction_chunks(self, extractions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine multiple extraction chunks into a single knowledge structure.
        
        Args:
            extractions: List of extraction results from chunks
            
        Returns:
            Combined knowledge extraction
        """
        if not extractions:
            return {}
            
        if len(extractions) == 1:
            return extractions[0]
            
        # Start with the first extraction
        combined = extractions[0]
        
        # Merge with remaining extractions
        for extraction in extractions[1:]:
            combined = self._merge_extractions(combined, extraction)
            
        return combined
    
    def _combine_extractions(self, extractions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine knowledge from multiple papers.
        
        Args:
            extractions: List of knowledge extracted from different papers
            
        Returns:
            Combined knowledge map
        """
        if not extractions:
            return self.knowledge_map
            
        combined = {
            "core_concepts": [],
            "terminology": {},
            "research_questions": [],
            "methodologies": [],
            "key_findings": [],
            "measurements": []
        }
        
        # Track concepts, terms, etc. to avoid duplicates
        concept_names = set()
        methodology_names = set()
        question_texts = set()
        finding_texts = set()
        measurement_names = set()
        
        # Process each paper's extraction
        for extraction in extractions:
            # Add core concepts
            for concept in extraction.get("core_concepts", []):
                concept_key = concept["concept"].lower()
                if concept_key not in concept_names:
                    combined["core_concepts"].append(concept)
                    concept_names.add(concept_key)
            
            # Add terminology
            combined["terminology"].update(extraction.get("terminology", {}))
            
            # Add research questions
            for question in extraction.get("research_questions", []):
                question_key = question["question"].lower()
                if question_key not in question_texts:
                    combined["research_questions"].append(question)
                    question_texts.add(question_key)
            
            # Add methodologies
            for method in extraction.get("methodologies", []):
                method_key = method["name"].lower()
                if method_key not in methodology_names:
                    combined["methodologies"].append(method)
                    methodology_names.add(method_key)
            
            # Add key findings
            for finding in extraction.get("key_findings", []):
                finding_key = finding["finding"].lower()
                if finding_key not in finding_texts:
                    combined["key_findings"].append(finding)
                    finding_texts.add(finding_key)
            
            # Add measurements
            for measurement in extraction.get("measurements", []):
                measurement_key = measurement["name"].lower()
                if measurement_key not in measurement_names:
                    combined["measurements"].append(measurement)
                    measurement_names.add(measurement_key)
        
        return combined
    
    def _create_taxonomy(self, knowledge_map: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a hierarchical taxonomy of the domain from the knowledge map.
        
        Args:
            knowledge_map: The knowledge map to create taxonomy from
            
        Returns:
            Dictionary representing the domain taxonomy
        """
        if not self.openai_client:
            logger.error("OpenAI client not configured. Cannot create taxonomy.")
            return {}
            
        # Format the knowledge map for the prompt
        core_concepts_text = "\n".join([
            f"- {concept['concept']}: {concept['definition']}"
            for concept in knowledge_map.get("core_concepts", [])
        ])
        
        terminology_text = "\n".join([
            f"- {term}: {definition}"
            for term, definition in knowledge_map.get("terminology", {}).items()
        ])
        
        system_prompt = """
        You are an expert in knowledge organization and taxonomy creation. Given information about a domain,
        create a hierarchical taxonomy that organizes concepts into a coherent structure.
        
        Return your response as a JSON object with the following structure:
        {
            "domain": "name_of_domain",
            "top_level_categories": [
                {
                    "category": "category_name",
                    "subcategories": [
                        {
                            "subcategory": "subcategory_name",
                            "concepts": ["concept1", "concept2"]
                        }
                    ]
                }
            ],
            "relationships": [
                {"from": "concept_or_category", "relation": "relation_type", "to": "concept_or_category"}
            ]
        }
        
        Guidelines:
        - Create 3-6 top-level categories that organize the domain effectively
        - Include all major concepts from the input in appropriate subcategories
        - Identify hierarchical relationships (is-a, part-of, etc.)
        - Include cross-cutting relationships between concepts
        - Ensure the taxonomy is balanced and comprehensive
        """
        
        prompt = f"""
        Create a domain taxonomy based on the following knowledge:
        
        CORE CONCEPTS:
        {core_concepts_text}
        
        TERMINOLOGY:
        {terminology_text}
        
        Create a hierarchical taxonomy that organizes these concepts into a coherent structure.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            taxonomy = json.loads(response.choices[0].message.content)
            
            # Save the taxonomy
            taxonomy_path = self.output_dir / "taxonomy.json"
            with open(taxonomy_path, "w") as f:
                json.dump(taxonomy, f, indent=2)
                
            logger.info(f"Taxonomy created and saved to {taxonomy_path}")
            return taxonomy
            
        except Exception as e:
            logger.error(f"Error creating taxonomy: {e}")
            return {}
    
    def _create_concept_graph(self) -> None:
        """
        Create a graph representation of concepts and their relationships.
        Saves visualization if dependencies are available.
        """
        if not DEPS_AVAILABLE:
            logger.warning("NetworkX not available. Skipping concept graph creation.")
            return
            
        try:
            # Create a directed graph
            G = nx.DiGraph()
            
            # Add concepts as nodes
            for concept in self.knowledge_map.get("core_concepts", []):
                G.add_node(concept["concept"], type="concept", definition=concept["definition"])
                
                # Add relationships to related concepts
                for related in concept.get("related_concepts", []):
                    G.add_edge(concept["concept"], related, type="related")
            
            # Add findings and their relationships to concepts
            for i, finding in enumerate(self.knowledge_map.get("key_findings", [])):
                finding_id = f"finding_{i}"
                G.add_node(finding_id, type="finding", description=finding["finding"])
                
                # Connect findings to concepts they support
                for concept in finding.get("supports", []):
                    if G.has_node(concept):
                        G.add_edge(finding_id, concept, type="supports")
            
            # Save the graph
            self.concept_graph = G
            
            # Optional: Save visualization
            try:
                # Position nodes using spring layout
                pos = nx.spring_layout(G)
                
                # Define node colors based on type
                node_colors = []
                for node in G.nodes():
                    if G.nodes[node].get("type") == "concept":
                        node_colors.append("lightblue")
                    else:
                        node_colors.append("lightgreen")
                
                # Create visualization
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 10))
                nx.draw(G, pos, with_labels=True, node_color=node_colors, 
                        node_size=1500, arrows=True, font_size=8)
                
                # Save to file
                plt.savefig(self.output_dir / "concept_graph.png", dpi=300, bbox_inches="tight")
                plt.close()
                
                logger.info(f"Concept graph visualization saved to {self.output_dir / 'concept_graph.png'}")
            except Exception as e:
                logger.warning(f"Could not save graph visualization: {e}")
                
        except Exception as e:
            logger.error(f"Error creating concept graph: {e}")
    
    def _split_text_into_chunks(self, text: str, max_tokens: int = 4000) -> List[str]:
        """
        Split text into chunks of approximately max_tokens.
        
        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        # If we have a tokenizer, use it for token counting
        if self.tokenizer:
            return self._split_text_with_tokenizer(text, max_tokens)
        else:
            # Approximate tokens by characters (very rough approximation)
            # Assuming ~4 characters per token on average
            chars_per_token = 4
            max_chars = max_tokens * chars_per_token
            
            # Split into paragraphs
            paragraphs = text.split('\n\n')
            
            chunks = []
            current_chunk = []
            current_length = 0
            
            for paragraph in paragraphs:
                paragraph_length = len(paragraph)
                
                if current_length + paragraph_length > max_chars and current_chunk:
                    # Start a new chunk
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = [paragraph]
                    current_length = paragraph_length
                else:
                    # Add to current chunk
                    current_chunk.append(paragraph)
                    current_length += paragraph_length
            
            # Add the last chunk if it's not empty
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            
            return chunks
    
    def _split_text_with_tokenizer(self, text: str, max_tokens: int) -> List[str]:
        """
        Split text using tokenizer for accurate token counts.
        
        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List of text chunks
        """
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            # Count tokens in this paragraph
            tokens = len(self.tokenizer.encode(paragraph))
            
            if current_length + tokens > max_tokens and current_chunk:
                # Start a new chunk
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_length = tokens
            else:
                # Add to current chunk
                current_chunk.append(paragraph)
                current_length += tokens
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def get_knowledge_map(self) -> Dict[str, Any]:
        """
        Get the current knowledge map.
        
        Returns:
            The knowledge map
        """
        return self.knowledge_map
    
    def get_taxonomy(self) -> Dict[str, Any]:
        """
        Get the domain taxonomy.
        
        Returns:
            The taxonomy
        """
        return self.taxonomy 