"""
Enhanced Knowledge Extraction Pipeline Module

This module implements advanced techniques for extracting knowledge from research papers
with improved accuracy, including:
- NLP-based entity recognition and extraction
- Accuracy validation through cross-validation
- Confidence scoring for extracted knowledge
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from collections import defaultdict

# Third-party imports
try:
    import openai
    import anthropic
    import numpy as np
    from transformers import (
        AutoTokenizer, 
        AutoModelForTokenClassification,
        pipeline
    )
    import spacy
    from sklearn.feature_extraction.text import TfidfVectorizer
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

from main.knowledge.knowledge_map import KnowledgeMap

logger = logging.getLogger(__name__)

class EnhancedKnowledgeExtractor:
    """
    Enhanced knowledge extraction system that improves accuracy through:
    - Entity recognition and linking
    - Cross-verification between multiple extraction methods
    - Confidence scoring for extracted information
    """
    
    def __init__(
        self, 
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        openai_model: str = "gpt-4",
        claude_model: str = "claude-3-opus-20240229",
        output_dir: str = "output/enhanced_extraction",
        use_spacy: bool = True,
        use_transformers: bool = True,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize the enhanced knowledge extractor.
        
        Args:
            openai_api_key: OpenAI API key (if None, tries to get from environment)
            anthropic_api_key: Anthropic API key (if None, tries to get from environment)
            openai_model: OpenAI model to use for extraction
            claude_model: Claude model to use for validation
            output_dir: Directory to save extracted knowledge
            use_spacy: Whether to use spaCy for NER
            use_transformers: Whether to use transformers for NER
            confidence_threshold: Minimum confidence score to include extracted knowledge
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
        
        # Initialize Anthropic client
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.anthropic_client = None
        if self.anthropic_api_key:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
                self.claude_model = claude_model
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic client: {e}")
        
        self.confidence_threshold = confidence_threshold
        
        # Initialize NLP tools if dependencies are available
        self.nlp = None
        self.ner_pipeline = None
        
        if DEPS_AVAILABLE:
            if use_spacy:
                try:
                    # Load spaCy model with scientific entity recognition
                    self.nlp = spacy.load("en_core_sci_scibert")
                    logger.info("Loaded spaCy scientific NER model")
                except Exception:
                    try:
                        # Fallback to standard English model
                        self.nlp = spacy.load("en_core_web_lg")
                        logger.info("Loaded spaCy standard NER model")
                    except Exception as e:
                        logger.warning(f"Failed to load spaCy model: {e}")
                        
            if use_transformers:
                try:
                    # Initialize transformers NER pipeline
                    self.ner_pipeline = pipeline(
                        "ner",
                        model="allenai/scibert_scivocab_uncased",
                        tokenizer="allenai/scibert_scivocab_uncased"
                    )
                    logger.info("Loaded transformers NER pipeline")
                except Exception as e:
                    logger.warning(f"Failed to load transformers pipeline: {e}")
        
        # Initialize TF-IDF vectorizer for content similarity
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english'
        ) if DEPS_AVAILABLE else None
        
        # Initialize the basic knowledge map for comparison
        self.knowledge_map = KnowledgeMap(
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            output_dir=output_dir
        )
        
        # Knowledge storage for extracted information
        self.extracted_entities = defaultdict(list)
        self.extracted_relations = defaultdict(list)
        self.concept_confidences = {}
        
    def extract_knowledge_from_paper(self, paper_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract knowledge from a single paper with enhanced accuracy.
        
        Args:
            paper_path: Path to the paper file (PDF or TXT)
            
        Returns:
            Dictionary containing extracted knowledge with confidence scores
        """
        path = Path(paper_path)
        if not path.exists():
            raise FileNotFoundError(f"Paper not found: {paper_path}")
            
        logger.info(f"Extracting knowledge from: {path.name}")
        
        # Read paper content
        text_content = self._read_paper_content(path)
        if not text_content:
            logger.warning(f"Failed to extract text from {path.name}")
            return {}
        
        # Extract knowledge using multiple methods for cross-validation
        llm_extraction = self._extract_with_llms(text_content, path.stem)
        entity_extraction = self._extract_entities(text_content)
        relation_extraction = self._extract_relations(text_content)
        
        # Combine and validate extractions
        combined_knowledge = self._combine_and_validate(
            llm_extraction,
            entity_extraction,
            relation_extraction
        )
        
        # Filter out low-confidence extractions
        validated_knowledge = self._filter_by_confidence(combined_knowledge)
        
        # Save extraction results
        self._save_extraction(validated_knowledge, path.stem)
        
        return validated_knowledge
    
    def _read_paper_content(self, paper_path: Path) -> str:
        """Read content from a paper file (PDF or text)."""
        if paper_path.suffix.lower() == '.pdf':
            try:
                import PyPDF2
                with open(paper_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    return text
            except Exception as e:
                logger.error(f"Error reading PDF: {e}")
                return ""
        else:
            # Assume it's a text file
            try:
                with open(paper_path, 'r', encoding='utf-8') as file:
                    return file.read()
            except Exception as e:
                logger.error(f"Error reading text file: {e}")
                return ""
    
    def _extract_with_llms(self, text: str, paper_id: str) -> Dict[str, Any]:
        """Extract knowledge using LLMs (OpenAI and/or Anthropic)."""
        extractions = []
        
        # Split text into chunks for processing
        chunks = self._split_text_into_chunks(text)
        
        # Extract with OpenAI if available
        if self.openai_client:
            for i, chunk in enumerate(chunks):
                try:
                    extraction = self._extract_with_openai(chunk, f"{paper_id}_chunk{i}")
                    if extraction:
                        extractions.append(extraction)
                except Exception as e:
                    logger.error(f"Error extracting with OpenAI: {e}")
        
        # Extract with Anthropic if available
        if self.anthropic_client:
            for i, chunk in enumerate(chunks):
                try:
                    extraction = self._extract_with_anthropic(chunk, f"{paper_id}_chunk{i}")
                    if extraction:
                        extractions.append(extraction)
                except Exception as e:
                    logger.error(f"Error extracting with Anthropic: {e}")
        
        # Combine chunk extractions
        if not extractions:
            return {}
            
        combined = self._combine_chunk_extractions(extractions)
        return combined
    
    def _extract_with_openai(self, text: str, chunk_id: str) -> Dict[str, Any]:
        """Extract knowledge from text using OpenAI."""
        prompt = f"""
        Extract key knowledge from this scientific text. Include:
        1. Core concepts and terminology with definitions
        2. Research questions and hypotheses
        3. Methodologies and approaches
        4. Key findings and results
        5. Measurements and statistical data
        
        Text:
        {text}
        
        Format the output as a structured JSON with the following keys:
        "core_concepts", "terminology", "research_questions", "methodologies", "key_findings", "measurements"
        
        For each extracted item, include a confidence score from 0.0 to 1.0 based on how clearly it is stated in the text.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are an expert research assistant that extracts scientific knowledge with high accuracy."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            extraction = json.loads(content)
            extraction["source"] = "openai"
            extraction["chunk_id"] = chunk_id
            return extraction
        except Exception as e:
            logger.error(f"OpenAI extraction error: {e}")
            return {}
    
    def _extract_with_anthropic(self, text: str, chunk_id: str) -> Dict[str, Any]:
        """Extract knowledge from text using Anthropic Claude."""
        prompt = f"""
        Extract key knowledge from this scientific text. Include:
        1. Core concepts and terminology with definitions
        2. Research questions and hypotheses
        3. Methodologies and approaches
        4. Key findings and results
        5. Measurements and statistical data
        
        Text:
        {text}
        
        Format the output as a structured JSON with the following keys:
        "core_concepts", "terminology", "research_questions", "methodologies", "key_findings", "measurements"
        
        For each extracted item, include a confidence score from 0.0 to 1.0 based on how clearly it is stated in the text.
        """
        
        try:
            response = self.anthropic_client.messages.create(
                model=self.claude_model,
                max_tokens=4000,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                system="You are an expert research assistant that extracts scientific knowledge with high accuracy."
            )
            
            # Parse JSON from Claude's response
            content = response.content[0].text
            match = re.search(r'{.*}', content, re.DOTALL)
            if not match:
                logger.error("Failed to parse JSON from Claude response")
                return {}
                
            json_str = match.group(0)
            extraction = json.loads(json_str)
            extraction["source"] = "anthropic"
            extraction["chunk_id"] = chunk_id
            return extraction
        except Exception as e:
            logger.error(f"Anthropic extraction error: {e}")
            return {}
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract named entities using NLP tools."""
        entities = {
            "scientific_entities": [],
            "methods": [],
            "metrics": []
        }
        
        if not DEPS_AVAILABLE:
            return entities
            
        # Extract using spaCy if available
        if self.nlp:
            try:
                doc = self.nlp(text[:100000])  # Limit text size to avoid OOM
                for ent in doc.ents:
                    entity = {
                        "text": ent.text,
                        "label": ent.label_,
                        "confidence": 0.8,  # Default confidence for spaCy
                        "source": "spacy"
                    }
                    
                    if ent.label_ in ("METHOD", "PROCESS"):
                        entities["methods"].append(entity)
                    elif ent.label_ in ("METRIC", "QUANTITY", "MEASUREMENT"):
                        entities["metrics"].append(entity)
                    else:
                        entities["scientific_entities"].append(entity)
            except Exception as e:
                logger.error(f"spaCy entity extraction error: {e}")
        
        # Extract using transformers if available
        if self.ner_pipeline:
            try:
                # Process text in chunks to avoid OOM
                chunks = [text[i:i+1000] for i in range(0, min(len(text), 50000), 1000)]
                for chunk in chunks:
                    results = self.ner_pipeline(chunk)
                    
                    # Group by entity
                    current_entity = ""
                    current_label = ""
                    current_score = 0.0
                    
                    for result in results:
                        if result["entity"].startswith("B-"):
                            # Save previous entity if exists
                            if current_entity:
                                entity = {
                                    "text": current_entity.strip(),
                                    "label": current_label,
                                    "confidence": current_score,
                                    "source": "transformers"
                                }
                                entities["scientific_entities"].append(entity)
                            
                            # Start new entity
                            current_entity = result["word"]
                            current_label = result["entity"][2:]  # Remove B- prefix
                            current_score = result["score"]
                        elif result["entity"].startswith("I-"):
                            # Continue current entity
                            current_entity += " " + result["word"]
                            current_score = (current_score + result["score"]) / 2  # Average confidence
                            
                    # Add the last entity
                    if current_entity:
                        entity = {
                            "text": current_entity.strip(),
                            "label": current_label,
                            "confidence": current_score,
                            "source": "transformers"
                        }
                        entities["scientific_entities"].append(entity)
            except Exception as e:
                logger.error(f"Transformers entity extraction error: {e}")
        
        return entities
    
    def _extract_relations(self, text: str) -> List[Dict[str, Any]]:
        """Extract relationships between entities."""
        relations = []
        
        if not self.openai_client:
            return relations
            
        # Extract relations using OpenAI
        prompt = """
        Extract important relationships between scientific concepts in this text.
        For each relationship, identify:
        1. The source entity
        2. The target entity
        3. The type of relationship (e.g., causes, influences, correlates with)
        4. Confidence level (0.0-1.0)
        
        Return the results as a JSON array of objects with the fields:
        "source", "target", "relation_type", "confidence", "evidence"
        """
        
        chunks = self._split_text_into_chunks(text, max_tokens=2000)
        
        for chunk in chunks[:5]:  # Limit to first 5 chunks to avoid excessive API usage
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": "Extract scientific relationships from the text."},
                        {"role": "user", "content": prompt + "\n\nText:\n" + chunk}
                    ],
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                result = json.loads(content)
                
                if "relationships" in result:
                    relations.extend(result["relationships"])
                elif "relations" in result:
                    relations.extend(result["relations"])
            except Exception as e:
                logger.error(f"Relation extraction error: {e}")
        
        return relations
    
    def _combine_and_validate(
        self,
        llm_extraction: Dict[str, Any],
        entity_extraction: Dict[str, Any],
        relation_extraction: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Combine and cross-validate extractions from different methods."""
        combined = {
            "core_concepts": [],
            "terminology": {},
            "research_questions": [],
            "methodologies": [],
            "key_findings": [],
            "measurements": [],
            "entities": {},
            "relations": []
        }
        
        # Start with LLM extraction if available
        if llm_extraction:
            for key in ["core_concepts", "research_questions", "methodologies", "key_findings", "measurements"]:
                if key in llm_extraction:
                    combined[key] = llm_extraction[key]
            
            if "terminology" in llm_extraction:
                combined["terminology"] = llm_extraction["terminology"]
        
        # Add entities from entity extraction
        if entity_extraction:
            combined["entities"] = entity_extraction
            
            # Cross-check with LLM extraction to improve confidence
            if "scientific_entities" in entity_extraction:
                for entity in entity_extraction["scientific_entities"]:
                    # Check if entity exists in core concepts
                    for concept in combined["core_concepts"]:
                        if isinstance(concept, dict) and "text" in concept:
                            concept_text = concept["text"]
                        else:
                            concept_text = str(concept)
                            
                        if entity["text"].lower() in concept_text.lower():
                            # Increase confidence if found in both methods
                            if isinstance(concept, dict) and "confidence" in concept:
                                concept["confidence"] = min(1.0, concept["confidence"] + 0.1)
                                
            # Add methods to methodologies
            if "methods" in entity_extraction:
                for method in entity_extraction["methods"]:
                    found = False
                    for existing_method in combined["methodologies"]:
                        if isinstance(existing_method, dict) and "text" in existing_method:
                            method_text = existing_method["text"]
                        else:
                            method_text = str(existing_method)
                            
                        if method["text"].lower() in method_text.lower():
                            found = True
                            break
                            
                    if not found:
                        combined["methodologies"].append({
                            "text": method["text"],
                            "confidence": method["confidence"],
                            "source": method["source"]
                        })
                        
            # Add metrics to measurements
            if "metrics" in entity_extraction:
                for metric in entity_extraction["metrics"]:
                    found = False
                    for existing_metric in combined["measurements"]:
                        if isinstance(existing_metric, dict) and "text" in existing_metric:
                            metric_text = existing_metric["text"]
                        else:
                            metric_text = str(existing_metric)
                            
                        if metric["text"].lower() in metric_text.lower():
                            found = True
                            break
                            
                    if not found:
                        combined["measurements"].append({
                            "text": metric["text"],
                            "confidence": metric["confidence"],
                            "source": metric["source"]
                        })
        
        # Add relations
        if relation_extraction:
            combined["relations"] = relation_extraction
            
            # Validate relations against core concepts to improve confidence
            for relation in combined["relations"]:
                source_found = False
                target_found = False
                
                for concept in combined["core_concepts"]:
                    if isinstance(concept, dict) and "text" in concept:
                        concept_text = concept["text"]
                    else:
                        concept_text = str(concept)
                        
                    if relation["source"].lower() in concept_text.lower():
                        source_found = True
                    if relation["target"].lower() in concept_text.lower():
                        target_found = True
                
                # Adjust confidence based on concept validation
                if source_found and target_found:
                    relation["confidence"] = min(1.0, relation["confidence"] + 0.1)
                elif not source_found and not target_found:
                    relation["confidence"] = max(0.0, relation["confidence"] - 0.2)
        
        return combined
    
    def _filter_by_confidence(self, knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out low-confidence extractions."""
        filtered = {}
        
        for key, value in knowledge.items():
            if isinstance(value, list):
                filtered[key] = []
                for item in value:
                    if isinstance(item, dict) and "confidence" in item:
                        if item["confidence"] >= self.confidence_threshold:
                            filtered[key].append(item)
                    else:
                        filtered[key].append(item)
            elif isinstance(value, dict):
                filtered[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, list):
                        filtered[key][sub_key] = []
                        for item in sub_value:
                            if isinstance(item, dict) and "confidence" in item:
                                if item["confidence"] >= self.confidence_threshold:
                                    filtered[key][sub_key].append(item)
                            else:
                                filtered[key][sub_key].append(item)
                    else:
                        filtered[key][sub_key] = sub_value
            else:
                filtered[key] = value
        
        return filtered
    
    def _save_extraction(self, extraction: Dict[str, Any], paper_id: str) -> None:
        """Save the extraction results to a file."""
        output_file = self.output_dir / f"enhanced_{paper_id}.json"
        with open(output_file, "w") as f:
            json.dump(extraction, f, indent=2)
            
        logger.info(f"Saved enhanced extraction to {output_file}")
    
    def _split_text_into_chunks(self, text: str, max_tokens: int = 4000) -> List[str]:
        """Split text into chunks for processing."""
        if len(text) < max_tokens:
            return [text]
            
        # Naive splitting by paragraphs
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) < max_tokens:
                current_chunk += paragraph + "\n\n"
            else:
                chunks.append(current_chunk)
                current_chunk = paragraph + "\n\n"
                
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def _combine_chunk_extractions(self, extractions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine extractions from multiple chunks."""
        if not extractions:
            return {}
            
        combined = {
            "core_concepts": [],
            "terminology": {},
            "research_questions": [],
            "methodologies": [],
            "key_findings": [],
            "measurements": []
        }
        
        for extraction in extractions:
            # Combine lists
            for key in ["core_concepts", "research_questions", "methodologies", "key_findings", "measurements"]:
                if key in extraction and extraction[key]:
                    combined[key].extend(extraction[key])
            
            # Merge terminology dictionaries
            if "terminology" in extraction and extraction["terminology"]:
                for term, definition in extraction["terminology"].items():
                    if term not in combined["terminology"]:
                        combined["terminology"][term] = definition
        
        # Remove duplicates while preserving confidence scores
        for key in ["core_concepts", "research_questions", "methodologies", "key_findings", "measurements"]:
            if isinstance(combined[key], list):
                unique_items = {}
                
                for item in combined[key]:
                    if isinstance(item, dict) and "text" in item:
                        item_text = item["text"]
                        if item_text in unique_items:
                            # Keep the item with higher confidence
                            if "confidence" in item and "confidence" in unique_items[item_text]:
                                if item["confidence"] > unique_items[item_text]["confidence"]:
                                    unique_items[item_text] = item
                        else:
                            unique_items[item_text] = item
                    else:
                        item_text = str(item)
                        if item_text not in unique_items:
                            unique_items[item_text] = item
                
                combined[key] = list(unique_items.values())
        
        return combined
    
    def process_papers(self, papers_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Process all papers in a directory with enhanced extraction.
        
        Args:
            papers_dir: Directory containing papers to process
            
        Returns:
            Combined extracted knowledge with confidence scores
        """
        papers_dir = Path(papers_dir)
        if not papers_dir.exists() or not papers_dir.is_dir():
            raise ValueError(f"Invalid papers directory: {papers_dir}")
            
        # Find paper files
        paper_files = list(papers_dir.glob("*.pdf")) + list(papers_dir.glob("*.txt"))
        logger.info(f"Found {len(paper_files)} papers in {papers_dir}")
        
        if not paper_files:
            logger.warning(f"No papers found in {papers_dir}")
            return {}
        
        # Process each paper
        all_extractions = []
        for paper_path in paper_files:
            try:
                extracted = self.extract_knowledge_from_paper(paper_path)
                if extracted:
                    all_extractions.append(extracted)
            except Exception as e:
                logger.error(f"Error processing {paper_path.name}: {str(e)}")
                
        # Combine all extractions
        combined = self._combine_all_extractions(all_extractions)
        
        # Save combined extraction
        with open(self.output_dir / "enhanced_knowledge_extraction.json", "w") as f:
            json.dump(combined, f, indent=2)
            
        logger.info(f"Enhanced extraction complete. Processed {len(paper_files)} papers.")
        return combined
    
    def _combine_all_extractions(self, extractions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine all extractions into a single knowledge representation."""
        if not extractions:
            return {}
            
        combined = {
            "core_concepts": [],
            "terminology": {},
            "research_questions": [],
            "methodologies": [],
            "key_findings": [],
            "measurements": [],
            "entities": {
                "scientific_entities": [],
                "methods": [],
                "metrics": []
            },
            "relations": []
        }
        
        # Track item texts to avoid duplicates
        seen_items = {
            "core_concepts": set(),
            "research_questions": set(),
            "methodologies": set(),
            "key_findings": set(),
            "measurements": set(),
            "scientific_entities": set(),
            "methods": set(),
            "metrics": set(),
            "relations": set()
        }
        
        for extraction in extractions:
            # Combine lists with deduplication
            for key in ["core_concepts", "research_questions", "methodologies", "key_findings", "measurements"]:
                if key in extraction:
                    for item in extraction[key]:
                        if isinstance(item, dict) and "text" in item:
                            item_text = item["text"]
                        else:
                            item_text = str(item)
                            
                        if item_text not in seen_items[key]:
                            seen_items[key].add(item_text)
                            combined[key].append(item)
            
            # Merge terminology
            if "terminology" in extraction:
                for term, definition in extraction["terminology"].items():
                    if term not in combined["terminology"]:
                        combined["terminology"][term] = definition
            
            # Combine entities
            if "entities" in extraction:
                for entity_type in ["scientific_entities", "methods", "metrics"]:
                    if entity_type in extraction["entities"]:
                        for entity in extraction["entities"][entity_type]:
                            entity_text = entity["text"]
                            if entity_text not in seen_items[entity_type]:
                                seen_items[entity_type].add(entity_text)
                                combined["entities"][entity_type].append(entity)
            
            # Combine relations with deduplication
            if "relations" in extraction:
                for relation in extraction["relations"]:
                    relation_key = f"{relation['source']}|{relation['relation_type']}|{relation['target']}"
                    if relation_key not in seen_items["relations"]:
                        seen_items["relations"].add(relation_key)
                        combined["relations"].append(relation)
        
        return combined 