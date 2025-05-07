"""
Entity Recognition and Linking Module

This module provides functionality to identify and extract named entities from text
and link them to standardized knowledge bases or ontologies.
"""

import logging
import os
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from pathlib import Path
import re
from collections import defaultdict

try:
    import spacy
    from spacy.tokens import Doc, Span
    import requests
    from transformers import (
        AutoTokenizer, 
        AutoModelForTokenClassification,
        pipeline
    )
    import numpy as np
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

logger = logging.getLogger(__name__)

class EntityRecognizer:
    """
    Entity recognition and linking system for knowledge extraction.
    Identifies domain-specific entities in text and links them to 
    knowledge bases or ontologies.
    """
    
    def __init__(
        self,
        domain: str = "scientific",
        use_spacy: bool = True,
        use_transformers: bool = True,
        use_custom_rules: bool = True,
        confidence_threshold: float = 0.5,
        cache_dir: Optional[str] = "cache/entity_recognition",
    ):
        """
        Initialize the entity recognizer.
        
        Args:
            domain: Domain for entity recognition (scientific, medical, etc.)
            use_spacy: Whether to use spaCy models
            use_transformers: Whether to use HuggingFace transformer models
            use_custom_rules: Whether to use custom rule-based entity extraction
            confidence_threshold: Minimum confidence for entity recognition
            cache_dir: Directory to cache entity recognition results
        """
        self.domain = domain
        self.use_spacy = use_spacy
        self.use_transformers = use_transformers
        self.use_custom_rules = use_custom_rules
        self.confidence_threshold = confidence_threshold
        
        # Create cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
            
        # Initialize NLP components if dependencies are available
        self.nlp = None
        self.ner_pipeline = None
        
        if not DEPS_AVAILABLE:
            logger.warning("Required dependencies not available. Install required packages.")
            return
        
        # Initialize spaCy models
        if use_spacy:
            try:
                if domain == "scientific":
                    # Try to load scientific model first
                    try:
                        self.nlp = spacy.load("en_core_sci_scibert")
                    except:
                        # Download and load scientific NER model if not found
                        os.system("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_scibert-0.5.0.tar.gz")
                        self.nlp = spacy.load("en_core_sci_scibert")
                elif domain == "medical":
                    try:
                        self.nlp = spacy.load("en_core_med7")
                    except:
                        # Fallback to standard model
                        self.nlp = spacy.load("en_core_web_lg")
                else:
                    # Default to standard English model
                    self.nlp = spacy.load("en_core_web_lg")
                    
                logger.info(f"Loaded spaCy model for {domain} domain")
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}")
                try:
                    # Fallback to basic model
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("Loaded fallback spaCy model")
                except:
                    logger.error("Could not load any spaCy model")
        
        # Initialize transformer models
        if use_transformers:
            try:
                model_name = "allenai/scibert_scivocab_uncased" if domain == "scientific" else "dslim/bert-base-NER"
                
                self.ner_pipeline = pipeline(
                    "ner",
                    model=model_name,
                    tokenizer=model_name,
                    aggregation_strategy="simple"
                )
                logger.info(f"Loaded transformers NER pipeline with {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load transformers pipeline: {e}")
        
        # Load custom entity patterns
        self.entity_patterns = self._load_entity_patterns()
        
        # Entity linking resources
        self.knowledge_bases = self._initialize_knowledge_bases()
        
    def _load_entity_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load custom entity patterns for different entity types."""
        patterns = {
            "METHOD": [
                {"pattern": [{"LOWER": "convolutional"}, {"LOWER": "neural"}, {"LOWER": "network"}], "label": "METHOD"},
                {"pattern": [{"LOWER": "random"}, {"LOWER": "forest"}], "label": "METHOD"},
                {"pattern": [{"LOWER": "bert"}], "label": "METHOD"},
                {"pattern": [{"LOWER": "transformer"}, {"POS": "NOUN"}], "label": "METHOD"},
            ],
            "METRIC": [
                {"pattern": [{"LOWER": "accuracy"}], "label": "METRIC"},
                {"pattern": [{"LOWER": "f1"}, {"LOWER": "score"}], "label": "METRIC"},
                {"pattern": [{"LOWER": "precision"}], "label": "METRIC"},
                {"pattern": [{"LOWER": "recall"}], "label": "METRIC"},
                {"pattern": [{"LOWER": "mean"}, {"LOWER": "average"}, {"LOWER": "precision"}], "label": "METRIC"},
            ],
            "PARAMETER": [
                {"pattern": [{"LOWER": "learning"}, {"LOWER": "rate"}], "label": "PARAMETER"},
                {"pattern": [{"LOWER": "batch"}, {"LOWER": "size"}], "label": "PARAMETER"},
                {"pattern": [{"LOWER": "momentum"}], "label": "PARAMETER"},
                {"pattern": [{"LOWER": "dropout"}], "label": "PARAMETER"},
            ]
        }
        return patterns
        
    def _initialize_knowledge_bases(self) -> Dict[str, Any]:
        """Initialize connections to external knowledge bases."""
        # In practice, this would connect to knowledge bases like UMLS, Wikidata, etc.
        # For now, we'll just return placeholders
        return {
            "scientific": {
                "url": "https://api.wikidata.org/wiki/",
                "cache": {}
            },
            "medical": {
                "url": "https://umlsks.nlm.nih.gov/kss",
                "cache": {}
            }
        }
    
    def recognize_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Recognize entities in text using multiple methods and combine results.
        
        Args:
            text: The text to process
            
        Returns:
            List of dictionaries containing entity information:
            {
                "text": entity text,
                "type": entity type,
                "start": start position,
                "end": end position,
                "confidence": confidence score,
                "method": method used for extraction,
                "links": linked entities in knowledge bases
            }
        """
        if not text or not DEPS_AVAILABLE:
            return []
            
        entities = []
        
        # Extract with spaCy if available
        if self.nlp and self.use_spacy:
            spacy_entities = self._extract_with_spacy(text)
            entities.extend(spacy_entities)
            
        # Extract with transformers if available
        if self.ner_pipeline and self.use_transformers:
            transformer_entities = self._extract_with_transformers(text)
            entities.extend(transformer_entities)
            
        # Apply custom rules if enabled
        if self.use_custom_rules:
            custom_entities = self._extract_with_custom_rules(text)
            entities.extend(custom_entities)
            
        # Merge overlapping entities and resolve conflicts
        merged_entities = self._merge_entities(entities)
        
        # Link entities to knowledge bases
        linked_entities = self._link_entities(merged_entities)
        
        # Filter by confidence threshold
        filtered_entities = [
            e for e in linked_entities 
            if e.get("confidence", 0) >= self.confidence_threshold
        ]
        
        return filtered_entities
    
    def _extract_with_spacy(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using spaCy."""
        entities = []
        
        doc = self.nlp(text)
        
        # Extract standard named entities
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": 0.8,  # SpaCy doesn't provide confidence scores, use default
                "method": "spacy"
            })
            
        # Add noun chunks as potential entities
        for chunk in doc.noun_chunks:
            # Skip single pronouns and determiners
            if chunk.root.pos_ in ("PRON", "DET") or len(chunk) <= 1:
                continue
                
            # Skip if already covered by a named entity
            if any(
                chunk.start_char >= ent["start"] and chunk.end_char <= ent["end"]
                for ent in entities
            ):
                continue
                
            entities.append({
                "text": chunk.text,
                "type": "CONCEPT" if chunk.root.pos_ == "NOUN" else "TERM",
                "start": chunk.start_char,
                "end": chunk.end_char,
                "confidence": 0.6,
                "method": "spacy_chunk"
            })
            
        return entities
    
    def _extract_with_transformers(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using transformer models."""
        entities = []
        
        try:
            # Process with transformer pipeline
            results = self.ner_pipeline(text)
            
            for ent in results:
                entities.append({
                    "text": ent["word"],
                    "type": ent["entity_group"],
                    "start": ent["start"],
                    "end": ent["end"],
                    "confidence": ent["score"],
                    "method": "transformers"
                })
        except Exception as e:
            logger.error(f"Error in transformer entity extraction: {e}")
            
        return entities
    
    def _extract_with_custom_rules(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using custom rules and patterns."""
        entities = []
        
        # Apply custom patterns if spaCy is available
        if self.nlp:
            # Add pattern matcher if not already in pipeline
            if "entity_ruler" not in self.nlp.pipe_names:
                ruler = self.nlp.add_pipe("entity_ruler", before="ner")
                
                # Add patterns
                patterns = []
                for entity_type, pattern_list in self.entity_patterns.items():
                    patterns.extend(pattern_list)
                    
                ruler.add_patterns(patterns)
            
            # Process with pattern matcher
            doc = self.nlp(text)
            
            # Extract pattern-matched entities
            for ent in doc.ents:
                # Only consider entities from our patterns
                if ent.label_ in self.entity_patterns:
                    entities.append({
                        "text": ent.text,
                        "type": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "confidence": 0.9,  # High confidence for pattern matches
                        "method": "pattern"
                    })
        
        # Apply regex-based extraction for specific entity types
        entities.extend(self._regex_extraction(text))
        
        return entities
    
    def _regex_extraction(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using regex patterns."""
        entities = []
        
        # Extract version numbers (e.g., v1.2.3)
        version_pattern = r'\bv\d+(\.\d+)+\b'
        for match in re.finditer(version_pattern, text):
            entities.append({
                "text": match.group(0),
                "type": "VERSION",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.95,
                "method": "regex"
            })
        
        # Extract percentages
        percentage_pattern = r'\b\d+(\.\d+)?%\b'
        for match in re.finditer(percentage_pattern, text):
            entities.append({
                "text": match.group(0),
                "type": "PERCENTAGE",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.95,
                "method": "regex"
            })
            
        # Extract URLs
        url_pattern = r'https?://[^\s]+'
        for match in re.finditer(url_pattern, text):
            entities.append({
                "text": match.group(0),
                "type": "URL",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.98,
                "method": "regex"
            })
            
        return entities
    
    def _merge_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge overlapping entities and resolve conflicts.
        
        Strategy:
        - Keep the entity with higher confidence if there's overlap
        - Merge entities of same type that are adjacent
        """
        if not entities:
            return []
            
        # Sort by start position and then by confidence
        sorted_entities = sorted(
            entities, 
            key=lambda e: (e["start"], -e.get("confidence", 0))
        )
        
        merged = []
        current = sorted_entities[0]
        
        for entity in sorted_entities[1:]:
            # Check for overlap
            if entity["start"] < current["end"]:
                # If overlapping entity has higher confidence, replace current
                if entity.get("confidence", 0) > current.get("confidence", 0):
                    current = entity
                # Otherwise, keep current entity
            else:
                # No overlap, add current to results and move to next
                merged.append(current)
                current = entity
                
        # Add the last entity
        merged.append(current)
        
        return merged
    
    def _link_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Link extracted entities to external knowledge bases.
        
        Args:
            entities: List of extracted entities
            
        Returns:
            Entities with added linking information
        """
        linked_entities = []
        
        for entity in entities:
            entity_text = entity["text"]
            entity_type = entity["type"]
            
            # Initialize links field
            entity["links"] = []
            
            # Get appropriate knowledge base based on domain and entity type
            kb = self.knowledge_bases.get(self.domain)
            if not kb:
                linked_entities.append(entity)
                continue
                
            # Check cache first
            cache_key = f"{entity_text}:{entity_type}"
            if cache_key in kb["cache"]:
                entity["links"] = kb["cache"][cache_key]
                linked_entities.append(entity)
                continue
                
            # Basic mock linking logic (in a real system, this would call APIs)
            # This is just a placeholder for demonstration
            if entity_type == "PERSON":
                entity["links"].append({
                    "kb": "wikidata",
                    "id": f"Q{hash(entity_text) % 1000000}",
                    "confidence": 0.8
                })
            elif entity_type in ("CONCEPT", "TERM", "METHOD"):
                entity["links"].append({
                    "kb": "domain_ontology",
                    "id": f"T{hash(entity_text) % 1000000}",
                    "confidence": 0.7
                })
                
            # Cache the result
            kb["cache"][cache_key] = entity["links"]
            linked_entities.append(entity)
            
        return linked_entities
    
    def batch_process(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Process multiple texts and extract entities from all of them.
        
        Args:
            texts: List of texts to process
            
        Returns:
            List of entity lists, one for each input text
        """
        results = []
        for text in texts:
            entities = self.recognize_entities(text)
            results.append(entities)
        return results
    
    def annotate_text(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Return the original text and entity annotations for visualization.
        
        Args:
            text: Text to annotate
            
        Returns:
            Tuple of (text, entities)
        """
        entities = self.recognize_entities(text)
        return text, entities 