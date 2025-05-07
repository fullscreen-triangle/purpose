"""
Incremental Knowledge Update Module

This module implements functionality for incrementally updating knowledge
without requiring full reprocessing of all source materials.
"""

import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from collections import defaultdict

# Third-party imports
try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

logger = logging.getLogger(__name__)

class IncrementalKnowledgeUpdater:
    """
    System for incrementally updating knowledge from new sources
    without full reprocessing of existing knowledge.
    """
    
    def __init__(
        self, 
        output_dir: str = "output/incremental",
        knowledge_base_path: Optional[str] = None,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize the incremental knowledge updater.
        
        Args:
            output_dir: Directory to save updated knowledge
            knowledge_base_path: Path to existing knowledge base (optional)
            similarity_threshold: Threshold for determining similar concepts
        """
        # Initialize outputs directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set parameters
        self.similarity_threshold = similarity_threshold
        
        # Initialize knowledge base
        self.knowledge_base = {}
        self.knowledge_history = []
        self.last_update_time = None
        
        # Load existing knowledge base if provided
        if knowledge_base_path:
            self.load_knowledge_base(knowledge_base_path)
            
        # Initialize sentence encoder for semantic similarity
        self.sentence_encoder = None
        if DEPS_AVAILABLE:
            try:
                self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Initialized sentence encoder for semantic similarity")
            except Exception as e:
                logger.warning(f"Failed to initialize sentence encoder: {e}")
                
        # Initialize concept embeddings
        self.concept_embeddings = {}
        if self.knowledge_base and self.sentence_encoder:
            self._create_concept_embeddings()
    
    def load_knowledge_base(self, path: Union[str, Path]) -> bool:
        """
        Load an existing knowledge base.
        
        Args:
            path: Path to knowledge base file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            # Check if the file has a specific format
            if isinstance(data, dict) and "knowledge" in data:
                self.knowledge_base = data["knowledge"]
                if "history" in data:
                    self.knowledge_history = data["history"]
                if "last_update" in data:
                    self.last_update_time = data["last_update"]
            else:
                # Assume the entire file is the knowledge base
                self.knowledge_base = data
                
            logger.info(f"Loaded knowledge base from {path}")
            
            # Create version history if not present
            if not self.knowledge_history:
                self.knowledge_history = [{
                    "version": 1,
                    "timestamp": datetime.now().isoformat(),
                    "description": "Initial version",
                    "source": f"Loaded from {path}"
                }]
                
            return True
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            return False
    
    def _create_concept_embeddings(self) -> None:
        """Create embeddings for concepts in the knowledge base."""
        if not self.sentence_encoder or not self.knowledge_base:
            return
            
        try:
            concepts = []
            
            # Extract concepts from core_concepts
            if "core_concepts" in self.knowledge_base:
                for concept in self.knowledge_base["core_concepts"]:
                    if isinstance(concept, dict) and "text" in concept:
                        concepts.append(concept["text"])
                    elif isinstance(concept, str):
                        concepts.append(concept)
            
            # Extract concepts from terminology
            if "terminology" in self.knowledge_base:
                for term in self.knowledge_base["terminology"]:
                    concepts.append(term)
            
            # Create embeddings
            logger.info(f"Creating embeddings for {len(concepts)} concepts")
            
            # Process in batches to avoid memory issues
            batch_size = 32
            for i in range(0, len(concepts), batch_size):
                batch = concepts[i:i + batch_size]
                embeddings = self.sentence_encoder.encode(batch)
                
                for j, concept in enumerate(batch):
                    self.concept_embeddings[concept] = embeddings[j]
                    
            logger.info(f"Created embeddings for {len(self.concept_embeddings)} concepts")
            
        except Exception as e:
            logger.error(f"Error creating concept embeddings: {e}")
    
    def update_with_new_knowledge(
        self, 
        new_knowledge: Dict[str, Any],
        source: str = "new_extraction",
        description: str = "Incremental update"
    ) -> Dict[str, Any]:
        """
        Update the knowledge base with new knowledge.
        
        Args:
            new_knowledge: New knowledge to incorporate
            source: Source of the new knowledge
            description: Description of the update
            
        Returns:
            Updated knowledge base
        """
        logger.info(f"Updating knowledge base with new knowledge from {source}")
        
        # If knowledge base is empty, just use the new knowledge
        if not self.knowledge_base:
            self.knowledge_base = new_knowledge.copy()
            
            # Add history entry
            self.knowledge_history.append({
                "version": 1,
                "timestamp": datetime.now().isoformat(),
                "description": "Initial version",
                "source": source
            })
            
            self.last_update_time = datetime.now().isoformat()
            
            # Create embeddings for the new knowledge
            if self.sentence_encoder:
                self._create_concept_embeddings()
                
            # Save updated knowledge base
            self._save_knowledge_base()
            
            return self.knowledge_base
        
        # Otherwise, perform incremental update
        updated_knowledge = self._merge_knowledge(self.knowledge_base, new_knowledge)
        
        # Add history entry
        version = len(self.knowledge_history) + 1
        self.knowledge_history.append({
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "source": source
        })
        
        self.last_update_time = datetime.now().isoformat()
        
        # Update knowledge base
        self.knowledge_base = updated_knowledge
        
        # Update embeddings
        if self.sentence_encoder:
            self._create_concept_embeddings()
            
        # Save updated knowledge base
        self._save_knowledge_base()
        
        return updated_knowledge
    
    def _merge_knowledge(
        self, 
        existing_knowledge: Dict[str, Any], 
        new_knowledge: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge existing knowledge with new knowledge.
        
        Args:
            existing_knowledge: Existing knowledge base
            new_knowledge: New knowledge to incorporate
            
        Returns:
            Merged knowledge
        """
        merged = existing_knowledge.copy()
        
        # Track changes for later reporting
        changes = {
            "added": {k: [] for k in ["core_concepts", "terminology", "research_questions", 
                                      "methodologies", "key_findings", "measurements"]},
            "updated": {k: [] for k in ["core_concepts", "terminology", "research_questions", 
                                       "methodologies", "key_findings", "measurements"]},
            "merged": {k: [] for k in ["core_concepts", "terminology", "research_questions", 
                                      "methodologies", "key_findings", "measurements"]}
        }
        
        # Merge list fields
        for field in ["core_concepts", "research_questions", "methodologies", "key_findings", "measurements"]:
            if field in new_knowledge:
                if field not in merged:
                    merged[field] = []
                    
                for new_item in new_knowledge[field]:
                    # Handle both dict and string items
                    if isinstance(new_item, dict) and "text" in new_item:
                        item_text = new_item["text"]
                        item_dict = new_item
                    else:
                        item_text = str(new_item)
                        item_dict = {"text": item_text}
                    
                    # Check if item exists in knowledge base
                    existing_item = self._find_similar_item(merged[field], item_text)
                    
                    if existing_item is None:
                        # Add new item
                        if "confidence" not in item_dict:
                            item_dict["confidence"] = 0.8
                        if "source" not in item_dict:
                            item_dict["source"] = "incremental_update"
                            
                        merged[field].append(item_dict)
                        changes["added"][field].append(item_text)
                    else:
                        # Merge with existing item
                        index, existing = existing_item
                        
                        # Update the item with higher confidence or newer information
                        if isinstance(existing, dict):
                            if "confidence" in item_dict and "confidence" in existing:
                                if item_dict["confidence"] > existing["confidence"]:
                                    # Keep source information
                                    if "sources" in existing:
                                        if "source" in item_dict:
                                            if item_dict["source"] not in existing["sources"]:
                                                existing["sources"].append(item_dict["source"])
                                    elif "source" in existing:
                                        existing["sources"] = [existing["source"]]
                                        if "source" in item_dict:
                                            existing["sources"].append(item_dict["source"])
                                        del existing["source"]
                                    
                                    # Update with new data but keep metadata
                                    merged[field][index] = {**existing, **item_dict}
                                    changes["updated"][field].append(item_text)
                                else:
                                    # Just add the source
                                    if "source" in item_dict:
                                        if "sources" in existing:
                                            if item_dict["source"] not in existing["sources"]:
                                                existing["sources"].append(item_dict["source"])
                                        else:
                                            existing["sources"] = [existing["source"], item_dict["source"]]
                                            del existing["source"]
                                            
                                    # Mark as merged
                                    changes["merged"][field].append(item_text)
                            else:
                                # Update without confidence information
                                merged[field][index] = {**existing, **item_dict}
                                changes["updated"][field].append(item_text)
                        else:
                            # Replace string item with dict
                            merged[field][index] = item_dict
                            changes["updated"][field].append(item_text)
        
        # Merge terminology
        if "terminology" in new_knowledge:
            if "terminology" not in merged:
                merged["terminology"] = {}
                
            for term, definition in new_knowledge["terminology"].items():
                if term not in merged["terminology"]:
                    # Add new term
                    merged["terminology"][term] = definition
                    changes["added"]["terminology"].append(term)
                else:
                    # Check which definition is better
                    existing_def = merged["terminology"][term]
                    
                    if isinstance(definition, dict) and isinstance(existing_def, dict):
                        # Both are dicts, merge with preference to higher confidence
                        if "confidence" in definition and "confidence" in existing_def:
                            if definition["confidence"] > existing_def["confidence"]:
                                merged["terminology"][term] = definition
                                changes["updated"]["terminology"].append(term)
                            else:
                                changes["merged"]["terminology"].append(term)
                        else:
                            # Merge the dictionaries
                            merged["terminology"][term] = {**existing_def, **definition}
                            changes["updated"]["terminology"].append(term)
                    elif isinstance(definition, dict) and not isinstance(existing_def, dict):
                        # New is dict, existing is string
                        if "definition" not in definition:
                            definition["definition"] = existing_def
                        merged["terminology"][term] = definition
                        changes["updated"]["terminology"].append(term)
                    elif not isinstance(definition, dict) and isinstance(existing_def, dict):
                        # New is string, existing is dict
                        existing_def["definition"] = definition
                        changes["merged"]["terminology"].append(term)
                    else:
                        # Both are strings, keep the longer one
                        if len(str(definition)) > len(str(existing_def)):
                            merged["terminology"][term] = definition
                            changes["updated"]["terminology"].append(term)
                        else:
                            changes["merged"]["terminology"].append(term)
        
        # Add metadata about changes
        merged["_metadata"] = merged.get("_metadata", {})
        merged["_metadata"]["last_update"] = datetime.now().isoformat()
        merged["_metadata"]["changes"] = changes
        merged["_metadata"]["version"] = len(self.knowledge_history) + 1
        
        return merged
    
    def _find_similar_item(self, items: List[Any], item_text: str) -> Optional[Tuple[int, Any]]:
        """
        Find similar item in a list based on text similarity.
        
        Args:
            items: List of items to search
            item_text: Text of the item to find
            
        Returns:
            Tuple of (index, item) if found, None otherwise
        """
        # First try exact matching
        for i, item in enumerate(items):
            if isinstance(item, dict) and "text" in item:
                if item["text"].lower() == item_text.lower():
                    return (i, item)
            elif isinstance(item, str):
                if item.lower() == item_text.lower():
                    return (i, item)
        
        # If not found, try semantic matching if available
        if self.sentence_encoder and item_text in self.concept_embeddings:
            item_embedding = self.concept_embeddings[item_text]
            
            best_match = None
            best_score = 0.0
            best_index = -1
            
            for i, item in enumerate(items):
                if isinstance(item, dict) and "text" in item:
                    text = item["text"]
                elif isinstance(item, str):
                    text = item
                else:
                    continue
                    
                if text in self.concept_embeddings:
                    embedding = self.concept_embeddings[text]
                    score = cosine_similarity([item_embedding], [embedding])[0][0]
                    
                    if score > best_score and score >= self.similarity_threshold:
                        best_score = score
                        best_match = item
                        best_index = i
            
            if best_match is not None:
                return (best_index, best_match)
        
        return None
    
    def _save_knowledge_base(self) -> None:
        """Save the current knowledge base with history."""
        # Create output file
        version = len(self.knowledge_history)
        timestamp = int(time.time())
        output_file = self.output_dir / f"knowledge_base_v{version}_{timestamp}.json"
        
        # Create data structure
        data = {
            "knowledge": self.knowledge_base,
            "history": self.knowledge_history,
            "last_update": self.last_update_time
        }
        
        # Save to file
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
            
        # Also save latest version
        latest_file = self.output_dir / "knowledge_base_latest.json"
        with open(latest_file, "w") as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Saved knowledge base v{version} to {output_file}")
    
    def update_from_extraction_file(
        self, 
        extraction_path: Union[str, Path],
        source: str = "extraction_file",
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update knowledge base from an extraction file.
        
        Args:
            extraction_path: Path to extraction file
            source: Source identifier
            description: Update description (if None, generates from file)
            
        Returns:
            Updated knowledge base
        """
        try:
            path = Path(extraction_path)
            
            with open(path, 'r') as f:
                extraction = json.load(f)
                
            # Generate description if not provided
            if description is None:
                description = f"Update from extraction file {path.name}"
                
            # Handle different extraction formats
            if "knowledge" in extraction:
                new_knowledge = extraction["knowledge"]
            elif "extraction" in extraction:
                new_knowledge = extraction["extraction"]
            else:
                new_knowledge = extraction
                
            # Update knowledge base
            updated = self.update_with_new_knowledge(
                new_knowledge,
                source=source,
                description=description
            )
            
            return updated
            
        except Exception as e:
            logger.error(f"Error updating from extraction file: {e}")
            return self.knowledge_base
    
    def get_update_stats(self) -> Dict[str, Any]:
        """
        Get statistics about knowledge base updates.
        
        Returns:
            Dictionary of update statistics
        """
        stats = {
            "versions": len(self.knowledge_history),
            "last_update": self.last_update_time,
            "history": self.knowledge_history,
            "knowledge_size": {}
        }
        
        # Calculate knowledge size statistics
        for field in ["core_concepts", "research_questions", "methodologies", "key_findings", "measurements"]:
            if field in self.knowledge_base:
                stats["knowledge_size"][field] = len(self.knowledge_base[field])
                
        if "terminology" in self.knowledge_base:
            stats["knowledge_size"]["terminology"] = len(self.knowledge_base["terminology"])
            
        return stats
    
    def get_knowledge_diffs(
        self, 
        version1: Optional[int] = None, 
        version2: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get differences between two knowledge base versions.
        
        Args:
            version1: First version (if None, uses earliest)
            version2: Second version (if None, uses latest)
            
        Returns:
            Dictionary of differences
        """
        # Load version history
        if not self.knowledge_history or len(self.knowledge_history) < 2:
            return {"error": "Not enough versions for comparison"}
            
        # Determine versions to compare
        if version1 is None:
            version1 = self.knowledge_history[0]["version"]
        if version2 is None:
            version2 = self.knowledge_history[-1]["version"]
            
        # Find version files
        v1_file = None
        v2_file = None
        
        for file in self.output_dir.glob("knowledge_base_v*_*.json"):
            filename = file.name
            try:
                v = int(filename.split("_v")[1].split("_")[0])
                if v == version1:
                    v1_file = file
                elif v == version2:
                    v2_file = file
            except:
                continue
                
        if v1_file is None or v2_file is None:
            return {"error": f"Could not find files for versions {version1} and {version2}"}
            
        # Load versions
        try:
            with open(v1_file, 'r') as f:
                v1_data = json.load(f)
            with open(v2_file, 'r') as f:
                v2_data = json.load(f)
                
            v1_knowledge = v1_data.get("knowledge", v1_data)
            v2_knowledge = v2_data.get("knowledge", v2_data)
            
            # Calculate differences
            diffs = {
                "version1": version1,
                "version2": version2,
                "added": {},
                "removed": {},
                "modified": {}
            }
            
            # Compare list fields
            for field in ["core_concepts", "research_questions", "methodologies", "key_findings", "measurements"]:
                v1_items = self._get_items_text(v1_knowledge.get(field, []))
                v2_items = self._get_items_text(v2_knowledge.get(field, []))
                
                added = [item for item in v2_items if item not in v1_items]
                removed = [item for item in v1_items if item not in v2_items]
                
                if added:
                    diffs["added"][field] = added
                if removed:
                    diffs["removed"][field] = removed
            
            # Compare terminology
            v1_terms = set(v1_knowledge.get("terminology", {}).keys())
            v2_terms = set(v2_knowledge.get("terminology", {}).keys())
            
            added_terms = v2_terms - v1_terms
            removed_terms = v1_terms - v2_terms
            common_terms = v1_terms.intersection(v2_terms)
            
            if added_terms:
                diffs["added"]["terminology"] = list(added_terms)
            if removed_terms:
                diffs["removed"]["terminology"] = list(removed_terms)
                
            # Check for modified terms
            modified_terms = []
            for term in common_terms:
                v1_def = v1_knowledge["terminology"][term]
                v2_def = v2_knowledge["terminology"][term]
                
                if isinstance(v1_def, dict) and isinstance(v2_def, dict):
                    if v1_def.get("definition") != v2_def.get("definition"):
                        modified_terms.append(term)
                elif isinstance(v1_def, dict) and not isinstance(v2_def, dict):
                    modified_terms.append(term)
                elif not isinstance(v1_def, dict) and isinstance(v2_def, dict):
                    modified_terms.append(term)
                elif v1_def != v2_def:
                    modified_terms.append(term)
                    
            if modified_terms:
                diffs["modified"]["terminology"] = modified_terms
            
            return diffs
            
        except Exception as e:
            logger.error(f"Error calculating diffs: {e}")
            return {"error": str(e)}
    
    def _get_items_text(self, items: List[Any]) -> List[str]:
        """Extract text from a list of items."""
        result = []
        for item in items:
            if isinstance(item, dict) and "text" in item:
                result.append(item["text"])
            elif isinstance(item, str):
                result.append(item)
                
        return result 