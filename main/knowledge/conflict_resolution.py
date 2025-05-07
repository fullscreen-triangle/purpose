"""
Knowledge Conflict Resolution Module

This module implements a system for detecting and resolving conflicts in 
extracted knowledge, ensuring consistency and accuracy.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from collections import defaultdict

# Third-party imports
try:
    import openai
    import anthropic
    import numpy as np
    from sentence_transformers import SentenceTransformer
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

logger = logging.getLogger(__name__)

class KnowledgeConflictResolver:
    """
    System for detecting and resolving conflicts in knowledge extracted from
    different sources, ensuring a consistent and accurate knowledge base.
    """
    
    def __init__(
        self, 
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        output_dir: str = "output/conflict_resolution",
        knowledge_base_path: Optional[str] = None,
        auto_resolve: bool = False,
        confidence_threshold: float = 0.8
    ):
        """
        Initialize the knowledge conflict resolver.
        
        Args:
            openai_api_key: OpenAI API key (if None, tries to get from environment)
            anthropic_api_key: Anthropic API key (if None, tries to get from environment)
            output_dir: Directory to save resolved knowledge
            knowledge_base_path: Path to existing knowledge base (optional)
            auto_resolve: Whether to automatically resolve conflicts
            confidence_threshold: Confidence threshold for auto-resolution
        """
        # Initialize outputs directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set parameters
        self.auto_resolve = auto_resolve
        self.confidence_threshold = confidence_threshold
        
        # Initialize API clients
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        
        if self.openai_api_key:
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
            logger.warning("No OpenAI API key provided.")
        
        self.anthropic_client = None
        if self.anthropic_api_key:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic client: {e}")
        
        # Initialize knowledge base
        self.knowledge_base = {}
        if knowledge_base_path:
            self.load_knowledge_base(knowledge_base_path)
            
        # Initialize sentence encoder for semantic similarity
        self.sentence_encoder = None
        if DEPS_AVAILABLE:
            try:
                self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Initialized sentence encoder for conflict detection")
            except Exception as e:
                logger.warning(f"Failed to initialize sentence encoder: {e}")
                
        # Tracking for resolved conflicts
        self.resolved_conflicts = []
    
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
            else:
                # Assume the entire file is the knowledge base
                self.knowledge_base = data
                
            logger.info(f"Loaded knowledge base from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            return False
    
    def detect_conflicts(
        self, 
        new_knowledge: Dict[str, Any],
        source: str = "new_extraction"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect conflicts between new knowledge and existing knowledge base.
        
        Args:
            new_knowledge: New knowledge to check for conflicts
            source: Source of the new knowledge
            
        Returns:
            Dictionary of detected conflicts
        """
        logger.info(f"Detecting conflicts with new knowledge from {source}")
        
        # If knowledge base is empty, no conflicts possible
        if not self.knowledge_base:
            return {}
            
        conflicts = {
            "terminology": [],
            "facts": [],
            "findings": [],
            "methodologies": []
        }
        
        # Check terminology conflicts
        if "terminology" in new_knowledge and "terminology" in self.knowledge_base:
            conflicts["terminology"] = self._detect_terminology_conflicts(
                self.knowledge_base["terminology"],
                new_knowledge["terminology"]
            )
        
        # Check factual conflicts in core concepts
        if "core_concepts" in new_knowledge and "core_concepts" in self.knowledge_base:
            conflicts["facts"].extend(self._detect_factual_conflicts(
                self.knowledge_base["core_concepts"],
                new_knowledge["core_concepts"],
                conflict_type="core_concept"
            ))
        
        # Check conflicts in findings
        if "key_findings" in new_knowledge and "key_findings" in self.knowledge_base:
            conflicts["findings"] = self._detect_factual_conflicts(
                self.knowledge_base["key_findings"],
                new_knowledge["key_findings"],
                conflict_type="finding"
            )
        
        # Check conflicts in methodologies
        if "methodologies" in new_knowledge and "methodologies" in self.knowledge_base:
            conflicts["methodologies"] = self._detect_factual_conflicts(
                self.knowledge_base["methodologies"],
                new_knowledge["methodologies"],
                conflict_type="methodology"
            )
            
        # Remove empty conflict categories
        conflicts = {k: v for k, v in conflicts.items() if v}
        
        # Log conflict summary
        total_conflicts = sum(len(v) for v in conflicts.values())
        logger.info(f"Detected {total_conflicts} conflicts: {', '.join(f'{len(v)} {k}' for k, v in conflicts.items())}")
        
        return conflicts
    
    def _detect_terminology_conflicts(
        self, 
        existing_terms: Dict[str, Any],
        new_terms: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect conflicts in terminology definitions.
        
        Args:
            existing_terms: Existing terminology
            new_terms: New terminology
            
        Returns:
            List of terminology conflicts
        """
        conflicts = []
        
        for term, new_def in new_terms.items():
            if term in existing_terms:
                existing_def = existing_terms[term]
                
                # Extract definition text
                if isinstance(new_def, dict) and "definition" in new_def:
                    new_def_text = new_def["definition"]
                else:
                    new_def_text = str(new_def)
                    
                if isinstance(existing_def, dict) and "definition" in existing_def:
                    existing_def_text = existing_def["definition"]
                else:
                    existing_def_text = str(existing_def)
                
                # Check if definitions are different
                if self._significant_text_difference(existing_def_text, new_def_text):
                    # Found a conflict in definition
                    conflict = {
                        "term": term,
                        "existing_definition": existing_def,
                        "new_definition": new_def,
                        "conflict_type": "terminology",
                        "resolved": False
                    }
                    conflicts.append(conflict)
        
        return conflicts
    
    def _detect_factual_conflicts(
        self, 
        existing_items: List[Any],
        new_items: List[Any],
        conflict_type: str = "fact"
    ) -> List[Dict[str, Any]]:
        """
        Detect conflicts in factual statements.
        
        Args:
            existing_items: Existing factual statements
            new_items: New factual statements
            conflict_type: Type of conflict (fact, finding, methodology)
            
        Returns:
            List of factual conflicts
        """
        conflicts = []
        
        # Normalize items to extract text
        existing_texts = {}
        for item in existing_items:
            if isinstance(item, dict) and "text" in item:
                text = item["text"]
                existing_texts[text] = item
            elif isinstance(item, str):
                existing_texts[item] = item
        
        # Check each new item for potential conflicts
        for new_item in new_items:
            if isinstance(new_item, dict) and "text" in new_item:
                new_text = new_item["text"]
                new_data = new_item
            elif isinstance(new_item, str):
                new_text = new_item
                new_data = new_text
            else:
                continue
                
            # Find semantically similar existing items
            similar_items = self._find_similar_items(existing_texts, new_text)
            
            for similar_text, similarity, existing_data in similar_items:
                # Check if statements potentially conflict
                if self._potentially_conflicting(similar_text, new_text):
                    conflict = {
                        "existing_statement": existing_data,
                        "new_statement": new_data,
                        "similarity": similarity,
                        "conflict_type": conflict_type,
                        "resolved": False
                    }
                    conflicts.append(conflict)
        
        return conflicts
    
    def _find_similar_items(
        self, 
        existing_items: Dict[str, Any],
        new_text: str,
        threshold: float = 0.75
    ) -> List[Tuple[str, float, Any]]:
        """
        Find semantically similar items in existing knowledge.
        
        Args:
            existing_items: Dictionary of existing items (text -> data)
            new_text: Text to find similar items for
            threshold: Similarity threshold
            
        Returns:
            List of tuples (similar_text, similarity_score, item_data)
        """
        similar_items = []
        
        # Use sentence encoder if available
        if self.sentence_encoder:
            try:
                # Encode new text
                new_embedding = self.sentence_encoder.encode(new_text)
                
                # Encode existing texts
                existing_texts = list(existing_items.keys())
                existing_embeddings = self.sentence_encoder.encode(existing_texts)
                
                # Calculate similarities
                similarities = np.dot(existing_embeddings, new_embedding) / (
                    np.linalg.norm(existing_embeddings, axis=1) * np.linalg.norm(new_embedding)
                )
                
                # Find similar items
                for i, (text, similarity) in enumerate(zip(existing_texts, similarities)):
                    if similarity >= threshold:
                        similar_items.append((text, float(similarity), existing_items[text]))
                
                # Sort by similarity
                similar_items.sort(key=lambda x: x[1], reverse=True)
                
            except Exception as e:
                logger.error(f"Error in semantic similarity: {e}")
                
        # Fallback to simple text matching if no similar items found
        if not similar_items:
            for text, data in existing_items.items():
                # Simple text overlap
                if len(new_text) > 10 and len(text) > 10:
                    overlap = sum(1 for w in new_text.lower().split() if w in text.lower().split())
                    total_words = len(new_text.split())
                    score = overlap / total_words if total_words > 0 else 0
                    
                    if score >= threshold:
                        similar_items.append((text, score, data))
        
        return similar_items
    
    def _potentially_conflicting(self, text1: str, text2: str) -> bool:
        """
        Determine if two statements are potentially conflicting.
        
        Args:
            text1: First statement
            text2: Second statement
            
        Returns:
            True if potentially conflicting, False otherwise
        """
        # Simple heuristic: look for contradictory language
        contradictions = [
            ("increase", "decrease"),
            ("higher", "lower"),
            ("more", "less"),
            ("positive", "negative"),
            ("enhances", "inhibits"),
            ("improves", "worsens"),
            ("effective", "ineffective"),
            ("significant", "insignificant"),
            ("causes", "prevents"),
            ("true", "false")
        ]
        
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # Check for direct contradictions
        for word1, word2 in contradictions:
            if (word1 in text1_lower and word2 in text2_lower) or (word2 in text1_lower and word1 in text2_lower):
                return True
        
        # Check for negation patterns
        negations = ["not", "no", "never", "doesn't", "isn't", "aren't", "can't", "cannot", "won't"]
        for negation in negations:
            # If one statement contains negation and other doesn't
            if negation in text1_lower and negation not in text2_lower:
                return True
            if negation in text2_lower and negation not in text1_lower:
                return True
        
        # If statements are very similar but not identical, they might be conflicting
        if self._significant_text_difference(text1, text2) and self._calculate_overlap(text1, text2) > 0.7:
            return True
            
        return False
    
    def _significant_text_difference(self, text1: str, text2: str) -> bool:
        """
        Determine if two texts have significant differences.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            True if significant differences, False otherwise
        """
        if text1 == text2:
            return False
            
        # Normalize texts
        text1_words = set(text1.lower().split())
        text2_words = set(text2.lower().split())
        
        # Calculate Jaccard distance
        intersection = len(text1_words.intersection(text2_words))
        union = len(text1_words.union(text2_words))
        
        if union == 0:
            return False
            
        similarity = intersection / union
        
        # If texts are very different or very similar but not identical
        return similarity < 0.5 or (similarity > 0.8 and text1 != text2)
    
    def _calculate_overlap(self, text1: str, text2: str) -> float:
        """
        Calculate word overlap between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Overlap score (0-1)
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1.intersection(words2))
        
        if not words1 or not words2:
            return 0.0
            
        return intersection / min(len(words1), len(words2))
    
    def resolve_conflicts(
        self, 
        conflicts: Dict[str, List[Dict[str, Any]]],
        auto_resolve: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Resolve conflicts in knowledge.
        
        Args:
            conflicts: Dictionary of conflicts by type
            auto_resolve: Whether to automatically resolve (overrides init)
            
        Returns:
            Dictionary of resolved conflicts and results
        """
        if not conflicts:
            return {"resolved": 0, "total": 0, "conflicts": {}}
            
        # Use parameter or instance setting
        do_auto_resolve = self.auto_resolve if auto_resolve is None else auto_resolve
        
        # Track results
        results = {
            "resolved": 0,
            "total": 0,
            "conflicts": {}
        }
        
        # Copy conflicts to avoid modifying original
        all_conflicts = []
        for conflict_type, conflict_list in conflicts.items():
            results["conflicts"][conflict_type] = []
            for conflict in conflict_list:
                conflict_copy = conflict.copy()
                conflict_copy["type"] = conflict_type
                all_conflicts.append(conflict_copy)
                results["total"] += 1
        
        # Resolve each conflict
        for conflict in all_conflicts:
            if do_auto_resolve:
                resolved = self._auto_resolve_conflict(conflict)
            else:
                resolved = conflict.copy()
                resolved["resolution"] = "manual_required"
                resolved["resolved"] = False
                
            results["conflicts"][conflict["type"]].append(resolved)
            
            if resolved.get("resolved", False):
                results["resolved"] += 1
                self.resolved_conflicts.append(resolved)
        
        # Save resolution results
        self._save_resolution_results(results)
        
        return results
    
    def _auto_resolve_conflict(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatically resolve a conflict if possible.
        
        Args:
            conflict: Conflict to resolve
            
        Returns:
            Resolved conflict with resolution details
        """
        resolved = conflict.copy()
        
        # Different resolution strategies based on conflict type
        conflict_type = conflict.get("type", "")
        
        if conflict_type == "terminology":
            resolved = self._resolve_terminology_conflict(resolved)
        else:
            resolved = self._resolve_factual_conflict(resolved)
            
        return resolved
    
    def _resolve_terminology_conflict(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve a terminology conflict.
        
        Args:
            conflict: Terminology conflict
            
        Returns:
            Resolved conflict
        """
        # Get existing and new definitions
        term = conflict["term"]
        
        if isinstance(conflict["existing_definition"], dict):
            existing_def = conflict["existing_definition"].get("definition", str(conflict["existing_definition"]))
            existing_confidence = conflict["existing_definition"].get("confidence", 0.8)
        else:
            existing_def = str(conflict["existing_definition"])
            existing_confidence = 0.8
            
        if isinstance(conflict["new_definition"], dict):
            new_def = conflict["new_definition"].get("definition", str(conflict["new_definition"]))
            new_confidence = conflict["new_definition"].get("confidence", 0.7)
        else:
            new_def = str(conflict["new_definition"])
            new_confidence = 0.7
        
        # Resolution strategies:
        # 1. Use LLM to resolve if available
        if self.openai_client or self.anthropic_client:
            llm_resolution = self._resolve_with_llm(
                term=term,
                definition1=existing_def,
                definition2=new_def,
                conflict_type="terminology"
            )
            
            if llm_resolution:
                conflict["resolution"] = llm_resolution["resolution"]
                conflict["resolved_definition"] = llm_resolution["resolved_text"]
                conflict["resolution_method"] = "llm"
                conflict["resolved"] = True
                conflict["explanation"] = llm_resolution.get("explanation", "Resolved using LLM")
                return conflict
        
        # 2. Resolution based on confidence
        if existing_confidence >= new_confidence:
            # Keep existing definition
            conflict["resolution"] = "keep_existing"
            conflict["resolved_definition"] = conflict["existing_definition"]
            conflict["resolution_method"] = "confidence"
            conflict["resolved"] = True
            conflict["explanation"] = f"Kept existing definition (confidence: {existing_confidence} vs {new_confidence})"
        else:
            # Use new definition
            conflict["resolution"] = "use_new"
            conflict["resolved_definition"] = conflict["new_definition"]
            conflict["resolution_method"] = "confidence"
            conflict["resolved"] = True
            conflict["explanation"] = f"Used new definition (confidence: {new_confidence} vs {existing_confidence})"
            
        return conflict
    
    def _resolve_factual_conflict(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve a factual conflict.
        
        Args:
            conflict: Factual conflict
            
        Returns:
            Resolved conflict
        """
        # Extract statements
        if isinstance(conflict["existing_statement"], dict):
            existing_text = conflict["existing_statement"].get("text", str(conflict["existing_statement"]))
            existing_confidence = conflict["existing_statement"].get("confidence", 0.8)
        else:
            existing_text = str(conflict["existing_statement"])
            existing_confidence = 0.8
            
        if isinstance(conflict["new_statement"], dict):
            new_text = conflict["new_statement"].get("text", str(conflict["new_statement"]))
            new_confidence = conflict["new_statement"].get("confidence", 0.7)
        else:
            new_text = str(conflict["new_statement"])
            new_confidence = 0.7
        
        # Resolution strategies:
        # 1. Use LLM to resolve if available
        if self.openai_client or self.anthropic_client:
            llm_resolution = self._resolve_with_llm(
                statement1=existing_text,
                statement2=new_text,
                conflict_type=conflict.get("conflict_type", "fact")
            )
            
            if llm_resolution:
                conflict["resolution"] = llm_resolution["resolution"]
                conflict["resolved_statement"] = llm_resolution["resolved_text"]
                conflict["resolution_method"] = "llm"
                conflict["resolved"] = True
                conflict["explanation"] = llm_resolution.get("explanation", "Resolved using LLM")
                return conflict
        
        # 2. Resolution based on confidence
        if existing_confidence >= new_confidence:
            # Keep existing statement
            conflict["resolution"] = "keep_existing"
            conflict["resolved_statement"] = conflict["existing_statement"]
            conflict["resolution_method"] = "confidence"
            conflict["resolved"] = True
            conflict["explanation"] = f"Kept existing statement (confidence: {existing_confidence} vs {new_confidence})"
        else:
            # Use new statement
            conflict["resolution"] = "use_new"
            conflict["resolved_statement"] = conflict["new_statement"]
            conflict["resolution_method"] = "confidence"
            conflict["resolved"] = True
            conflict["explanation"] = f"Used new statement (confidence: {new_confidence} vs {existing_confidence})"
            
        return conflict
    
    def _resolve_with_llm(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Resolve a conflict using LLM.
        
        Args:
            **kwargs: Conflict details
            
        Returns:
            Resolution details if successful, None otherwise
        """
        if self.openai_client:
            return self._resolve_with_openai(**kwargs)
        elif self.anthropic_client:
            return self._resolve_with_anthropic(**kwargs)
        else:
            return None
    
    def _resolve_with_openai(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Resolve a conflict using OpenAI.
        
        Args:
            **kwargs: Conflict details
            
        Returns:
            Resolution details if successful, None otherwise
        """
        # Construct prompt based on conflict type
        conflict_type = kwargs.get("conflict_type", "fact")
        
        if conflict_type == "terminology":
            term = kwargs.get("term", "")
            definition1 = kwargs.get("definition1", "")
            definition2 = kwargs.get("definition2", "")
            
            prompt = f"""
            I have two different definitions for the term "{term}" that appear to conflict:
            
            Definition 1: {definition1}
            
            Definition 2: {definition2}
            
            Please analyze these definitions and:
            1. Determine if they genuinely conflict or if they're compatible
            2. If they conflict, provide a unified, accurate definition that resolves the conflict
            3. Explain your reasoning for the resolution
            
            Respond with a JSON object with these fields:
            - resolution: "compatible", "merge", "prefer_1", "prefer_2"
            - resolved_text: the unified definition
            - explanation: brief explanation of your reasoning
            """
        else:
            statement1 = kwargs.get("statement1", "")
            statement2 = kwargs.get("statement2", "")
            
            prompt = f"""
            I have two statements that appear to conflict:
            
            Statement 1: {statement1}
            
            Statement 2: {statement2}
            
            Please analyze these statements and:
            1. Determine if they genuinely conflict or if they're compatible
            2. If they conflict, provide a unified, accurate statement that resolves the conflict
            3. Explain your reasoning for the resolution
            
            Respond with a JSON object with these fields:
            - resolution: "compatible", "merge", "prefer_1", "prefer_2"
            - resolved_text: the unified statement
            - explanation: brief explanation of your reasoning
            """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a scientific knowledge conflict resolver."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            return result
            
        except Exception as e:
            logger.error(f"Error resolving with OpenAI: {e}")
            return None
    
    def _resolve_with_anthropic(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Resolve a conflict using Anthropic Claude.
        
        Args:
            **kwargs: Conflict details
            
        Returns:
            Resolution details if successful, None otherwise
        """
        # Construct prompt based on conflict type
        conflict_type = kwargs.get("conflict_type", "fact")
        
        if conflict_type == "terminology":
            term = kwargs.get("term", "")
            definition1 = kwargs.get("definition1", "")
            definition2 = kwargs.get("definition2", "")
            
            prompt = f"""
            I have two different definitions for the term "{term}" that appear to conflict:
            
            Definition 1: {definition1}
            
            Definition 2: {definition2}
            
            Please analyze these definitions and:
            1. Determine if they genuinely conflict or if they're compatible
            2. If they conflict, provide a unified, accurate definition that resolves the conflict
            3. Explain your reasoning for the resolution
            
            Respond with a JSON object with these fields:
            - resolution: "compatible", "merge", "prefer_1", "prefer_2"
            - resolved_text: the unified definition
            - explanation: brief explanation of your reasoning
            """
        else:
            statement1 = kwargs.get("statement1", "")
            statement2 = kwargs.get("statement2", "")
            
            prompt = f"""
            I have two statements that appear to conflict:
            
            Statement 1: {statement1}
            
            Statement 2: {statement2}
            
            Please analyze these statements and:
            1. Determine if they genuinely conflict or if they're compatible
            2. If they conflict, provide a unified, accurate statement that resolves the conflict
            3. Explain your reasoning for the resolution
            
            Respond with a JSON object with these fields:
            - resolution: "compatible", "merge", "prefer_1", "prefer_2"
            - resolved_text: the unified statement
            - explanation: brief explanation of your reasoning
            """
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                system="You are a scientific knowledge conflict resolver. You analyze conflicting statements or definitions and provide resolutions."
            )
            
            content = response.content[0].text
            
            # Extract JSON from Claude's response
            import re
            json_match = re.search(r'{[\s\S]*}', content)
            
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                return result
            else:
                logger.error("Failed to extract JSON from Claude response")
                return None
                
        except Exception as e:
            logger.error(f"Error resolving with Anthropic: {e}")
            return None
    
    def _save_resolution_results(self, results: Dict[str, Any]) -> None:
        """
        Save conflict resolution results to file.
        
        Args:
            results: Resolution results
        """
        # Generate a filename
        import time
        timestamp = int(time.time())
        output_file = self.output_dir / f"conflict_resolution_{timestamp}.json"
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Saved conflict resolution results to {output_file}")
    
    def apply_resolutions(
        self, 
        knowledge_base: Dict[str, Any],
        resolved_conflicts: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Apply resolved conflicts to a knowledge base.
        
        Args:
            knowledge_base: Knowledge base to apply resolutions to
            resolved_conflicts: List of resolved conflicts (if None, uses stored)
            
        Returns:
            Updated knowledge base
        """
        if resolved_conflicts is None:
            resolved_conflicts = self.resolved_conflicts
            
        if not resolved_conflicts:
            return knowledge_base
            
        logger.info(f"Applying {len(resolved_conflicts)} resolved conflicts to knowledge base")
        
        # Create a copy of the knowledge base
        updated_kb = knowledge_base.copy()
        
        # Track changes
        changes = {
            "updated_terminology": [],
            "updated_facts": [],
            "updated_findings": [],
            "updated_methodologies": []
        }
        
        # Apply each resolved conflict
        for conflict in resolved_conflicts:
            if not conflict.get("resolved", False):
                continue
                
            conflict_type = conflict.get("type", "")
            
            if conflict_type == "terminology":
                # Update terminology
                term = conflict["term"]
                if "resolved_definition" in conflict:
                    updated_kb["terminology"][term] = conflict["resolved_definition"]
                    changes["updated_terminology"].append(term)
            else:
                # Update factual statements
                if "resolved_statement" not in conflict:
                    continue
                    
                field = {
                    "core_concept": "core_concepts",
                    "finding": "key_findings",
                    "methodology": "methodologies"
                }.get(conflict.get("conflict_type", ""), "")
                
                if not field or field not in updated_kb:
                    continue
                    
                # Find the item to update
                existing_statement = conflict["existing_statement"]
                
                for i, item in enumerate(updated_kb[field]):
                    if isinstance(item, dict) and isinstance(existing_statement, dict):
                        if item.get("text") == existing_statement.get("text"):
                            updated_kb[field][i] = conflict["resolved_statement"]
                            changes[f"updated_{field}"].append(existing_statement.get("text", ""))
                            break
                    elif item == existing_statement:
                        updated_kb[field][i] = conflict["resolved_statement"]
                        changes[f"updated_{field}"].append(str(existing_statement))
                        break
        
        # Add metadata about changes
        if "_metadata" not in updated_kb:
            updated_kb["_metadata"] = {}
            
        updated_kb["_metadata"]["conflict_resolution"] = {
            "timestamp": datetime.now().isoformat(),
            "num_conflicts_resolved": len(resolved_conflicts),
            "changes": changes
        }
        
        # Save updated knowledge base
        self._save_updated_knowledge_base(updated_kb)
        
        return updated_kb
    
    def _save_updated_knowledge_base(self, knowledge_base: Dict[str, Any]) -> None:
        """
        Save updated knowledge base to file.
        
        Args:
            knowledge_base: Updated knowledge base
        """
        # Generate a filename
        import time
        timestamp = int(time.time())
        output_file = self.output_dir / f"knowledge_base_resolved_{timestamp}.json"
        
        with open(output_file, "w") as f:
            json.dump(knowledge_base, f, indent=2)
            
        logger.info(f"Saved updated knowledge base to {output_file}")
    
    def process_knowledge(
        self, 
        new_knowledge: Dict[str, Any],
        source: str = "new_extraction",
        auto_resolve: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Process new knowledge by detecting and resolving conflicts.
        
        Args:
            new_knowledge: New knowledge to process
            source: Source of the new knowledge
            auto_resolve: Whether to automatically resolve conflicts
            
        Returns:
            Dictionary of process results
        """
        # Process pipeline:
        # 1. Detect conflicts
        conflicts = self.detect_conflicts(new_knowledge, source)
        
        # 2. Resolve conflicts
        resolution_results = self.resolve_conflicts(conflicts, auto_resolve)
        
        # 3. Apply resolutions to create updated knowledge
        if self.knowledge_base:
            updated_kb = self.apply_resolutions(self.knowledge_base)
        else:
            updated_kb = new_knowledge
            
        # Return results
        return {
            "conflicts_detected": len(conflicts) > 0,
            "num_conflicts": sum(len(v) for v in conflicts.values()),
            "resolution_results": resolution_results,
            "updated_knowledge_base": updated_kb
        } 