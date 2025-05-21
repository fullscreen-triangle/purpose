"""
Advanced Query Generation Module

This module implements sophisticated query generation strategies for
creating high-quality, diverse queries for language model training.
"""

import os
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from collections import defaultdict

# Third-party imports
try:
    import openai
    import anthropic
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

from main.knowledge.query_generation import StratifiedQueryGenerator

logger = logging.getLogger(__name__)

class AdvancedQueryGenerator:
    """
    Advanced query generation system that creates diverse, high-quality queries
    based on domain knowledge using sophisticated generation strategies.
    """
    
    def __init__(
        self, 
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        openai_model: str = "gpt-4",
        claude_model: str = "claude-3-opus-20240229",
        output_dir: str = "output/advanced_queries",
        knowledge_map_path: Optional[str] = None,
        taxonomy_path: Optional[str] = None,
        diversity_weight: float = 0.7,
        creativity_weight: float = 0.5
    ):
        """
        Initialize the advanced query generator.
        
        Args:
            openai_api_key: OpenAI API key (if None, tries to get from environment)
            anthropic_api_key: Anthropic API key (if None, tries to get from environment)
            openai_model: OpenAI model to use for query generation
            claude_model: Claude model to use for query generation
            output_dir: Directory to save generated queries
            knowledge_map_path: Path to knowledge map JSON file
            taxonomy_path: Path to taxonomy JSON file
            diversity_weight: Weight for diversity in query generation (0-1)
            creativity_weight: Weight for creativity in query generation (0-1)
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
        
        # Set weights for query generation
        self.diversity_weight = min(max(diversity_weight, 0.0), 1.0)
        self.creativity_weight = min(max(creativity_weight, 0.0), 1.0)
        
        # Load knowledge map and taxonomy if provided
        self.knowledge_map = None
        self.taxonomy = None
        
        if knowledge_map_path:
            self._load_knowledge_map(knowledge_map_path)
            
        if taxonomy_path:
            self._load_taxonomy(taxonomy_path)
            
        # Initialize base query generator for compatibility
        self.base_generator = StratifiedQueryGenerator(
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            output_dir=output_dir,
            knowledge_map_path=knowledge_map_path
        )
        
        # Initialize vectorizer for semantic clustering
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english'
        ) if DEPS_AVAILABLE else None
        
        # Query templates for different query types
        self.query_templates = {
            "factual": [
                "What is {concept}?",
                "Define {concept} in the context of {domain}.",
                "Explain the meaning of {concept}.",
                "What are the key characteristics of {concept}?",
                "What is the significance of {concept} in {domain}?"
            ],
            "comparative": [
                "Compare and contrast {concept1} and {concept2}.",
                "What are the key differences between {concept1} and {concept2}?",
                "How does {concept1} differ from {concept2} in terms of {aspect}?",
                "What advantages does {concept1} have over {concept2}?",
                "In what scenarios would {concept1} be preferred over {concept2}?"
            ],
            "analytical": [
                "Analyze the relationship between {concept1} and {concept2}.",
                "How does {concept1} influence {concept2}?",
                "What factors affect the interaction between {concept1} and {concept2}?",
                "Explain the mechanism by which {concept1} affects {concept2}.",
                "What is the theoretical basis for the relationship between {concept1} and {concept2}?"
            ],
            "hypothetical": [
                "What would happen if {concept1} were applied to {concept2}?",
                "How might {concept} evolve in the next five years?",
                "Imagine a scenario where {concept} is no longer valid. What would the implications be?",
                "What if {concept1} could be combined with {concept2}? What would be the result?",
                "How would the field of {domain} change if {concept} was revolutionized?"
            ],
            "application": [
                "How is {concept} applied in real-world situations?",
                "Give an example of {concept} being used to solve a practical problem.",
                "What are the practical applications of {concept} in {domain}?",
                "How can {concept} be implemented in {context}?",
                "What challenges arise when applying {concept} in practice?"
            ],
            "synthesis": [
                "Synthesize the key ideas related to {concept}.",
                "How do {concept1}, {concept2}, and {concept3} work together?",
                "Create a framework that integrates {concept1} and {concept2}.",
                "How could {concept1} and {concept2} be combined to address {problem}?",
                "What new insights emerge when connecting {concept1} and {concept2}?"
            ],
            "evaluation": [
                "Evaluate the effectiveness of {concept} in addressing {problem}.",
                "What are the strengths and limitations of {concept}?",
                "How reliable is {concept} as a method for {task}?",
                "Assess the validity of {concept} in light of recent research.",
                "What criteria should be used to evaluate the success of {concept}?"
            ]
        }
        
        # Advanced query types
        self.advanced_query_types = {
            "counterfactual": [
                "If {assumption} about {concept} were false, how would this change our understanding of {domain}?",
                "What would be different if {concept1} worked opposite to how it currently does?",
                "Imagine {concept} didn't exist. How would researchers in {domain} solve {problem}?",
                "If the relationship between {concept1} and {concept2} were reversed, what would be the implications?",
                "How would {domain} have developed differently if {concept} had been discovered earlier?"
            ],
            "ethical": [
                "What ethical considerations arise when applying {concept}?",
                "How might {concept} impact different stakeholders?",
                "What are the potential unintended consequences of using {concept}?",
                "Are there ethical limits to how {concept} should be applied?",
                "How can {concept} be implemented responsibly?"
            ],
            "interdisciplinary": [
                "How is {concept} viewed differently across {domain1} and {domain2}?",
                "What insights from {domain1} could enhance our understanding of {concept} in {domain2}?",
                "How has {concept} evolved differently in {domain1} versus {domain2}?",
                "What methodological approaches from {domain1} could be applied to {concept} in {domain2}?",
                "What are the boundary areas between {domain1} and {domain2} concerning {concept}?"
            ],
            "historical": [
                "How has {concept} evolved over time?",
                "What were the key milestones in the development of {concept}?",
                "How has our understanding of {concept} changed over the past decade?",
                "What historical factors influenced the development of {concept}?",
                "Compare the early understanding of {concept} with current perspectives."
            ],
            "edge_case": [
                "Under what extreme conditions does {concept} fail to apply?",
                "What are the boundary cases where {concept} becomes problematic?",
                "When pushed to its limits, how does {concept} behave?",
                "What rare scenarios challenge the validity of {concept}?",
                "What exceptions exist to the general principles of {concept}?"
            ]
        }
    
    def _load_knowledge_map(self, path: Union[str, Path]) -> bool:
        """Load a knowledge map from a JSON file."""
        try:
            with open(path, 'r') as f:
                self.knowledge_map = json.load(f)
            logger.info(f"Loaded knowledge map from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading knowledge map: {e}")
            return False
    
    def _load_taxonomy(self, path: Union[str, Path]) -> bool:
        """Load a taxonomy from a JSON file."""
        try:
            with open(path, 'r') as f:
                self.taxonomy = json.load(f)
            logger.info(f"Loaded taxonomy from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading taxonomy: {e}")
            return False
    
    def generate_advanced_queries(
        self,
        num_queries: int = 100,
        domain: str = "general",
        query_types: Optional[List[str]] = None,
        advanced_types_ratio: float = 0.3,
        focus_concepts: Optional[List[str]] = None,
        semantic_clustering: bool = True,
        cluster_count: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate advanced queries using sophisticated strategies.
        
        Args:
            num_queries: Number of queries to generate
            domain: Domain for queries
            query_types: Types of queries to generate (if None, uses all types)
            advanced_types_ratio: Ratio of advanced query types to include
            focus_concepts: List of concepts to focus on (if None, uses all)
            semantic_clustering: Whether to use semantic clustering for diversity
            cluster_count: Number of semantic clusters for query diversity
            
        Returns:
            List of generated queries with metadata
        """
        if not self.knowledge_map:
            logger.error("No knowledge map loaded. Load a knowledge map first.")
            return []
        
        logger.info(f"Generating {num_queries} advanced queries for domain: {domain}")
        
        # Get all available concepts
        concepts = self._get_concepts(focus_concepts)
        if not concepts:
            logger.error("No concepts available for query generation.")
            return []
        
        # Determine query type distribution
        if query_types is None:
            # Use all available query types
            standard_types = list(self.query_templates.keys())
            advanced_types = list(self.advanced_query_types.keys())
        else:
            # Filter to specified query types
            standard_types = [qt for qt in query_types if qt in self.query_templates]
            advanced_types = [qt for qt in query_types if qt in self.advanced_query_types]
        
        # Calculate how many of each type to generate
        advanced_count = int(num_queries * advanced_types_ratio)
        standard_count = num_queries - advanced_count
        
        logger.info(f"Generating {standard_count} standard queries and {advanced_count} advanced queries")
        
        # Generate queries in two phases
        standard_queries = self._generate_standard_queries(
            standard_count, domain, standard_types, concepts, semantic_clustering, cluster_count
        )
        
        advanced_queries = self._generate_advanced_queries(
            advanced_count, domain, advanced_types, concepts
        )
        
        # Combine all queries
        all_queries = standard_queries + advanced_queries
        random.shuffle(all_queries)
        
        # Save queries
        self._save_queries(all_queries)
        
        return all_queries
    
    def _get_concepts(self, focus_concepts: Optional[List[str]] = None) -> List[str]:
        """Get concepts to use for query generation."""
        all_concepts = []
        
        # Extract concepts from knowledge map
        if self.knowledge_map:
            if "core_concepts" in self.knowledge_map:
                for concept in self.knowledge_map["core_concepts"]:
                    if isinstance(concept, dict) and "text" in concept:
                        all_concepts.append(concept["text"])
                    elif isinstance(concept, str):
                        all_concepts.append(concept)
            
            if "terminology" in self.knowledge_map:
                all_concepts.extend(list(self.knowledge_map["terminology"].keys()))
        
        # Extract concepts from taxonomy
        if self.taxonomy:
            for category, items in self.taxonomy.items():
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict) and "name" in item:
                            all_concepts.append(item["name"])
                        elif isinstance(item, str):
                            all_concepts.append(item)
        
        # Remove duplicates and filter if focus concepts provided
        unique_concepts = list(set(all_concepts))
        
        if focus_concepts:
            return [c for c in unique_concepts if c in focus_concepts]
        
        return unique_concepts
    
    def _generate_standard_queries(
        self,
        count: int,
        domain: str,
        query_types: List[str],
        concepts: List[str],
        semantic_clustering: bool,
        cluster_count: int
    ) -> List[Dict[str, Any]]:
        """Generate standard queries using templates."""
        queries = []
        
        # If semantic clustering is enabled, organize concepts into clusters
        if semantic_clustering and DEPS_AVAILABLE and len(concepts) > cluster_count:
            concept_clusters = self._cluster_concepts(concepts, cluster_count)
        else:
            # Just use all concepts without clustering
            concept_clusters = {"all": concepts}
        
        # Calculate queries per cluster
        queries_per_cluster = count // max(1, len(concept_clusters))
        extra_queries = count % max(1, len(concept_clusters))
        
        # Generate queries for each cluster
        for cluster_name, cluster_concepts in concept_clusters.items():
            cluster_query_count = queries_per_cluster
            if extra_queries > 0:
                cluster_query_count += 1
                extra_queries -= 1
                
            cluster_queries = self._generate_queries_for_concept_group(
                cluster_query_count, domain, query_types, cluster_concepts
            )
            
            queries.extend(cluster_queries)
        
        return queries
    
    def _generate_queries_for_concept_group(
        self,
        count: int,
        domain: str,
        query_types: List[str],
        concepts: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate queries for a specific group of concepts."""
        if not concepts or not query_types:
            return []
            
        queries = []
        concept_pairs = []
        
        # Generate concept pairs for comparative, analytical, etc. queries
        for i in range(min(len(concepts), 10)):
            for j in range(i+1, min(len(concepts), 10)):
                concept_pairs.append((concepts[i], concepts[j]))
        
        # Ensure we have enough concept pairs
        if not concept_pairs and len(query_types) > 1:
            # Remove query types that require pairs
            query_types = [qt for qt in query_types if qt not in ["comparative", "analytical", "synthesis"]]
        
        # Generate queries using templates
        for _ in range(count):
            query_type = random.choice(query_types)
            template = random.choice(self.query_templates[query_type])
            
            if query_type in ["comparative", "analytical", "synthesis"] and concept_pairs:
                # Types requiring two concepts
                concept1, concept2 = random.choice(concept_pairs)
                aspect = random.choice(["methodology", "application", "effectiveness", "theory", "implementation"])
                
                # For synthesis, we might need three concepts
                if query_type == "synthesis" and "{concept3}" in template:
                    available_concepts = [c for c in concepts if c != concept1 and c != concept2]
                    concept3 = random.choice(available_concepts) if available_concepts else concept1
                    
                    query_text = template.format(
                        concept1=concept1,
                        concept2=concept2,
                        concept3=concept3,
                        problem=f"issues in {domain}"
                    )
                else:
                    query_text = template.format(
                        concept1=concept1,
                        concept2=concept2,
                        aspect=aspect,
                        problem=f"challenges in {domain}"
                    )
            else:
                # Types requiring single concept
                concept = random.choice(concepts)
                query_text = template.format(
                    concept=concept,
                    domain=domain,
                    context=f"the field of {domain}",
                    task=f"solving problems in {domain}",
                    problem=f"issues in {domain}"
                )
            
            # Add the query
            query = {
                "text": query_text,
                "type": query_type,
                "domain": domain,
                "generated_by": "template",
                "complexity": random.choice(["basic", "intermediate", "advanced", "expert"])
            }
            
            queries.append(query)
        
        return queries
    
    def _generate_advanced_queries(
        self,
        count: int,
        domain: str,
        query_types: List[str],
        concepts: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate advanced queries using sophisticated templates."""
        if not query_types:
            query_types = list(self.advanced_query_types.keys())
            
        queries = []
        
        # Generate domain pairs for interdisciplinary queries
        related_domains = self._get_related_domains(domain)
        
        # Generate some queries using templates
        template_queries = []
        for query_type in query_types:
            templates = self.advanced_query_types.get(query_type, [])
            if not templates:
                continue
                
            for _ in range(count // len(query_types) + 1):
                if len(template_queries) >= count:
                    break
                    
                template = random.choice(templates)
                concept = random.choice(concepts)
                
                if query_type == "counterfactual":
                    # Generate a counterfactual query
                    assumption = random.choice([
                        "the fundamental principles", 
                        "common assumptions", 
                        "the causal mechanisms", 
                        "theoretical underpinnings"
                    ])
                    query_text = template.format(
                        concept=concept,
                        domain=domain,
                        assumption=assumption,
                        problem=f"challenges in {domain}",
                        concept1=concept,
                        concept2=random.choice([c for c in concepts if c != concept])
                    )
                elif query_type == "interdisciplinary" and related_domains:
                    # Generate an interdisciplinary query
                    domain1 = domain
                    domain2 = random.choice(related_domains)
                    query_text = template.format(
                        concept=concept,
                        domain1=domain1,
                        domain2=domain2
                    )
                else:
                    # Generate other advanced query types
                    query_text = template.format(
                        concept=concept,
                        domain=domain,
                        problem=f"challenges in {domain}",
                        concept1=concept,
                        concept2=random.choice([c for c in concepts if c != concept])
                    )
                
                query = {
                    "text": query_text,
                    "type": query_type,
                    "domain": domain,
                    "generated_by": "advanced_template",
                    "complexity": "expert"
                }
                
                template_queries.append(query)
        
        # Generate remaining queries using LLM
        llm_query_count = count - len(template_queries)
        if llm_query_count > 0 and self.openai_client:
            llm_queries = self._generate_queries_with_llm(
                llm_query_count, domain, query_types, concepts
            )
            queries.extend(llm_queries)
        
        # Add template queries
        queries.extend(template_queries[:count - len(queries)])
        
        return queries
    
    def _generate_queries_with_llm(
        self,
        count: int,
        domain: str,
        query_types: List[str],
        concepts: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate queries using LLM for maximum sophistication."""
        if not self.openai_client:
            return []
            
        queries = []
        batch_size = min(10, count)  # Generate in batches to avoid too large prompts
        
        # Create concept context
        concept_text = ", ".join(random.sample(concepts, min(20, len(concepts))))
        
        for i in range(0, count, batch_size):
            batch_count = min(batch_size, count - i)
            
            # Create query types text
            query_types_text = ", ".join(query_types)
            
            # Creativity parameter based on current settings
            creativity_param = "high" if self.creativity_weight > 0.7 else "medium" if self.creativity_weight > 0.3 else "low"
            
            # Construct the prompt
            prompt = f"""
            Generate {batch_count} sophisticated, advanced-level questions about the domain of {domain}.
            
            Available concepts: {concept_text}
            
            Query types to include: {query_types_text}
            
            Guidelines:
            1. Focus on creating thought-provoking questions that would challenge experts in {domain}
            2. Create questions that require deep understanding and analysis
            3. Creativity level: {creativity_param}
            4. Make questions specific and well-defined
            5. Include some questions that connect multiple concepts
            
            Return the results as a JSON array where each item has:
            - "text": the question text
            - "type": the query type (from the provided types)
            - "concepts": concepts addressed in the question
            - "complexity": the complexity level (should be "advanced" or "expert")
            """
            
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": f"You are an expert in {domain} who creates sophisticated questions."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                result = json.loads(content)
                
                if "questions" in result:
                    batch_queries = result["questions"]
                else:
                    # Assume the response is the array itself
                    batch_queries = result
                
                for query in batch_queries:
                    # Add metadata
                    query["domain"] = domain
                    query["generated_by"] = "llm"
                    queries.append(query)
                    
            except Exception as e:
                logger.error(f"Error generating queries with LLM: {e}")
        
        return queries
    
    def _cluster_concepts(self, concepts: List[str], num_clusters: int) -> Dict[str, List[str]]:
        """Cluster concepts semantically for more diverse query generation."""
        if not DEPS_AVAILABLE or not self.vectorizer:
            return {"all": concepts}
            
        try:
            # Create document-term matrix
            concept_matrix = self.vectorizer.fit_transform([c.lower() for c in concepts])
            
            # Apply k-means clustering
            n_clusters = min(num_clusters, len(concepts))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(concept_matrix)
            
            labels = kmeans.labels_
            
            # Group concepts by cluster
            clusters = defaultdict(list)
            for i, concept in enumerate(concepts):
                clusters[f"cluster_{labels[i]}"].append(concept)
                
            return dict(clusters)
            
        except Exception as e:
            logger.error(f"Error clustering concepts: {e}")
            return {"all": concepts}
    
    def _get_related_domains(self, domain: str) -> List[str]:
        """Get domains related to the given domain for interdisciplinary queries."""
        # This could be expanded with a more sophisticated domain relationship map
        domain_relationships = {
            "physics": ["mathematics", "chemistry", "engineering", "astronomy"],
            "biology": ["chemistry", "medicine", "ecology", "genetics"],
            "computer_science": ["mathematics", "electrical_engineering", "data_science", "artificial_intelligence"],
            "medicine": ["biology", "chemistry", "psychology", "public_health"],
            "psychology": ["neuroscience", "sociology", "medicine", "education"],
            "economics": ["mathematics", "political_science", "sociology", "finance"],
            "linguistics": ["psychology", "computer_science", "anthropology", "cognitive_science"],
            "general": ["mathematics", "biology", "physics", "psychology", "computer_science"]
        }
        
        return domain_relationships.get(domain.lower(), ["mathematics", "computer_science", "psychology"])
    
    def _save_queries(self, queries: List[Dict[str, Any]]) -> None:
        """Save generated queries to output directory."""
        timestamp = random.randint(10000, 99999)  # Simple unique identifier
        output_file = self.output_dir / f"advanced_queries_{timestamp}.json"
        
        with open(output_file, "w") as f:
            json.dump(queries, f, indent=2)
            
        logger.info(f"Saved {len(queries)} advanced queries to {output_file}")
    
    def generate_query_response_pairs(
        self,
        queries: List[Dict[str, Any]],
        model_name: str = "gpt-4",
        save_results: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for a set of queries using a specified model.
        
        This delegates to the base query generator for compatibility.
        
        Args:
            queries: List of query dictionaries
            model_name: Model to use for generating responses
            save_results: Whether to save results to file
            
        Returns:
            List of query-response pairs
        """
        return self.base_generator.generate_query_response_pairs(
            queries, model_name, save_results
        ) 