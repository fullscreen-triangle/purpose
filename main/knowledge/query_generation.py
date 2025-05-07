"""
Query Generation Module

This module implements the second stage of the enhanced distillation process:
- Strategic query generation based on knowledge maps
- Generation of stratified queries across different knowledge dimensions
"""

import os
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple

# Third-party imports (assuming these are available)
try:
    import openai
    import anthropic
    from tqdm.auto import tqdm
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

logger = logging.getLogger(__name__)

class StratifiedQueryGenerator:
    """
    Implements the strategic query generation phase from the
    enhanced distillation process described in knowledge.md.
    """
    
    def __init__(
        self, 
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        openai_model: str = "gpt-4",
        claude_model: str = "claude-3-opus-20240229",
        output_dir: str = "output/queries",
        knowledge_map_path: Optional[str] = None
    ):
        """
        Initialize the StratifiedQueryGenerator with API keys and configuration.
        
        Args:
            openai_api_key: OpenAI API key (if None, tries to get from environment)
            anthropic_api_key: Anthropic API key (if None, tries to get from environment)
            openai_model: OpenAI model to use for query generation
            claude_model: Claude model to use (if available)
            output_dir: Directory to save generated queries
            knowledge_map_path: Path to the knowledge map JSON file (optional)
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
        if self.anthropic_api_key:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
                self.claude_model = claude_model
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic client: {e}")
        
        # Initialize knowledge structures
        self.knowledge_map = None
        self.taxonomy = None
        
        # Load knowledge map if provided
        if knowledge_map_path:
            self.load_knowledge_map(knowledge_map_path)
        
        # Query dimensions for stratification
        self.query_dimensions = {
            "knowledge_depth": [
                "basic", 
                "intermediate", 
                "advanced", 
                "expert"
            ],
            "query_type": [
                "factual", 
                "comparative", 
                "analytical", 
                "hypothetical", 
                "application", 
                "synthesis", 
                "evaluation"
            ],
            "cognitive_process": [
                "recall", 
                "understand", 
                "apply", 
                "analyze", 
                "evaluate", 
                "create"
            ]
        }
    
    def load_knowledge_map(self, knowledge_map_path: Union[str, Path]) -> bool:
        """
        Load a knowledge map from a JSON file.
        
        Args:
            knowledge_map_path: Path to the knowledge map JSON file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        path = Path(knowledge_map_path)
        if not path.exists():
            logger.error(f"Knowledge map file not found: {path}")
            return False
        
        try:
            with open(path, 'r') as f:
                self.knowledge_map = json.load(f)
            logger.info(f"Loaded knowledge map from {path}")
            
            # Try to load taxonomy from the same directory
            taxonomy_path = path.parent / "taxonomy.json"
            if taxonomy_path.exists():
                with open(taxonomy_path, 'r') as f:
                    self.taxonomy = json.load(f)
                logger.info(f"Loaded taxonomy from {taxonomy_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading knowledge map: {e}")
            return False
    
    def generate_queries(
        self,
        num_queries: int = 100,
        stratified: bool = True,
        focus_concepts: Optional[List[str]] = None,
        balance_across_dimensions: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate stratified queries based on the knowledge map.
        
        Args:
            num_queries: Number of queries to generate
            stratified: Whether to generate stratified queries across dimensions
            focus_concepts: List of concepts to focus on (if None, uses all)
            balance_across_dimensions: Whether to balance queries across dimensions
            
        Returns:
            List of generated queries with metadata
        """
        if not self.knowledge_map:
            logger.error("No knowledge map loaded. Load a knowledge map first.")
            return []
        
        # Determine query distribution
        if stratified and balance_across_dimensions:
            # Distribute queries across dimensions
            dimensions = self._create_dimension_distribution(num_queries)
        else:
            # Just create a placeholder for all queries
            dimensions = [{"knowledge_depth": "random", 
                          "query_type": "random", 
                          "cognitive_process": "random"}] * num_queries
        
        logger.info(f"Generating {num_queries} queries, stratified={stratified}")
        
        # Generate queries based on the distribution
        all_queries = []
        
        # Get concepts to focus on
        concepts = self._get_concepts_for_queries(focus_concepts)
        if not concepts:
            logger.error("No concepts available for query generation.")
            return []
        
        # Track concepts already used to ensure coverage
        concept_usage_count = {concept: 0 for concept in concepts}
        
        # Generate queries in batches for efficiency
        batch_size = min(10, num_queries)
        for i in range(0, num_queries, batch_size):
            batch_end = min(i + batch_size, num_queries)
            batch_dimensions = dimensions[i:batch_end]
            
            # Select concepts for this batch, prioritizing less-used concepts
            batch_concepts = self._select_batch_concepts(
                concepts, 
                batch_end - i, 
                concept_usage_count
            )
            
            # Generate batch of queries
            batch_queries = self._generate_query_batch(
                batch_concepts,
                batch_dimensions
            )
            
            # Update concept usage counts
            for query in batch_queries:
                for concept in query.get("concepts", []):
                    if concept in concept_usage_count:
                        concept_usage_count[concept] += 1
            
            all_queries.extend(batch_queries)
            logger.info(f"Generated {len(all_queries)}/{num_queries} queries")
        
        # Save generated queries
        self._save_queries(all_queries)
        
        return all_queries
    
    def _get_concepts_for_queries(self, focus_concepts: Optional[List[str]] = None) -> List[str]:
        """
        Get the list of concepts to use for query generation.
        
        Args:
            focus_concepts: List of concepts to focus on (if provided)
            
        Returns:
            List of concept names
        """
        if not self.knowledge_map:
            return []
        
        # If focus concepts provided, validate them against the knowledge map
        if focus_concepts:
            available_concepts = {
                concept["concept"] for concept in self.knowledge_map.get("core_concepts", [])
            }
            
            # Filter to only include valid concepts
            valid_concepts = [c for c in focus_concepts if c in available_concepts]
            
            if len(valid_concepts) != len(focus_concepts):
                logger.warning(
                    f"{len(focus_concepts) - len(valid_concepts)} concept(s) not found in knowledge map"
                )
            
            if not valid_concepts:
                logger.warning("No valid focus concepts found. Using all concepts.")
                return [c["concept"] for c in self.knowledge_map.get("core_concepts", [])]
            
            return valid_concepts
        
        # Otherwise, use all concepts from the knowledge map
        return [c["concept"] for c in self.knowledge_map.get("core_concepts", [])]
    
    def _create_dimension_distribution(self, num_queries: int) -> List[Dict[str, str]]:
        """
        Create a distribution of dimension values for stratified query generation.
        
        Args:
            num_queries: Number of queries to generate
            
        Returns:
            List of dictionaries with dimension values for each query
        """
        dimensions = []
        
        # Calculate how many queries should be allocated to each combination
        # We'll use a simplified approach to ensure coverage across dimensions
        
        # Get all dimension values
        knowledge_depths = self.query_dimensions["knowledge_depth"]
        query_types = self.query_dimensions["query_type"]
        cognitive_processes = self.query_dimensions["cognitive_process"]
        
        # Calculate total number of combinations
        num_combinations = len(knowledge_depths) * len(query_types) * len(cognitive_processes)
        
        # Determine minimum number of queries per combination
        base_per_combination = max(1, num_queries // num_combinations)
        extras = num_queries - (base_per_combination * num_combinations)
        
        # Create distribution ensuring coverage
        combination_counter = 0
        for depth in knowledge_depths:
            for q_type in query_types:
                for process in cognitive_processes:
                    # Determine how many of this combination to add
                    count = base_per_combination
                    if combination_counter < extras:
                        count += 1
                    
                    # Add the combination (repeated count times)
                    for _ in range(count):
                        dimensions.append({
                            "knowledge_depth": depth,
                            "query_type": q_type,
                            "cognitive_process": process
                        })
                    
                    combination_counter += 1
                    
                    # If we have enough dimensions, stop
                    if len(dimensions) >= num_queries:
                        break
                
                if len(dimensions) >= num_queries:
                    break
            
            if len(dimensions) >= num_queries:
                break
        
        # If we need to trim (shouldn't happen with the calculations above)
        if len(dimensions) > num_queries:
            dimensions = dimensions[:num_queries]
        
        # If we need to add more (shouldn't happen with the calculations above)
        while len(dimensions) < num_queries:
            # Add random combinations
            dimensions.append({
                "knowledge_depth": random.choice(knowledge_depths),
                "query_type": random.choice(query_types),
                "cognitive_process": random.choice(cognitive_processes)
            })
        
        # Shuffle the dimensions to avoid patterns
        random.shuffle(dimensions)
        
        return dimensions
    
    def _select_batch_concepts(
        self, 
        concepts: List[str], 
        batch_size: int, 
        usage_counts: Dict[str, int]
    ) -> List[str]:
        """
        Select concepts for a batch of queries, prioritizing less-used concepts.
        
        Args:
            concepts: List of available concepts
            batch_size: Size of the batch
            usage_counts: Dictionary tracking concept usage counts
            
        Returns:
            List of concepts for the batch
        """
        # Sort concepts by usage count (less used first)
        sorted_concepts = sorted(concepts, key=lambda c: usage_counts[c])
        
        # Take the least used concepts up to batch_size
        selected = sorted_concepts[:batch_size]
        
        # If we need more, add random concepts
        if len(selected) < batch_size:
            additional = random.sample(
                concepts, 
                min(batch_size - len(selected), len(concepts))
            )
            selected.extend(additional)
        
        # Ensure uniqueness
        return list(set(selected))
    
    def _generate_query_batch(
        self, 
        concepts: List[str], 
        dimensions: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Generate a batch of queries for the given concepts and dimensions.
        
        Args:
            concepts: List of concepts to include in the queries
            dimensions: List of dimension specifications for each query
            
        Returns:
            List of generated queries with metadata
        """
        if not self.openai_client:
            logger.error("OpenAI client not configured. Cannot generate queries.")
            return []
            
        # Prepare concept context from knowledge map
        concept_context = self._prepare_concept_context(concepts)
        
        # Create a batch prompt
        system_prompt = """
        You are an expert in creating educational and assessment questions.
        Your task is to create a diverse set of high-quality queries based on provided concepts.
        
        For each query, you should:
        1. Focus on the assigned concept(s)
        2. Match the specified knowledge depth (basic, intermediate, advanced, expert)
        3. Create the appropriate query type (factual, comparative, analytical, etc.)
        4. Target the specified cognitive process (recall, understand, apply, etc.)
        
        Format your response as a JSON array where each query is an object with:
        {
            "query": "The actual query text",
            "concepts": ["concept1", "concept2"], 
            "knowledge_depth": "depth_level",
            "query_type": "type_of_query",
            "cognitive_process": "cognitive_process",
            "expected_knowledge": "Brief description of the knowledge needed to answer this query",
            "domain": "Specific domain or subdomain this query belongs to"
        }
        
        Guidelines:
        - Queries should be clear, concise, and focused
        - Vary the format (questions, tasks, problems to solve)
        - Ensure appropriate difficulty level for the knowledge depth
        - Make connections between concepts where appropriate
        - Each query should primarily relate to at least one concept
        """
        
        # Create a batch of query specifications
        query_specs = []
        for i, dim in enumerate(dimensions):
            # Get 1-2 concepts for this query
            query_concepts = [concepts[i % len(concepts)]]
            if random.random() < 0.5 and len(concepts) > 1:  # 50% chance of adding a second concept
                second_concept = random.choice([c for c in concepts if c != query_concepts[0]])
                query_concepts.append(second_concept)
            
            query_specs.append({
                "concepts": query_concepts,
                "knowledge_depth": dim["knowledge_depth"],
                "query_type": dim["query_type"],
                "cognitive_process": dim["cognitive_process"]
            })
        
        # Create the user prompt
        user_prompt = f"""
        Please create {len(dimensions)} queries based on the following specifications.
        
        CONCEPT INFORMATION:
        {concept_context}
        
        QUERY SPECIFICATIONS:
        {json.dumps(query_specs, indent=2)}
        
        Generate a unique, high-quality query for each specification.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            
            # The result should be a list of queries
            if isinstance(result, dict) and "queries" in result:
                queries = result["queries"]
            elif isinstance(result, list):
                queries = result
            else:
                logger.warning("Unexpected response format. Using empty list.")
                queries = []
            
            # Validate and clean up queries
            cleaned_queries = []
            for query in queries:
                # Ensure all required fields are present
                if "query" not in query:
                    continue
                
                # Add any missing fields with defaults
                if "concepts" not in query or not query["concepts"]:
                    query["concepts"] = random.sample(concepts, min(2, len(concepts)))
                
                # Add other metadata if missing
                for dim_name in self.query_dimensions:
                    if dim_name not in query or not query[dim_name]:
                        query[dim_name] = random.choice(self.query_dimensions[dim_name])
                
                cleaned_queries.append(query)
            
            return cleaned_queries
            
        except Exception as e:
            logger.error(f"Error generating queries with OpenAI: {e}")
            return []
    
    def _prepare_concept_context(self, concepts: List[str]) -> str:
        """
        Prepare context information about the concepts for the prompt.
        
        Args:
            concepts: List of concept names to prepare context for
            
        Returns:
            Formatted string with concept information
        """
        if not self.knowledge_map:
            return "No knowledge map available."
        
        # Get concept definitions and related info
        concept_info = []
        for concept_name in concepts:
            # Find the concept in the knowledge map
            concept_data = next(
                (c for c in self.knowledge_map.get("core_concepts", []) 
                 if c["concept"].lower() == concept_name.lower()),
                None
            )
            
            if concept_data:
                # Format concept information
                concept_text = f"CONCEPT: {concept_data['concept']}\n"
                concept_text += f"DEFINITION: {concept_data['definition']}\n"
                
                # Add related concepts if available
                if "related_concepts" in concept_data and concept_data["related_concepts"]:
                    related = ", ".join(concept_data["related_concepts"])
                    concept_text += f"RELATED CONCEPTS: {related}\n"
                
                # Find related findings
                related_findings = [
                    f["finding"] for f in self.knowledge_map.get("key_findings", [])
                    if "supports" in f and concept_data["concept"] in f["supports"]
                ]
                
                if related_findings:
                    findings_text = "\n- " + "\n- ".join(related_findings)
                    concept_text += f"KEY FINDINGS: {findings_text}\n"
                
                concept_info.append(concept_text)
            else:
                # Just include the concept name if no details found
                concept_info.append(f"CONCEPT: {concept_name}\n")
        
        return "\n\n".join(concept_info)
    
    def _save_queries(self, queries: List[Dict[str, Any]]) -> None:
        """
        Save generated queries to a JSON file.
        
        Args:
            queries: List of generated query objects
        """
        if not queries:
            logger.warning("No queries to save.")
            return
        
        # Create a filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"queries_{timestamp}.json"
        file_path = self.output_dir / filename
        
        # Save queries to file
        with open(file_path, 'w') as f:
            json.dump(queries, f, indent=2)
            
        logger.info(f"Saved {len(queries)} queries to {file_path}")
    
    def generate_query_response_pairs(
        self,
        queries: List[Dict[str, Any]],
        model_name: str = "gpt-4",
        save_results: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate response pairs for a list of queries using the specified model.
        
        Args:
            queries: List of query objects to generate responses for
            model_name: Name of the model to use ('gpt-4' or 'claude')
            save_results: Whether to save the results to a file
            
        Returns:
            List of query-response pairs
        """
        if not queries:
            logger.warning("No queries provided for response generation.")
            return []
        
        # Select the client based on the model name
        client = None
        if model_name.startswith("gpt"):
            if not self.openai_client:
                logger.error("OpenAI client not configured. Cannot generate responses.")
                return []
            client = self.openai_client
            model = self.openai_model
        elif model_name.startswith("claude"):
            if not self.anthropic_client:
                logger.error("Anthropic client not configured. Cannot generate responses.")
                return []
            client = self.anthropic_client
            model = self.claude_model
        else:
            logger.error(f"Unsupported model: {model_name}")
            return []
        
        # Process each query
        pairs = []
        for query in tqdm(queries, desc=f"Generating responses with {model_name}"):
            try:
                response = self._generate_response_for_query(query, client, model, model_name)
                pair = {
                    "query": query,
                    "response": response,
                    "model": model_name
                }
                pairs.append(pair)
            except Exception as e:
                logger.error(f"Error generating response: {e}")
        
        # Save results if requested
        if save_results and pairs:
            self._save_query_response_pairs(pairs, model_name)
        
        return pairs
    
    def _generate_response_for_query(
        self, 
        query: Dict[str, Any],
        client: Any,
        model: str,
        model_type: str
    ) -> str:
        """
        Generate a response for a single query using the specified model.
        
        Args:
            query: Query object to generate response for
            client: API client (OpenAI or Anthropic)
            model: Model name to use
            model_type: Type of model ('gpt' or 'claude')
            
        Returns:
            Generated response text
        """
        # Create system prompt for comprehensive response
        system_prompt = f"""
        You are an expert AI assistant with deep knowledge in various domains.
        You're answering a query that requires {query.get('knowledge_depth', 'intermediate')} knowledge level.
        This is a {query.get('query_type', 'factual')} type of query targeting the {query.get('cognitive_process', 'understand')} cognitive process.
        
        Provide a comprehensive, accurate, and informative response to the query.
        
        Guidelines:
        - Be thorough and address all aspects of the query
        - Include relevant facts, concepts, and theories
        - Provide examples or illustrations where helpful
        - Structure your response logically
        - Maintain appropriate depth based on the knowledge level
        - Cite sources or references if applicable
        """
        
        # The query text
        query_text = query.get("query", "")
        
        if model_type.startswith("gpt"):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query_text}
                    ],
                    temperature=0.3
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error with OpenAI response generation: {e}")
                return ""
                
        elif model_type.startswith("claude"):
            try:
                response = client.messages.create(
                    model=model,
                    system=system_prompt,
                    max_tokens=2000,
                    messages=[
                        {"role": "user", "content": query_text}
                    ],
                    temperature=0.3
                )
                return response.content[0].text
            except Exception as e:
                logger.error(f"Error with Claude response generation: {e}")
                return ""
        
        return ""
    
    def _save_query_response_pairs(
        self, 
        pairs: List[Dict[str, Any]], 
        model_name: str
    ) -> None:
        """
        Save query-response pairs to a JSON file.
        
        Args:
            pairs: List of query-response pairs
            model_name: Name of the model used for responses
        """
        if not pairs:
            return
        
        # Create a filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"query_responses_{model_name}_{timestamp}.json"
        file_path = self.output_dir / filename
        
        # Save pairs to file
        with open(file_path, 'w') as f:
            json.dump(pairs, f, indent=2)
            
        logger.info(f"Saved {len(pairs)} query-response pairs to {file_path}")
    
    def generate_training_data(
        self,
        num_queries: int = 100,
        models: List[str] = ["gpt-4", "claude"],
        stratified: bool = True,
        focus_concepts: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate a complete training dataset with stratified queries and responses.
        
        Args:
            num_queries: Number of queries to generate
            models: List of models to use for generating responses
            stratified: Whether to generate stratified queries across dimensions
            focus_concepts: List of concepts to focus on (if None, uses all)
            
        Returns:
            Dictionary mapping model names to lists of query-response pairs
        """
        # First, generate the queries
        queries = self.generate_queries(
            num_queries=num_queries,
            stratified=stratified,
            focus_concepts=focus_concepts
        )
        
        if not queries:
            logger.error("Failed to generate queries.")
            return {}
            
        # Generate responses for each model
        results = {}
        for model in models:
            if model.startswith("gpt") and not self.openai_client:
                logger.warning(f"Skipping {model} - OpenAI client not configured")
                continue
                
            if model.startswith("claude") and not self.anthropic_client:
                logger.warning(f"Skipping {model} - Anthropic client not configured")
                continue
                
            logger.info(f"Generating responses with {model}")
            pairs = self.generate_query_response_pairs(
                queries=queries,
                model_name=model,
                save_results=True
            )
            
            if pairs:
                results[model] = pairs
                
        # Save the complete dataset
        if results:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_dataset_{timestamp}.json"
            file_path = self.output_dir / filename
            
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"Saved complete training dataset to {file_path}")
                
        return results 