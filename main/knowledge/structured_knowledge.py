"""
Structured Knowledge Representation Module

This module implements structured knowledge representation formats including:
- Knowledge graphs
- Taxonomic hierarchies
- Concept maps
- Ontologies
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from collections import defaultdict

# Third-party imports
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    from rdflib import Graph, Literal, RDF, URIRef, Namespace, BNode
    from rdflib.namespace import RDFS, OWL
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    import spacy
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

logger = logging.getLogger(__name__)

class StructuredKnowledgeBase:
    """
    Implements structured knowledge representation formats including
    knowledge graphs, taxonomies, and ontologies.
    """
    
    def __init__(
        self, 
        output_dir: str = "output/structured_knowledge",
        knowledge_map_path: Optional[str] = None,
        taxonomy_path: Optional[str] = None,
        enhanced_extraction_path: Optional[str] = None
    ):
        """
        Initialize the structured knowledge representation system.
        
        Args:
            output_dir: Directory to save structured knowledge artifacts
            knowledge_map_path: Path to knowledge map JSON file (optional)
            taxonomy_path: Path to taxonomy JSON file (optional)
            enhanced_extraction_path: Path to enhanced extraction JSON file (optional)
        """
        # Initialize outputs directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize knowledge structures
        self.knowledge_map = None
        self.taxonomy = None
        self.enhanced_extraction = None
        
        # Load data if paths provided
        if knowledge_map_path:
            self._load_knowledge_map(knowledge_map_path)
            
        if taxonomy_path:
            self._load_taxonomy(taxonomy_path)
            
        if enhanced_extraction_path:
            self._load_enhanced_extraction(enhanced_extraction_path)
        
        # Initialize graph structures
        self.knowledge_graph = nx.DiGraph() if DEPS_AVAILABLE else None
        self.rdf_graph = Graph() if DEPS_AVAILABLE else None
        self.ontology = None
        
        # Define namespaces for RDF
        if DEPS_AVAILABLE:
            self.domain_ns = Namespace("http://purpose.ai/domain#")
            self.concept_ns = Namespace("http://purpose.ai/concept#")
            self.relation_ns = Namespace("http://purpose.ai/relation#")
            
            # Add namespaces to RDF graph
            self.rdf_graph.bind("domain", self.domain_ns)
            self.rdf_graph.bind("concept", self.concept_ns)
            self.rdf_graph.bind("relation", self.relation_ns)
            self.rdf_graph.bind("rdfs", RDFS)
            self.rdf_graph.bind("owl", OWL)
        
        # Initialize NLP for entity extraction if available
        self.nlp = None
        if DEPS_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_lg")
                logger.info("Loaded spaCy model for entity extraction")
            except Exception as e:
                logger.warning(f"Could not load spaCy model: {e}")
    
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
    
    def _load_enhanced_extraction(self, path: Union[str, Path]) -> bool:
        """Load enhanced extraction results from a JSON file."""
        try:
            with open(path, 'r') as f:
                self.enhanced_extraction = json.load(f)
            logger.info(f"Loaded enhanced extraction from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading enhanced extraction: {e}")
            return False
    
    def create_knowledge_graph(self) -> nx.DiGraph:
        """
        Create a knowledge graph from available knowledge.
        
        Returns:
            NetworkX DiGraph representing the knowledge graph
        """
        if not DEPS_AVAILABLE:
            logger.error("NetworkX is required for knowledge graph creation.")
            return None
            
        # Initialize the graph
        graph = nx.DiGraph()
        
        # Process knowledge map if available
        if self.knowledge_map:
            self._add_knowledge_map_to_graph(graph)
            
        # Process enhanced extraction if available
        if self.enhanced_extraction:
            self._add_enhanced_extraction_to_graph(graph)
            
        # Process taxonomy if available
        if self.taxonomy:
            self._add_taxonomy_to_graph(graph)
            
        # Store the graph
        self.knowledge_graph = graph
        
        # Save the graph
        self._save_knowledge_graph()
        
        return graph
    
    def _add_knowledge_map_to_graph(self, graph: nx.DiGraph) -> None:
        """Add knowledge map information to the graph."""
        # Add core concepts as nodes
        if "core_concepts" in self.knowledge_map:
            for concept in self.knowledge_map["core_concepts"]:
                if isinstance(concept, dict) and "text" in concept:
                    concept_text = concept["text"]
                    confidence = concept.get("confidence", 0.8)
                    source = concept.get("source", "knowledge_map")
                else:
                    concept_text = str(concept)
                    confidence = 0.8
                    source = "knowledge_map"
                    
                graph.add_node(
                    concept_text,
                    type="concept",
                    confidence=confidence,
                    source=source
                )
        
        # Add terminology
        if "terminology" in self.knowledge_map:
            for term, definition in self.knowledge_map["terminology"].items():
                graph.add_node(
                    term,
                    type="term",
                    definition=definition,
                    confidence=0.9,
                    source="knowledge_map"
                )
        
        # Add methodologies
        if "methodologies" in self.knowledge_map:
            for method in self.knowledge_map["methodologies"]:
                if isinstance(method, dict) and "text" in method:
                    method_text = method["text"]
                    confidence = method.get("confidence", 0.8)
                    source = method.get("source", "knowledge_map")
                else:
                    method_text = str(method)
                    confidence = 0.8
                    source = "knowledge_map"
                    
                graph.add_node(
                    method_text,
                    type="methodology",
                    confidence=confidence,
                    source=source
                )
                
                # Connect methodologies to related concepts
                if "related_concepts" in method:
                    for concept in method["related_concepts"]:
                        graph.add_edge(
                            method_text,
                            concept,
                            type="applied_to",
                            confidence=confidence
                        )
    
    def _add_enhanced_extraction_to_graph(self, graph: nx.DiGraph) -> None:
        """Add enhanced extraction information to the graph."""
        if not self.enhanced_extraction:
            return
            
        # Add entities
        if "entities" in self.enhanced_extraction:
            entities = self.enhanced_extraction["entities"]
            
            # Scientific entities
            if "scientific_entities" in entities:
                for entity in entities["scientific_entities"]:
                    graph.add_node(
                        entity["text"],
                        type="entity",
                        label=entity.get("label", "entity"),
                        confidence=entity.get("confidence", 0.8),
                        source=entity.get("source", "enhanced_extraction")
                    )
            
            # Methods
            if "methods" in entities:
                for method in entities["methods"]:
                    graph.add_node(
                        method["text"],
                        type="method",
                        confidence=method.get("confidence", 0.8),
                        source=method.get("source", "enhanced_extraction")
                    )
            
            # Metrics
            if "metrics" in entities:
                for metric in entities["metrics"]:
                    graph.add_node(
                        metric["text"],
                        type="metric",
                        confidence=metric.get("confidence", 0.8),
                        source=metric.get("source", "enhanced_extraction")
                    )
        
        # Add relations
        if "relations" in self.enhanced_extraction:
            for relation in self.enhanced_extraction["relations"]:
                source_node = relation["source"]
                target_node = relation["target"]
                relation_type = relation["relation_type"]
                confidence = relation.get("confidence", 0.7)
                
                # Add nodes if they don't exist
                if source_node not in graph:
                    graph.add_node(
                        source_node,
                        type="entity",
                        confidence=confidence,
                        source="enhanced_extraction"
                    )
                    
                if target_node not in graph:
                    graph.add_node(
                        target_node,
                        type="entity",
                        confidence=confidence,
                        source="enhanced_extraction"
                    )
                
                # Add the edge
                graph.add_edge(
                    source_node,
                    target_node,
                    type=relation_type,
                    confidence=confidence,
                    evidence=relation.get("evidence", ""),
                    source="enhanced_extraction"
                )
    
    def _add_taxonomy_to_graph(self, graph: nx.DiGraph) -> None:
        """Add taxonomy information to the graph."""
        if not self.taxonomy:
            return
            
        # Process each category in the taxonomy
        for category, items in self.taxonomy.items():
            # Add category node
            graph.add_node(
                category,
                type="category",
                confidence=0.95,
                source="taxonomy"
            )
            
            # Add items and connect to category
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and "name" in item:
                        item_name = item["name"]
                        item_properties = {k: v for k, v in item.items() if k != "name"}
                        
                        graph.add_node(
                            item_name,
                            type="concept",
                            confidence=0.9,
                            source="taxonomy",
                            **item_properties
                        )
                        
                        graph.add_edge(
                            category,
                            item_name,
                            type="has_member",
                            confidence=0.9
                        )
                    elif isinstance(item, str):
                        graph.add_node(
                            item,
                            type="concept",
                            confidence=0.9,
                            source="taxonomy"
                        )
                        
                        graph.add_edge(
                            category,
                            item,
                            type="has_member",
                            confidence=0.9
                        )
    
    def _save_knowledge_graph(self) -> None:
        """Save the knowledge graph in multiple formats."""
        if not DEPS_AVAILABLE or not self.knowledge_graph:
            return
            
        # Save as GraphML
        graphml_path = self.output_dir / "knowledge_graph.graphml"
        nx.write_graphml(self.knowledge_graph, graphml_path)
        
        # Save as JSON for web visualization
        json_path = self.output_dir / "knowledge_graph.json"
        
        # Convert to format suitable for visualization libraries
        nodes = []
        for node, attrs in self.knowledge_graph.nodes(data=True):
            nodes.append({
                "id": node,
                "label": node,
                **attrs
            })
            
        edges = []
        for source, target, attrs in self.knowledge_graph.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                **attrs
            })
            
        graph_data = {
            "nodes": nodes,
            "edges": edges
        }
        
        with open(json_path, "w") as f:
            json.dump(graph_data, f, indent=2)
            
        logger.info(f"Saved knowledge graph to {graphml_path} and {json_path}")
        
        # Visualize a sample of the graph
        self._visualize_graph_sample()
    
    def _visualize_graph_sample(self, max_nodes: int = 50) -> None:
        """Visualize a sample of the knowledge graph."""
        if not DEPS_AVAILABLE or not self.knowledge_graph:
            return
            
        # Sample a smaller graph for visualization
        if len(self.knowledge_graph) > max_nodes:
            # Take a connected subgraph
            connected_components = list(nx.weakly_connected_components(self.knowledge_graph))
            if connected_components:
                largest_component = max(connected_components, key=len)
                if len(largest_component) > max_nodes:
                    # Sample nodes from the largest component
                    sampled_nodes = list(largest_component)[:max_nodes]
                    sample_graph = self.knowledge_graph.subgraph(sampled_nodes)
                else:
                    sample_graph = self.knowledge_graph.subgraph(largest_component)
            else:
                # Just take the first max_nodes
                sample_graph = self.knowledge_graph.subgraph(list(self.knowledge_graph.nodes())[:max_nodes])
        else:
            sample_graph = self.knowledge_graph
            
        try:
            # Create visualization
            plt.figure(figsize=(12, 9))
            
            # Use different colors for different node types
            node_colors = []
            for node in sample_graph.nodes():
                node_type = sample_graph.nodes[node].get("type", "unknown")
                if node_type == "concept":
                    node_colors.append("skyblue")
                elif node_type == "term":
                    node_colors.append("lightgreen")
                elif node_type == "methodology":
                    node_colors.append("orange")
                elif node_type == "entity":
                    node_colors.append("pink")
                elif node_type == "category":
                    node_colors.append("yellow")
                else:
                    node_colors.append("gray")
            
            # Draw the graph
            pos = nx.spring_layout(sample_graph, seed=42)
            nx.draw_networkx_nodes(sample_graph, pos, node_color=node_colors, alpha=0.8)
            nx.draw_networkx_edges(sample_graph, pos, alpha=0.5, arrows=True)
            nx.draw_networkx_labels(sample_graph, pos, font_size=8)
            
            # Save the visualization
            viz_path = self.output_dir / "knowledge_graph_visualization.png"
            plt.savefig(viz_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            logger.info(f"Saved graph visualization to {viz_path}")
        except Exception as e:
            logger.error(f"Error visualizing graph: {e}")
    
    def create_rdf_ontology(self) -> Graph:
        """
        Create an RDF ontology from the knowledge structures.
        
        Returns:
            RDFLib Graph containing the ontology
        """
        if not DEPS_AVAILABLE:
            logger.error("RDFLib is required for RDF ontology creation.")
            return None
            
        # Initialize a new RDF graph
        g = Graph()
        
        # Add namespaces
        g.bind("domain", self.domain_ns)
        g.bind("concept", self.concept_ns)
        g.bind("relation", self.relation_ns)
        g.bind("rdfs", RDFS)
        g.bind("owl", OWL)
        
        # Define the ontology
        ontology_uri = URIRef("http://purpose.ai/ontology")
        g.add((ontology_uri, RDF.type, OWL.Ontology))
        g.add((ontology_uri, RDFS.label, Literal("Purpose Domain Ontology")))
        
        # Define classes
        concept_class = URIRef("http://purpose.ai/ontology#Concept")
        method_class = URIRef("http://purpose.ai/ontology#Method")
        finding_class = URIRef("http://purpose.ai/ontology#Finding")
        category_class = URIRef("http://purpose.ai/ontology#Category")
        
        g.add((concept_class, RDF.type, OWL.Class))
        g.add((method_class, RDF.type, OWL.Class))
        g.add((finding_class, RDF.type, OWL.Class))
        g.add((category_class, RDF.type, OWL.Class))
        
        # Define properties
        has_relation = URIRef("http://purpose.ai/ontology#hasRelation")
        has_confidence = URIRef("http://purpose.ai/ontology#hasConfidence")
        has_source = URIRef("http://purpose.ai/ontology#hasSource")
        has_definition = URIRef("http://purpose.ai/ontology#hasDefinition")
        
        g.add((has_relation, RDF.type, OWL.ObjectProperty))
        g.add((has_confidence, RDF.type, OWL.DatatypeProperty))
        g.add((has_source, RDF.type, OWL.DatatypeProperty))
        g.add((has_definition, RDF.type, OWL.DatatypeProperty))
        
        # Add knowledge from the knowledge graph
        if self.knowledge_graph:
            self._add_graph_to_rdf(g)
        
        # Store the RDF graph
        self.rdf_graph = g
        
        # Save the RDF graph
        self._save_rdf_ontology()
        
        return g
    
    def _add_graph_to_rdf(self, g: Graph) -> None:
        """Add knowledge from the knowledge graph to the RDF graph."""
        if not self.knowledge_graph:
            return
            
        # Add nodes
        for node, attrs in self.knowledge_graph.nodes(data=True):
            # Create a safe URI
            node_uri = self._create_safe_uri(node)
            
            # Add node based on type
            node_type = attrs.get("type", "concept")
            
            if node_type == "concept":
                g.add((node_uri, RDF.type, URIRef("http://purpose.ai/ontology#Concept")))
            elif node_type == "methodology" or node_type == "method":
                g.add((node_uri, RDF.type, URIRef("http://purpose.ai/ontology#Method")))
            elif node_type == "category":
                g.add((node_uri, RDF.type, URIRef("http://purpose.ai/ontology#Category")))
            elif node_type == "term":
                g.add((node_uri, RDF.type, URIRef("http://purpose.ai/ontology#Concept")))
                if "definition" in attrs:
                    g.add((node_uri, URIRef("http://purpose.ai/ontology#hasDefinition"), Literal(attrs["definition"])))
            
            # Add label
            g.add((node_uri, RDFS.label, Literal(node)))
            
            # Add confidence if available
            if "confidence" in attrs:
                g.add((node_uri, URIRef("http://purpose.ai/ontology#hasConfidence"), Literal(attrs["confidence"])))
            
            # Add source if available
            if "source" in attrs:
                g.add((node_uri, URIRef("http://purpose.ai/ontology#hasSource"), Literal(attrs["source"])))
        
        # Add edges
        for source, target, attrs in self.knowledge_graph.edges(data=True):
            source_uri = self._create_safe_uri(source)
            target_uri = self._create_safe_uri(target)
            
            # Create edge based on type
            edge_type = attrs.get("type", "related_to")
            relation_uri = URIRef(f"http://purpose.ai/relation#{edge_type}")
            
            # Add the triple
            g.add((source_uri, relation_uri, target_uri))
            
            # Add relation metadata
            if "confidence" in attrs:
                # Create a blank node for the relation
                rel_node = BNode()
                g.add((rel_node, URIRef("http://purpose.ai/ontology#relatesSource"), source_uri))
                g.add((rel_node, URIRef("http://purpose.ai/ontology#relatesTarget"), target_uri))
                g.add((rel_node, URIRef("http://purpose.ai/ontology#relationType"), Literal(edge_type)))
                g.add((rel_node, URIRef("http://purpose.ai/ontology#hasConfidence"), Literal(attrs["confidence"])))
    
    def _create_safe_uri(self, text: str) -> URIRef:
        """Create a safe URI from text."""
        # Replace spaces and special characters
        safe_text = text.replace(" ", "_").replace("-", "_").replace(".", "_").replace(",", "_")
        safe_text = "".join(c for c in safe_text if c.isalnum() or c == "_")
        
        # Create URI based on text type
        if safe_text.lower().startswith(("method", "approach", "technique")):
            return URIRef(f"http://purpose.ai/method#{safe_text}")
        elif safe_text.lower().startswith(("category", "class", "type")):
            return URIRef(f"http://purpose.ai/category#{safe_text}")
        else:
            return URIRef(f"http://purpose.ai/concept#{safe_text}")
    
    def _save_rdf_ontology(self) -> None:
        """Save the RDF ontology in multiple formats."""
        if not DEPS_AVAILABLE or not self.rdf_graph:
            return
            
        # Save as Turtle
        turtle_path = self.output_dir / "ontology.ttl"
        self.rdf_graph.serialize(destination=str(turtle_path), format="turtle")
        
        # Save as RDF/XML
        rdfxml_path = self.output_dir / "ontology.rdf"
        self.rdf_graph.serialize(destination=str(rdfxml_path), format="xml")
        
        # Save as JSON-LD
        jsonld_path = self.output_dir / "ontology.jsonld"
        self.rdf_graph.serialize(destination=str(jsonld_path), format="json-ld")
        
        logger.info(f"Saved RDF ontology in multiple formats to {self.output_dir}")
    
    def extract_concept_map(self, focus_concepts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract a concept map focused on specific concepts.
        
        Args:
            focus_concepts: List of concepts to focus on (if None, uses most connected concepts)
            
        Returns:
            Dictionary representing the concept map
        """
        if not DEPS_AVAILABLE or not self.knowledge_graph:
            logger.error("NetworkX and knowledge graph required for concept map extraction.")
            return {}
            
        # If no focus concepts provided, use the most connected nodes
        if not focus_concepts:
            # Get the top 20 most connected nodes
            centrality = nx.degree_centrality(self.knowledge_graph)
            focus_concepts = sorted(centrality, key=centrality.get, reverse=True)[:20]
        
        # Extract the subgraph containing the focus concepts and their neighbors
        focus_nodes = set(focus_concepts)
        for concept in focus_concepts:
            if concept in self.knowledge_graph:
                # Add immediate neighbors
                focus_nodes.update(self.knowledge_graph.neighbors(concept))
                
        # Create the subgraph
        concept_graph = self.knowledge_graph.subgraph(focus_nodes)
        
        # Convert to a format suitable for visualization
        concept_map = {
            "concepts": [],
            "relations": []
        }
        
        # Add concepts
        for node, attrs in concept_graph.nodes(data=True):
            concept = {
                "id": node,
                "label": node,
                "type": attrs.get("type", "concept"),
                "isCore": node in focus_concepts
            }
            
            # Add additional properties if available
            for key, value in attrs.items():
                if key not in ["type"]:
                    concept[key] = value
                    
            concept_map["concepts"].append(concept)
        
        # Add relations
        for source, target, attrs in concept_graph.edges(data=True):
            relation = {
                "source": source,
                "target": target,
                "type": attrs.get("type", "related_to")
            }
            
            # Add additional properties if available
            for key, value in attrs.items():
                if key not in ["type"]:
                    relation[key] = value
                    
            concept_map["relations"].append(relation)
        
        # Save the concept map
        self._save_concept_map(concept_map)
        
        return concept_map
    
    def _save_concept_map(self, concept_map: Dict[str, Any]) -> None:
        """Save the concept map as JSON."""
        concept_map_path = self.output_dir / "concept_map.json"
        
        with open(concept_map_path, "w") as f:
            json.dump(concept_map, f, indent=2)
            
        logger.info(f"Saved concept map to {concept_map_path}")
    
    def generate_structured_knowledge(
        self,
        knowledge_map_path: Optional[str] = None,
        taxonomy_path: Optional[str] = None,
        enhanced_extraction_path: Optional[str] = None,
        create_knowledge_graph: bool = True,
        create_rdf_ontology: bool = True,
        create_concept_map: bool = True,
        focus_concepts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate complete structured knowledge representations from available sources.
        
        Args:
            knowledge_map_path: Path to knowledge map (if not already loaded)
            taxonomy_path: Path to taxonomy (if not already loaded)
            enhanced_extraction_path: Path to enhanced extraction (if not already loaded)
            create_knowledge_graph: Whether to create a knowledge graph
            create_rdf_ontology: Whether to create an RDF ontology
            create_concept_map: Whether to create a concept map
            focus_concepts: Concepts to focus on for concept map
            
        Returns:
            Dictionary containing paths to generated artifacts
        """
        # Load data if provided
        if knowledge_map_path:
            self._load_knowledge_map(knowledge_map_path)
            
        if taxonomy_path:
            self._load_taxonomy(taxonomy_path)
            
        if enhanced_extraction_path:
            self._load_enhanced_extraction(enhanced_extraction_path)
            
        # Check if we have data to work with
        if not self.knowledge_map and not self.enhanced_extraction and not self.taxonomy:
            logger.error("No knowledge data available. Please provide at least one knowledge source.")
            return {}
            
        artifacts = {}
        
        # Create knowledge graph
        if create_knowledge_graph:
            logger.info("Creating knowledge graph...")
            graph = self.create_knowledge_graph()
            if graph:
                artifacts["knowledge_graph"] = {
                    "graphml": str(self.output_dir / "knowledge_graph.graphml"),
                    "json": str(self.output_dir / "knowledge_graph.json"),
                    "visualization": str(self.output_dir / "knowledge_graph_visualization.png")
                }
                
        # Create RDF ontology
        if create_rdf_ontology:
            logger.info("Creating RDF ontology...")
            ontology = self.create_rdf_ontology()
            if ontology:
                artifacts["rdf_ontology"] = {
                    "turtle": str(self.output_dir / "ontology.ttl"),
                    "rdf_xml": str(self.output_dir / "ontology.rdf"),
                    "jsonld": str(self.output_dir / "ontology.jsonld")
                }
                
        # Create concept map
        if create_concept_map:
            logger.info("Creating concept map...")
            concept_map = self.extract_concept_map(focus_concepts)
            if concept_map:
                artifacts["concept_map"] = str(self.output_dir / "concept_map.json")
                
        logger.info(f"Structured knowledge generation complete. Created artifacts: {list(artifacts.keys())}")
        
        # Save artifacts summary
        with open(self.output_dir / "artifacts_summary.json", "w") as f:
            json.dump(artifacts, f, indent=2)
            
        return artifacts 