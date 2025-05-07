"""
Knowledge Graph Representation Module

This module provides functionality to create, query, and manipulate knowledge graphs
from extracted entities and relations.
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Iterator
from pathlib import Path
from collections import defaultdict, Counter

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import pandas as pd
    from tqdm import tqdm
    import numpy as np
    from rdflib import Graph, Literal, RDF, URIRef, Namespace, BNode
    from rdflib.namespace import RDFS, OWL
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """
    Knowledge graph representation for structured knowledge.
    
    Supports multiple graph formats:
    - NetworkX for in-memory graph analysis and visualization
    - RDFLib for RDF-based knowledge graph compatibility
    """
    
    def __init__(
        self,
        name: str = "knowledge_graph",
        persist_dir: Optional[str] = "data/knowledge_graphs",
        namespace: str = "http://purpose.ai/kg/",
        load_existing: bool = True,
    ):
        """
        Initialize the knowledge graph.
        
        Args:
            name: Name of the knowledge graph
            persist_dir: Directory to store/load the graph
            namespace: Base namespace for RDF representation
            load_existing: Whether to load existing graph if present
        """
        self.name = name
        self.persist_dir = Path(persist_dir) if persist_dir else None
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize graph representations
        self.nx_graph = nx.MultiDiGraph(name=name)
        
        # Initialize RDF graph if dependencies available
        self.rdf_graph = None
        self.ns = None
        if DEPS_AVAILABLE:
            self.rdf_graph = Graph()
            self.ns = Namespace(namespace)
            self.rdf_graph.bind("kg", self.ns)
            self.rdf_graph.bind("rdfs", RDFS)
            self.rdf_graph.bind("owl", OWL)
        
        # Load existing graph if requested
        if load_existing and self.persist_dir:
            self._load_graph()
            
        # Stats tracking
        self.stats = {
            "nodes": 0,
            "edges": 0,
            "node_types": Counter(),
            "edge_types": Counter(),
        }
        
        # Cache for efficient querying
        self._node_cache = {}
        self._relation_cache = defaultdict(list)
    
    def _load_graph(self) -> None:
        """Load existing graph from persistence directory."""
        nx_path = self.persist_dir / f"{self.name}_nx.json"
        rdf_path = self.persist_dir / f"{self.name}.ttl"
        
        # Try to load NetworkX graph
        if nx_path.exists():
            try:
                with open(nx_path, 'r') as f:
                    graph_data = json.load(f)
                    
                # Recreate graph from JSON
                self.nx_graph = nx.readwrite.json_graph.node_link_graph(graph_data)
                logger.info(f"Loaded NetworkX graph from {nx_path} with {len(self.nx_graph.nodes)} nodes and {len(self.nx_graph.edges)} edges")
            except Exception as e:
                logger.error(f"Failed to load NetworkX graph: {e}")
        
        # Try to load RDF graph
        if DEPS_AVAILABLE and rdf_path.exists() and self.rdf_graph is not None:
            try:
                self.rdf_graph.parse(str(rdf_path), format="turtle")
                logger.info(f"Loaded RDF graph from {rdf_path} with {len(self.rdf_graph)} triples")
            except Exception as e:
                logger.error(f"Failed to load RDF graph: {e}")
                
        # Update stats
        self._update_stats()
    
    def save(self) -> None:
        """Save the knowledge graph to disk."""
        if not self.persist_dir:
            logger.warning("No persistence directory specified. Graph not saved.")
            return
            
        # Save NetworkX graph
        nx_path = self.persist_dir / f"{self.name}_nx.json"
        try:
            graph_data = nx.readwrite.json_graph.node_link_data(self.nx_graph)
            with open(nx_path, 'w') as f:
                json.dump(graph_data, f)
            logger.info(f"Saved NetworkX graph to {nx_path}")
        except Exception as e:
            logger.error(f"Failed to save NetworkX graph: {e}")
        
        # Save RDF graph
        if DEPS_AVAILABLE and self.rdf_graph is not None:
            rdf_path = self.persist_dir / f"{self.name}.ttl"
            try:
                self.rdf_graph.serialize(destination=str(rdf_path), format="turtle")
                logger.info(f"Saved RDF graph to {rdf_path}")
            except Exception as e:
                logger.error(f"Failed to save RDF graph: {e}")
    
    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        properties: Dict[str, Any],
        source: Optional[str] = None,
        confidence: float = 1.0
    ) -> str:
        """
        Add an entity to the knowledge graph.
        
        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type of the entity (e.g., "Person", "Organization")
            properties: Dictionary of entity properties
            source: Source of the entity information
            confidence: Confidence score for entity (0-1)
            
        Returns:
            The entity ID
        """
        # Normalize entity ID
        entity_id = self._normalize_id(entity_id)
        
        # Skip if entity already exists with same type
        if self.nx_graph.has_node(entity_id):
            node_data = self.nx_graph.nodes[entity_id]
            if node_data.get("type") == entity_type:
                # Update properties if new ones provided
                for k, v in properties.items():
                    if k not in node_data or node_data[k] != v:
                        node_data[k] = v
                        # Update in RDF graph if available
                        if DEPS_AVAILABLE and self.rdf_graph is not None and self.ns is not None:
                            self._add_property_to_rdf(entity_id, k, v)
                return entity_id
        
        # Add to NetworkX graph
        self.nx_graph.add_node(
            entity_id,
            type=entity_type,
            source=source,
            confidence=confidence,
            **properties
        )
        
        # Add to RDF graph if available
        if DEPS_AVAILABLE and self.rdf_graph is not None and self.ns is not None:
            entity_uri = self.ns[entity_id]
            type_uri = self.ns[entity_type]
            
            # Add type triple
            self.rdf_graph.add((entity_uri, RDF.type, type_uri))
            
            # Add properties
            for prop, value in properties.items():
                self._add_property_to_rdf(entity_id, prop, value)
            
            # Add source and confidence if provided
            if source:
                self.rdf_graph.add((entity_uri, self.ns.source, Literal(source)))
            self.rdf_graph.add((entity_uri, self.ns.confidence, Literal(confidence)))
        
        # Update cache and stats
        self._node_cache[entity_id] = {
            "id": entity_id,
            "type": entity_type,
            "properties": properties,
            "source": source,
            "confidence": confidence
        }
        self.stats["nodes"] += 1
        self.stats["node_types"][entity_type] += 1
        
        return entity_id
    
    def _add_property_to_rdf(self, entity_id: str, prop: str, value: Any) -> None:
        """Add a property to the RDF graph for the given entity."""
        if not DEPS_AVAILABLE or self.rdf_graph is None or self.ns is None:
            return
            
        entity_uri = self.ns[entity_id]
        prop_uri = self.ns[prop]
        
        # Handle different value types
        if isinstance(value, (int, float, str, bool)):
            self.rdf_graph.add((entity_uri, prop_uri, Literal(value)))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, (int, float, str, bool)):
                    self.rdf_graph.add((entity_uri, prop_uri, Literal(item)))
                elif isinstance(item, dict) and "id" in item:
                    # Reference to another entity
                    ref_id = self._normalize_id(item["id"])
                    self.rdf_graph.add((entity_uri, prop_uri, self.ns[ref_id]))
    
    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        properties: Optional[Dict[str, Any]] = None,
        source_doc: Optional[str] = None,
        confidence: float = 1.0,
        bidirectional: bool = False
    ) -> Tuple[str, str, int]:
        """
        Add a relation between two entities.
        
        Args:
            source_id: ID of the source entity
            target_id: ID of the target entity
            relation_type: Type of the relation
            properties: Additional properties for the relation
            source_doc: Source document for the relation
            confidence: Confidence score (0-1)
            bidirectional: Whether to add relation in both directions
            
        Returns:
            Tuple of (source_id, target_id, edge_key)
        """
        # Normalize IDs
        source_id = self._normalize_id(source_id)
        target_id = self._normalize_id(target_id)
        
        # Initialize properties dict if None
        properties = properties or {}
        
        # Add to NetworkX graph
        self.nx_graph.add_edge(
            source_id, 
            target_id,
            key=relation_type,
            type=relation_type,
            source_doc=source_doc,
            confidence=confidence,
            **properties
        )
        
        # Add to RDF graph if available
        if DEPS_AVAILABLE and self.rdf_graph is not None and self.ns is not None:
            source_uri = self.ns[source_id]
            target_uri = self.ns[target_id]
            relation_uri = self.ns[relation_type]
            
            # Add main relation triple
            self.rdf_graph.add((source_uri, relation_uri, target_uri))
            
            # Add properties as reified statements
            if properties or source_doc is not None or confidence < 1.0:
                # Create a blank node for the statement
                stmt = BNode()
                
                # Reify the statement
                self.rdf_graph.add((stmt, RDF.type, RDF.Statement))
                self.rdf_graph.add((stmt, RDF.subject, source_uri))
                self.rdf_graph.add((stmt, RDF.predicate, relation_uri))
                self.rdf_graph.add((stmt, RDF.object, target_uri))
                
                # Add properties to the statement
                for prop, value in properties.items():
                    prop_uri = self.ns[prop]
                    self.rdf_graph.add((stmt, prop_uri, Literal(value)))
                
                # Add source doc and confidence
                if source_doc:
                    self.rdf_graph.add((stmt, self.ns.sourceDoc, Literal(source_doc)))
                self.rdf_graph.add((stmt, self.ns.confidence, Literal(confidence)))
        
        # Update cache
        rel_info = {
            "source": source_id,
            "target": target_id,
            "type": relation_type,
            "properties": properties,
            "source_doc": source_doc,
            "confidence": confidence
        }
        self._relation_cache[(source_id, target_id, relation_type)].append(rel_info)
        
        # Update stats
        self.stats["edges"] += 1
        self.stats["edge_types"][relation_type] += 1
        
        # Add bidirectional relation if requested
        if bidirectional:
            inverse_relation = f"inverse_{relation_type}"
            self.add_relation(
                target_id, 
                source_id,
                inverse_relation,
                properties,
                source_doc,
                confidence,
                bidirectional=False
            )
        
        return source_id, target_id, relation_type
    
    def _normalize_id(self, entity_id: str) -> str:
        """Normalize entity ID to be valid in URIs and as graph nodes."""
        # Replace spaces and special chars with underscores
        normalized = str(entity_id).replace(" ", "_").replace("-", "_")
        # Remove any non-alphanumeric characters except underscore
        normalized = "".join(c for c in normalized if c.isalnum() or c == "_")
        return normalized
    
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity by ID."""
        entity_id = self._normalize_id(entity_id)
        
        # Check cache first
        if entity_id in self._node_cache:
            return self._node_cache[entity_id]
        
        # Check in graph
        if self.nx_graph.has_node(entity_id):
            node_data = self.nx_graph.nodes[entity_id]
            
            entity = {
                "id": entity_id,
                "type": node_data.get("type"),
                "properties": {k: v for k, v in node_data.items() 
                              if k not in ("type", "source", "confidence")},
                "source": node_data.get("source"),
                "confidence": node_data.get("confidence", 1.0)
            }
            
            # Update cache
            self._node_cache[entity_id] = entity
            return entity
        
        return None
    
    def get_relations(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        relation_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get relations matching the specified criteria.
        
        Args:
            source_id: Optional source entity ID
            target_id: Optional target entity ID
            relation_type: Optional relation type
            
        Returns:
            List of matching relations
        """
        results = []
        
        # Normalize IDs if provided
        source_id = self._normalize_id(source_id) if source_id else None
        target_id = self._normalize_id(target_id) if target_id else None
        
        # Use NetworkX to find matching edges
        if source_id and target_id and relation_type:
            # Specific edge lookup
            key = (source_id, target_id, relation_type)
            if key in self._relation_cache:
                return self._relation_cache[key]
            
            if self.nx_graph.has_edge(source_id, target_id, key=relation_type):
                edge_data = self.nx_graph.get_edge_data(source_id, target_id)[relation_type]
                rel = {
                    "source": source_id,
                    "target": target_id,
                    "type": relation_type,
                    "properties": {k: v for k, v in edge_data.items() 
                                  if k not in ("type", "source_doc", "confidence")},
                    "source_doc": edge_data.get("source_doc"),
                    "confidence": edge_data.get("confidence", 1.0)
                }
                results.append(rel)
                
        elif source_id and target_id:
            # All relations between two entities
            if self.nx_graph.has_edge(source_id, target_id):
                for rel_type, edge_data in self.nx_graph.get_edge_data(source_id, target_id).items():
                    rel = {
                        "source": source_id,
                        "target": target_id,
                        "type": rel_type,
                        "properties": {k: v for k, v in edge_data.items() 
                                      if k not in ("type", "source_doc", "confidence")},
                        "source_doc": edge_data.get("source_doc"),
                        "confidence": edge_data.get("confidence", 1.0)
                    }
                    results.append(rel)
                    
        elif source_id and relation_type:
            # All relations from source with specific type
            for _, target, key, edge_data in self.nx_graph.out_edges(source_id, data=True, keys=True):
                if key == relation_type:
                    rel = {
                        "source": source_id,
                        "target": target,
                        "type": relation_type,
                        "properties": {k: v for k, v in edge_data.items() 
                                      if k not in ("type", "source_doc", "confidence")},
                        "source_doc": edge_data.get("source_doc"),
                        "confidence": edge_data.get("confidence", 1.0)
                    }
                    results.append(rel)
                    
        elif target_id and relation_type:
            # All relations to target with specific type
            for source, _, key, edge_data in self.nx_graph.in_edges(target_id, data=True, keys=True):
                if key == relation_type:
                    rel = {
                        "source": source,
                        "target": target_id,
                        "type": relation_type,
                        "properties": {k: v for k, v in edge_data.items() 
                                      if k not in ("type", "source_doc", "confidence")},
                        "source_doc": edge_data.get("source_doc"),
                        "confidence": edge_data.get("confidence", 1.0)
                    }
                    results.append(rel)
                    
        elif source_id:
            # All outgoing relations from source
            for _, target, key, edge_data in self.nx_graph.out_edges(source_id, data=True, keys=True):
                rel = {
                    "source": source_id,
                    "target": target,
                    "type": key,
                    "properties": {k: v for k, v in edge_data.items() 
                                  if k not in ("type", "source_doc", "confidence")},
                    "source_doc": edge_data.get("source_doc"),
                    "confidence": edge_data.get("confidence", 1.0)
                }
                results.append(rel)
                
        elif target_id:
            # All incoming relations to target
            for source, _, key, edge_data in self.nx_graph.in_edges(target_id, data=True, keys=True):
                rel = {
                    "source": source,
                    "target": target_id,
                    "type": key,
                    "properties": {k: v for k, v in edge_data.items() 
                                  if k not in ("type", "source_doc", "confidence")},
                    "source_doc": edge_data.get("source_doc"),
                    "confidence": edge_data.get("confidence", 1.0)
                }
                results.append(rel)
                
        elif relation_type:
            # All relations of specific type
            for source, target, key, edge_data in self.nx_graph.edges(data=True, keys=True):
                if key == relation_type:
                    rel = {
                        "source": source,
                        "target": target,
                        "type": relation_type,
                        "properties": {k: v for k, v in edge_data.items() 
                                      if k not in ("type", "source_doc", "confidence")},
                        "source_doc": edge_data.get("source_doc"),
                        "confidence": edge_data.get("confidence", 1.0)
                    }
                    results.append(rel)
        
        return results
    
    def query(self, sparql_query: str) -> List[Dict[str, Any]]:
        """
        Execute a SPARQL query on the RDF graph.
        
        Args:
            sparql_query: SPARQL query string
            
        Returns:
            List of query results as dictionaries
        """
        if not DEPS_AVAILABLE or self.rdf_graph is None:
            logger.warning("RDFLib not available, cannot execute SPARQL query")
            return []
            
        try:
            results = []
            for row in self.rdf_graph.query(sparql_query):
                result = {}
                for i, var in enumerate(row.labels):
                    value = row[i]
                    # Convert URIRef to string, extract local name
                    if isinstance(value, URIRef):
                        if value.startswith(self.ns):
                            # Extract local name from namespace
                            local_name = str(value).replace(str(self.ns), "")
                            result[var] = local_name
                        else:
                            result[var] = str(value)
                    else:
                        result[var] = value
                results.append(result)
            return results
        except Exception as e:
            logger.error(f"SPARQL query error: {e}")
            return []
    
    def get_entity_neighborhood(
        self,
        entity_id: str,
        depth: int = 1,
        max_relations: int = 100,
        relation_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get the neighborhood of an entity up to a certain depth.
        
        Args:
            entity_id: The central entity ID
            depth: How many hops to explore
            max_relations: Maximum number of relations to return
            relation_types: Optional filter for relation types
            
        Returns:
            Dictionary with nodes and edges in the neighborhood
        """
        entity_id = self._normalize_id(entity_id)
        
        # Check if entity exists
        if not self.nx_graph.has_node(entity_id):
            return {"nodes": [], "edges": []}
            
        # Create neighborhood subgraph
        if depth == 1:
            # Direct neighbors only
            nodes = {entity_id} | set(self.nx_graph.predecessors(entity_id)) | set(self.nx_graph.successors(entity_id))
            subgraph = self.nx_graph.subgraph(nodes)
        else:
            # Use NetworkX ego graph
            subgraph = nx.ego_graph(self.nx_graph, entity_id, radius=depth, undirected=True)
            
        # Filter by relation types if specified
        if relation_types:
            edges_to_keep = []
            for u, v, k, data in subgraph.edges(keys=True, data=True):
                if k in relation_types:
                    edges_to_keep.append((u, v, k))
                    
            # Create new subgraph with filtered edges
            filtered_subgraph = nx.MultiDiGraph()
            for node in subgraph.nodes():
                filtered_subgraph.add_node(node, **subgraph.nodes[node])
                
            for u, v, k in edges_to_keep:
                edge_data = subgraph.get_edge_data(u, v)[k]
                filtered_subgraph.add_edge(u, v, key=k, **edge_data)
                
            subgraph = filtered_subgraph
            
        # Convert to serializable format
        nodes = []
        for node_id in subgraph.nodes():
            node_data = subgraph.nodes[node_id]
            nodes.append({
                "id": node_id,
                "type": node_data.get("type", "Unknown"),
                "properties": {k: v for k, v in node_data.items() 
                              if k not in ("type", "source", "confidence")},
                "source": node_data.get("source"),
                "confidence": node_data.get("confidence", 1.0)
            })
            
        edges = []
        for u, v, k, data in subgraph.edges(keys=True, data=True):
            edges.append({
                "source": u,
                "target": v,
                "type": k,
                "properties": {k2: v2 for k2, v2 in data.items() 
                              if k2 not in ("type", "source_doc", "confidence")},
                "source_doc": data.get("source_doc"),
                "confidence": data.get("confidence", 1.0)
            })
            
        # Limit number of relations if necessary
        if len(edges) > max_relations:
            # Sort by confidence and take top relations
            edges = sorted(edges, key=lambda x: x["confidence"], reverse=True)[:max_relations]
            
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    def merge_entity(self, source_id: str, target_id: str) -> str:
        """
        Merge source entity into target entity.
        All relations pointing to source will be redirected to target.
        
        Args:
            source_id: ID of the source entity to be merged
            target_id: ID of the target entity to keep
            
        Returns:
            ID of the target entity
        """
        source_id = self._normalize_id(source_id)
        target_id = self._normalize_id(target_id)
        
        # Check if both entities exist
        if not (self.nx_graph.has_node(source_id) and self.nx_graph.has_node(target_id)):
            logger.warning(f"Cannot merge: one or both entities don't exist")
            return target_id
            
        # Get all relations involving the source entity
        incoming = []
        for u, _, k, data in self.nx_graph.in_edges(source_id, keys=True, data=True):
            if u != target_id:  # Avoid self-loops
                incoming.append((u, k, data))
                
        outgoing = []
        for _, v, k, data in self.nx_graph.out_edges(source_id, keys=True, data=True):
            if v != target_id:  # Avoid self-loops
                outgoing.append((v, k, data))
                
        # Redirect relations to target
        for u, k, data in incoming:
            self.nx_graph.add_edge(u, target_id, key=k, **data)
            
        for v, k, data in outgoing:
            self.nx_graph.add_edge(target_id, v, key=k, **data)
            
        # Add "same_as" relation in RDF graph
        if DEPS_AVAILABLE and self.rdf_graph is not None and self.ns is not None:
            source_uri = self.ns[source_id]
            target_uri = self.ns[target_id]
            self.rdf_graph.add((source_uri, OWL.sameAs, target_uri))
            
        # Remove the source entity
        self.nx_graph.remove_node(source_id)
        
        # Update cache
        if source_id in self._node_cache:
            del self._node_cache[source_id]
            
        # Update relation cache
        new_relation_cache = defaultdict(list)
        for (s, t, k), rels in self._relation_cache.items():
            new_s = target_id if s == source_id else s
            new_t = target_id if t == source_id else t
            new_relation_cache[(new_s, new_t, k)].extend(rels)
        self._relation_cache = new_relation_cache
        
        # Update stats
        self._update_stats()
        
        return target_id
    
    def _update_stats(self) -> None:
        """Update graph statistics."""
        self.stats["nodes"] = self.nx_graph.number_of_nodes()
        self.stats["edges"] = self.nx_graph.number_of_edges()
        
        # Count node types
        node_types = Counter()
        for _, data in self.nx_graph.nodes(data=True):
            node_type = data.get("type", "Unknown")
            node_types[node_type] += 1
        self.stats["node_types"] = node_types
        
        # Count edge types
        edge_types = Counter()
        for _, _, k in self.nx_graph.edges(keys=True):
            edge_types[k] += 1
        self.stats["edge_types"] = edge_types
    
    def visualize(
        self,
        output_file: Optional[str] = None,
        max_nodes: int = 100,
        central_entity: Optional[str] = None,
        highlight_types: Optional[Dict[str, str]] = None
    ) -> Any:
        """
        Visualize the knowledge graph.
        
        Args:
            output_file: Optional path to save the visualization
            max_nodes: Maximum number of nodes to visualize
            central_entity: Optional central entity to focus on
            highlight_types: Optional dict mapping entity types to colors
            
        Returns:
            Plot figure in notebooks, saves to file otherwise
        """
        if not DEPS_AVAILABLE:
            logger.warning("Visualization dependencies not available")
            return None
            
        # Determine which subgraph to visualize
        if central_entity:
            central_entity = self._normalize_id(central_entity)
            if not self.nx_graph.has_node(central_entity):
                logger.warning(f"Central entity {central_entity} not found")
                return None
                
            # Get neighborhood of central entity
            subgraph = nx.ego_graph(self.nx_graph, central_entity, radius=2, undirected=True)
        else:
            subgraph = self.nx_graph
            
        # Limit to max_nodes if needed
        if subgraph.number_of_nodes() > max_nodes:
            # First try limiting to degree
            degree = dict(subgraph.degree())
            top_nodes = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            subgraph = subgraph.subgraph([n for n, _ in top_nodes])
            
        # Set up colors for node types
        if not highlight_types:
            highlight_types = {
                "Person": "skyblue",
                "Organization": "orange",
                "Location": "green",
                "Concept": "red",
                "Term": "purple",
                "Method": "pink",
                "Unknown": "gray"
            }
            
        # Create node colors
        node_colors = []
        for node in subgraph.nodes():
            node_type = subgraph.nodes[node].get("type", "Unknown")
            color = highlight_types.get(node_type, "gray")
            node_colors.append(color)
            
        # Set up the plot
        plt.figure(figsize=(12, 10))
        
        # Use spring layout for visualization
        pos = nx.spring_layout(subgraph, k=0.15, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            subgraph, 
            pos, 
            node_color=node_colors,
            node_size=500,
            alpha=0.8
        )
        
        # Draw edges with arrows
        nx.draw_networkx_edges(
            subgraph,
            pos,
            width=1.0,
            alpha=0.5,
            arrowsize=15
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            subgraph,
            pos,
            font_size=8,
            font_family="sans-serif"
        )
        
        # Draw edge labels for smaller graphs
        if subgraph.number_of_edges() < 50:
            edge_labels = {}
            for u, v, k in subgraph.edges(keys=True):
                if (u, v) in edge_labels:
                    edge_labels[(u, v)] += f", {k}"
                else:
                    edge_labels[(u, v)] = k
                    
            nx.draw_networkx_edge_labels(
                subgraph,
                pos,
                edge_labels=edge_labels,
                font_size=6
            )
            
        # Add a legend for node types
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=color, markersize=10, label=node_type)
                          for node_type, color in highlight_types.items() 
                          if node_type in [subgraph.nodes[n].get("type", "Unknown") for n in subgraph.nodes()]]
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.axis("off")
        plt.tight_layout()
        
        # Save or show
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            logger.info(f"Saved visualization to {output_file}")
            plt.close()
            return None
        else:
            return plt.gcf()
    
    def export_to_df(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Export the knowledge graph to pandas DataFrames.
        
        Returns:
            Tuple of (nodes_df, edges_df)
        """
        if not DEPS_AVAILABLE:
            logger.warning("Pandas not available for DataFrame export")
            return None, None
            
        # Export nodes
        nodes = []
        for node_id, data in self.nx_graph.nodes(data=True):
            node_data = {
                "id": node_id,
                "type": data.get("type", "Unknown"),
                "source": data.get("source"),
                "confidence": data.get("confidence", 1.0)
            }
            
            # Add all other properties
            for k, v in data.items():
                if k not in ("type", "source", "confidence"):
                    node_data[f"prop_{k}"] = v
                    
            nodes.append(node_data)
            
        # Export edges
        edges = []
        for u, v, k, data in self.nx_graph.edges(keys=True, data=True):
            edge_data = {
                "source": u,
                "target": v,
                "type": k,
                "source_doc": data.get("source_doc"),
                "confidence": data.get("confidence", 1.0)
            }
            
            # Add all other properties
            for prop, val in data.items():
                if prop not in ("type", "source_doc", "confidence"):
                    edge_data[f"prop_{prop}"] = val
                    
            edges.append(edge_data)
            
        nodes_df = pd.DataFrame(nodes)
        edges_df = pd.DataFrame(edges)
        
        return nodes_df, edges_df
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        self._update_stats()
        return self.stats
    
    def validate(self) -> List[Dict[str, Any]]:
        """
        Validate the knowledge graph for common issues.
        
        Returns:
            List of validation issues
        """
        issues = []
        
        # Check for disconnected nodes
        isolated = list(nx.isolates(self.nx_graph))
        if isolated:
            issues.append({
                "type": "disconnected_nodes",
                "count": len(isolated),
                "examples": isolated[:5],
                "severity": "warning"
            })
            
        # Check for duplicate edges
        duplicate_edges = []
        for u, v, data in self.nx_graph.edges(data=True):
            keys = list(data.keys())
            if len(keys) > 1:
                duplicate_edges.append((u, v, keys))
                
        if duplicate_edges:
            issues.append({
                "type": "duplicate_relations",
                "count": len(duplicate_edges),
                "examples": duplicate_edges[:5],
                "severity": "info"
            })
            
        # Check for node conflicts in RDF graph
        if DEPS_AVAILABLE and self.rdf_graph is not None and self.ns is not None:
            try:
                # Query for nodes with multiple types
                query = """
                SELECT ?node (COUNT(DISTINCT ?type) as ?typeCount) WHERE {
                    ?node rdf:type ?type .
                }
                GROUP BY ?node
                HAVING (?typeCount > 1)
                """
                results = list(self.rdf_graph.query(query))
                
                if results:
                    nodes_with_multiple_types = []
                    for row in results:
                        node = str(row[0]).replace(str(self.ns), "")
                        nodes_with_multiple_types.append(node)
                        
                    issues.append({
                        "type": "multiple_types",
                        "count": len(nodes_with_multiple_types),
                        "examples": nodes_with_multiple_types[:5],
                        "severity": "warning"
                    })
            except Exception as e:
                logger.warning(f"Failed to check for node conflicts: {e}")
                
        return issues
    
    def clear(self) -> None:
        """Clear the knowledge graph."""
        self.nx_graph = nx.MultiDiGraph(name=self.name)
        
        if DEPS_AVAILABLE and self.rdf_graph is not None:
            self.rdf_graph = Graph()
            if self.ns:
                self.rdf_graph.bind("kg", self.ns)
                self.rdf_graph.bind("rdfs", RDFS)
                self.rdf_graph.bind("owl", OWL)
                
        self._node_cache = {}
        self._relation_cache = defaultdict(list)
        self._update_stats() 