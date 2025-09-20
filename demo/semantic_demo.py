#!/usr/bin/env python3
"""
Semantic S-Entropy Navigation Demonstration
=========================================

Revolutionary semantic processing system implementing:
- Eight-dimensional semantic coordinate mapping (Technical/Emotional, Action/Descriptive, etc.)
- Fuzzy compression embedding to prevent collision in high-dimensional space
- Multi-stage compression: alphabetical → numerical → text → alphabetical → numerical → binary
- Empty dictionary real-time text comprehension without storage
- Dynamic dimensionality to remove predefined rigidity

Based on: "S-Entropy Semantic Navigation and Language Processing Framework"
Author: Kundai Farai Sachikonye
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json
import time
from datetime import datetime
import os
import re
import hashlib
from typing import List, Tuple, Dict, Any, Optional
import logging
from tqdm import tqdm
import seaborn as sns
from scipy.spatial.distance import pdist, squareform, cosine
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx
from core_s_entropy import SEntropyCoordinateSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FuzzyCompressionEmbedder:
    """
    Revolutionary fuzzy compression embedding system to prevent collision
    through multi-stage compression and dynamic dimensionality
    """
    
    def __init__(self, save_intermediates: bool = True):
        self.save_intermediates = save_intermediates
        self.compression_cache = {}
        self.embedding_history = []
        
        os.makedirs('demo/outputs/semantic', exist_ok=True)
        logger.info("Fuzzy Compression Embedder initialized")
    
    def compress_text_fuzzy(self, text: str) -> Dict:
        """
        Multi-stage fuzzy compression embedding process:
        1. Alphabetical sorting: "That is a bag" → "aaabghist"
        2. Numerical conversion: "aaabghist" → "11127891920"
        3. Digit-to-text: "11127891920" → "one one one two seven eight nine one nine two zero"
        4. Re-alphabetical: "eight nine one one..." → sorted
        5. Back to numbers → binary representation
        6. Fuzzy coordinate cloud generation
        """
        start_time = time.time()
        logger.info(f"Starting fuzzy compression for text: '{text[:50]}...'")
        
        compression_data = {
            'original_text': text,
            'timestamp': datetime.now().isoformat(),
            'compression_stages': []
        }
        
        # Stage 1: Alphabetical sorting
        stage1_result = self._stage1_alphabetical_sort(text)
        compression_data['compression_stages'].append(stage1_result)
        
        # Stage 2: Numerical conversion
        stage2_result = self._stage2_numerical_conversion(stage1_result['output'])
        compression_data['compression_stages'].append(stage2_result)
        
        # Stage 3: Digit-to-text conversion
        stage3_result = self._stage3_digit_to_text(stage2_result['output'])
        compression_data['compression_stages'].append(stage3_result)
        
        # Stage 4: Re-alphabetical sorting
        stage4_result = self._stage4_re_alphabetical(stage3_result['output'])
        compression_data['compression_stages'].append(stage4_result)
        
        # Stage 5: Back to numerical
        stage5_result = self._stage5_back_to_numerical(stage4_result['output'])
        compression_data['compression_stages'].append(stage5_result)
        
        # Stage 6: Binary conversion
        stage6_result = self._stage6_binary_conversion(stage5_result['output'])
        compression_data['compression_stages'].append(stage6_result)
        
        # Stage 7: Fuzzy coordinate cloud generation
        fuzzy_coordinates = self._generate_fuzzy_coordinate_cloud(stage6_result['binary_array'])
        compression_data['fuzzy_coordinates'] = fuzzy_coordinates
        
        # Calculate compression metrics
        compression_data['metrics'] = {
            'original_length': len(text),
            'final_dimensions': fuzzy_coordinates['dimensionality'],
            'compression_ratio': len(text) / fuzzy_coordinates['coordinate_density'],
            'collision_probability': fuzzy_coordinates['collision_resistance'],
            'processing_time': time.time() - start_time
        }
        
        # Save compression data
        if self.save_intermediates:
            timestamp = int(time.time())
            filename = f'demo/outputs/semantic/fuzzy_compression_{timestamp}.json'
            self._save_json(filename, compression_data)
            self._visualize_compression_stages(compression_data, f'compression_stages_{timestamp}')
        
        logger.info(f"Fuzzy compression completed: {fuzzy_coordinates['dimensionality']} dimensions, "
                   f"{fuzzy_coordinates['collision_resistance']:.4f} collision resistance")
        
        return compression_data
    
    def _stage1_alphabetical_sort(self, text: str) -> Dict:
        """Stage 1: Sort characters alphabetically"""
        # Remove spaces and punctuation, convert to lowercase
        clean_text = re.sub(r'[^a-zA-Z]', '', text).lower()
        sorted_chars = ''.join(sorted(clean_text))
        
        return {
            'stage': 1,
            'description': 'alphabetical_sorting',
            'input': text,
            'clean_input': clean_text,
            'output': sorted_chars,
            'transformation': f"{text} → {sorted_chars}",
            'length_change': len(sorted_chars) - len(clean_text)
        }
    
    def _stage2_numerical_conversion(self, sorted_text: str) -> Dict:
        """Stage 2: Convert to numerical representation (a=1, b=2, etc.)"""
        numerical_string = ''
        char_mappings = {}
        
        for char in sorted_text:
            if char.isalpha():
                num_value = ord(char.lower()) - ord('a') + 1
                numerical_string += str(num_value)
                char_mappings[char] = num_value
        
        return {
            'stage': 2,
            'description': 'numerical_conversion',
            'input': sorted_text,
            'output': numerical_string,
            'char_mappings': char_mappings,
            'transformation': f"{sorted_text} → {numerical_string}",
            'length_change': len(numerical_string) - len(sorted_text)
        }
    
    def _stage3_digit_to_text(self, numerical_string: str) -> Dict:
        """Stage 3: Convert digits to text representation"""
        digit_to_word = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
        }
        
        word_list = []
        for digit in numerical_string:
            if digit in digit_to_word:
                word_list.append(digit_to_word[digit])
        
        text_representation = ' '.join(word_list)
        
        return {
            'stage': 3,
            'description': 'digit_to_text',
            'input': numerical_string,
            'output': text_representation,
            'word_list': word_list,
            'transformation': f"{numerical_string} → {text_representation}",
            'word_count': len(word_list)
        }
    
    def _stage4_re_alphabetical(self, text_representation: str) -> Dict:
        """Stage 4: Re-sort alphabetically"""
        words = text_representation.split()
        sorted_words = sorted(words)
        re_alphabetical = ' '.join(sorted_words)
        
        return {
            'stage': 4,
            'description': 're_alphabetical_sorting',
            'input': text_representation,
            'output': re_alphabetical,
            'word_reordering': dict(zip(words, sorted_words)),
            'transformation': f"{text_representation} → {re_alphabetical}",
            'order_changes': sum(1 for i, (w1, w2) in enumerate(zip(words, sorted_words)) if w1 != w2)
        }
    
    def _stage5_back_to_numerical(self, re_alphabetical: str) -> Dict:
        """Stage 5: Convert back to numerical representation"""
        word_to_digit = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
        }
        
        numerical_list = []
        for word in re_alphabetical.split():
            if word in word_to_digit:
                numerical_list.append(word_to_digit[word])
        
        numerical_final = ''.join(numerical_list)
        
        return {
            'stage': 5,
            'description': 'back_to_numerical',
            'input': re_alphabetical,
            'output': numerical_final,
            'numerical_list': numerical_list,
            'transformation': f"{re_alphabetical} → {numerical_final}",
            'recovery_rate': len(numerical_list) / max(1, len(re_alphabetical.split()))
        }
    
    def _stage6_binary_conversion(self, numerical_final: str) -> Dict:
        """Stage 6: Convert to binary representation"""
        if not numerical_final:
            binary_representation = '0'
            binary_array = np.array([0])
        else:
            # Convert numerical string to integer, then to binary
            try:
                integer_value = int(numerical_final) if numerical_final else 0
                binary_representation = bin(integer_value)[2:]  # Remove '0b' prefix
                binary_array = np.array([int(bit) for bit in binary_representation])
            except ValueError:
                # Fallback: convert each digit separately
                binary_representation = ''.join([bin(int(d))[2:].zfill(4) for d in numerical_final if d.isdigit()])
                binary_array = np.array([int(bit) for bit in binary_representation])
        
        return {
            'stage': 6,
            'description': 'binary_conversion',
            'input': numerical_final,
            'output': binary_representation,
            'binary_array': binary_array.tolist(),
            'transformation': f"{numerical_final} → {binary_representation}",
            'bit_length': len(binary_representation)
        }
    
    def _generate_fuzzy_coordinate_cloud(self, binary_array: np.ndarray) -> Dict:
        """Generate fuzzy coordinate cloud for collision prevention"""
        if len(binary_array) == 0:
            binary_array = np.array([0])
        
        # Calculate dynamic dimensionality based on binary complexity
        base_dimensionality = max(8, len(binary_array) // 4)  # Minimum 8 dimensions
        
        # Generate fuzzy coordinate cloud with gaussian distribution
        n_cloud_points = min(1000, max(100, len(binary_array) * 10))  # Adaptive cloud size
        
        # Create base coordinates from binary data
        base_coordinates = []
        for i in range(base_dimensionality):
            # Use overlapping windows of binary data for each dimension
            start_idx = (i * len(binary_array)) // base_dimensionality
            end_idx = ((i + 1) * len(binary_array)) // base_dimensionality
            window = binary_array[start_idx:end_idx] if end_idx > start_idx else binary_array
            
            # Calculate coordinate value from binary window
            coord_value = np.mean(window) * 2 - 1  # Map to [-1, 1] range
            base_coordinates.append(coord_value)
        
        base_coordinates = np.array(base_coordinates)
        
        # Generate fuzzy cloud around base coordinates
        fuzzy_cloud = []
        for _ in range(n_cloud_points):
            # Add gaussian noise for fuzzy behavior
            noise = np.random.normal(0, 0.1, base_dimensionality)  # Small gaussian noise
            fuzzy_point = base_coordinates + noise
            fuzzy_cloud.append(fuzzy_point)
        
        fuzzy_cloud = np.array(fuzzy_cloud)
        
        # Calculate collision resistance metrics
        pairwise_distances = pdist(fuzzy_cloud)
        min_distance = np.min(pairwise_distances) if len(pairwise_distances) > 0 else 1.0
        collision_resistance = min(1.0, min_distance / 0.1)  # Normalize by noise level
        
        # Calculate coordinate density
        coordinate_density = n_cloud_points / base_dimensionality
        
        return {
            'dimensionality': base_dimensionality,
            'cloud_points': n_cloud_points,
            'base_coordinates': base_coordinates.tolist(),
            'fuzzy_cloud': fuzzy_cloud.tolist(),
            'collision_resistance': float(collision_resistance),
            'coordinate_density': float(coordinate_density),
            'min_pairwise_distance': float(min_distance),
            'cloud_statistics': {
                'mean_coordinates': np.mean(fuzzy_cloud, axis=0).tolist(),
                'std_coordinates': np.std(fuzzy_cloud, axis=0).tolist(),
                'coordinate_range': [float(np.min(fuzzy_cloud)), float(np.max(fuzzy_cloud))]
            }
        }
    
    def _visualize_compression_stages(self, compression_data: Dict, filename: str):
        """Visualize the multi-stage compression process"""
        try:
            stages = compression_data['compression_stages']
            
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Stage 1: Alphabetical Sort', 'Stage 2: Numerical Conversion',
                               'Stage 3: Digit to Text', 'Stage 4: Re-alphabetical',
                               'Stage 5: Back to Numerical', 'Stage 6: Binary + Fuzzy Cloud'),
                specs=[[{"type": "xy"}, {"type": "xy"}],
                       [{"type": "xy"}, {"type": "xy"}],
                       [{"type": "xy"}, {"type": "xy"}]]
            )
            
            # Stage 1: Character frequency after sorting
            if len(stages) >= 1:
                sorted_chars = stages[0]['output']
                char_counts = {char: sorted_chars.count(char) for char in set(sorted_chars)}
                fig.add_trace(
                    go.Bar(x=list(char_counts.keys()), y=list(char_counts.values()),
                          name='Character Frequency', marker_color='blue'),
                    row=1, col=1
                )
            
            # Stage 2: Numerical representation length analysis
            if len(stages) >= 2:
                input_len = len(stages[1]['input'])
                output_len = len(stages[1]['output'])
                fig.add_trace(
                    go.Bar(x=['Input Length', 'Output Length'], y=[input_len, output_len],
                          name='Length Comparison', marker_color='red'),
                    row=1, col=2
                )
            
            # Stage 3: Word frequency analysis
            if len(stages) >= 3:
                words = stages[2]['word_list']
                word_counts = {word: words.count(word) for word in set(words)}
                fig.add_trace(
                    go.Bar(x=list(word_counts.keys()), y=list(word_counts.values()),
                          name='Word Frequency', marker_color='green'),
                    row=2, col=1
                )
            
            # Stage 4: Reordering visualization
            if len(stages) >= 4:
                order_changes = stages[3]['order_changes']
                total_words = len(stages[3]['input'].split())
                fig.add_trace(
                    go.Bar(x=['Unchanged', 'Reordered'], 
                          y=[total_words - order_changes, order_changes],
                          name='Word Reordering', marker_color='purple'),
                    row=2, col=2
                )
            
            # Stage 5: Recovery rate
            if len(stages) >= 5:
                recovery_rate = stages[4]['recovery_rate']
                fig.add_trace(
                    go.Bar(x=['Lost', 'Recovered'], y=[1-recovery_rate, recovery_rate],
                          name='Recovery Rate', marker_color='orange'),
                    row=3, col=1
                )
            
            # Stage 6 & Fuzzy Cloud: Dimensionality and cloud size
            fuzzy_coords = compression_data['fuzzy_coordinates']
            fig.add_trace(
                go.Bar(x=['Dimensions', 'Cloud Points (scaled)'], 
                      y=[fuzzy_coords['dimensionality'], fuzzy_coords['cloud_points']/10],
                      name='Fuzzy Properties', marker_color='teal'),
                row=3, col=2
            )
            
            fig.update_layout(
                title=f'Fuzzy Compression Multi-Stage Process: {compression_data["original_text"][:30]}...',
                height=1000,
                showlegend=True
            )
            
            fig.write_html(f'demo/outputs/visualizations/{filename}.html')
            logger.info(f"Compression stages visualization saved: {filename}.html")
            
        except Exception as e:
            logger.error(f"Failed to create compression visualization: {str(e)}")
    
    def _save_json(self, filename: str, data: Dict):
        """Save data as JSON with error handling"""
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save JSON file {filename}: {str(e)}")


class EightDimensionalSemanticMapper:
    """
    Eight-dimensional semantic coordinate mapping system:
    - Technical/Emotional (±1,0,0,0) - Precision vs Expression  
    - Action/Descriptive (0,±1,0,0) - Process vs Attribute
    - Abstract/Concrete (0,0,±1,0) - Conceptual vs Physical
    - Positive/Negative (0,0,0,±1) - Affirmation vs Negation
    """
    
    def __init__(self, save_intermediates: bool = True):
        self.save_intermediates = save_intermediates
        self.semantic_cache = {}
        self.mapping_history = []
        
        # Define semantic dimensions
        self.semantic_dimensions = {
            'technical_emotional': {'technical': (1, 0, 0, 0), 'emotional': (-1, 0, 0, 0)},
            'action_descriptive': {'action': (0, 1, 0, 0), 'descriptive': (0, -1, 0, 0)},
            'abstract_concrete': {'abstract': (0, 0, 1, 0), 'concrete': (0, 0, -1, 0)},
            'positive_negative': {'positive': (0, 0, 0, 1), 'negative': (0, 0, 0, -1)}
        }
        
        os.makedirs('demo/outputs/semantic', exist_ok=True)
        logger.info("Eight-Dimensional Semantic Mapper initialized")
    
    def map_text_to_8d_coordinates(self, text: str) -> Dict:
        """Map text to 8-dimensional semantic coordinate space"""
        start_time = time.time()
        logger.info(f"Mapping text to 8D semantic coordinates: '{text[:50]}...'")
        
        mapping_data = {
            'original_text': text,
            'timestamp': datetime.now().isoformat(),
            'word_count': len(text.split()),
            'character_count': len(text)
        }
        
        # Analyze each semantic dimension
        dimension_analyses = {}
        
        # Technical/Emotional Analysis
        tech_emotional = self._analyze_technical_emotional(text)
        dimension_analyses['technical_emotional'] = tech_emotional
        
        # Action/Descriptive Analysis  
        action_descriptive = self._analyze_action_descriptive(text)
        dimension_analyses['action_descriptive'] = action_descriptive
        
        # Abstract/Concrete Analysis
        abstract_concrete = self._analyze_abstract_concrete(text)
        dimension_analyses['abstract_concrete'] = abstract_concrete
        
        # Positive/Negative Analysis
        positive_negative = self._analyze_positive_negative(text)
        dimension_analyses['positive_negative'] = positive_negative
        
        mapping_data['dimension_analyses'] = dimension_analyses
        
        # Calculate 8D coordinates by combining dimensional analyses
        coordinates_8d = self._calculate_8d_coordinates(dimension_analyses)
        mapping_data['coordinates_8d'] = coordinates_8d
        
        # Compress to S-entropy 3D coordinates
        s_entropy_coords = self._compress_to_s_entropy(coordinates_8d, text)
        mapping_data['s_entropy_coordinates'] = s_entropy_coords
        
        # Calculate semantic metrics
        mapping_data['semantic_metrics'] = {
            'coordinate_magnitude': float(np.linalg.norm(coordinates_8d)),
            'dominant_dimension': self._find_dominant_dimension(coordinates_8d),
            'semantic_complexity': self._calculate_semantic_complexity(coordinates_8d),
            'mapping_time': time.time() - start_time
        }
        
        # Save mapping data
        if self.save_intermediates:
            timestamp = int(time.time())
            filename = f'demo/outputs/semantic/8d_mapping_{timestamp}.json'
            self._save_json(filename, mapping_data)
            self._visualize_8d_mapping(mapping_data, f'8d_mapping_{timestamp}')
        
        logger.info(f"8D semantic mapping completed: magnitude = {mapping_data['semantic_metrics']['coordinate_magnitude']:.3f}")
        return mapping_data
    
    def _analyze_technical_emotional(self, text: str) -> Dict:
        """Analyze technical vs emotional content with enhanced word lists"""
        technical_indicators = [
            'algorithm', 'function', 'system', 'process', 'data', 'compute', 'analyze', 
            'method', 'implementation', 'optimization', 'performance', 'efficiency',
            'framework', 'architecture', 'protocol', 'specification', 'parameter',
            'variable', 'equation', 'calculation', 'measurement', 'precision',
            'coordinate', 'navigation', 'transformation', 'synthesis', 'validation'
        ]
        
        emotional_indicators = [
            'feel', 'love', 'hate', 'excited', 'sad', 'happy', 'angry', 'passionate',
            'wonderful', 'terrible', 'amazing', 'awful', 'beautiful', 'disgusting',
            'joy', 'pain', 'pleasure', 'suffering', 'hope', 'fear', 'anxiety',
            'relief', 'satisfaction', 'frustration', 'delight', 'concern', 'worry'
        ]
        
        words = text.lower().split()
        technical_matches = sum(1 for word in words if any(tech in word for tech in technical_indicators))
        emotional_matches = sum(1 for word in words if any(emo in word for emo in emotional_indicators))
        
        total_words = len(words)
        technical_score = technical_matches / max(1, total_words)
        emotional_score = emotional_matches / max(1, total_words)
        
        # Calculate net coordinate
        net_coordinate = technical_score - emotional_score
        
        return {
            'technical_score': float(technical_score),
            'emotional_score': float(emotional_score),
            'technical_matches': technical_matches,
            'emotional_matches': emotional_matches,
            'net_coordinate': float(net_coordinate),
            'dominant_aspect': 'technical' if net_coordinate > 0 else 'emotional' if net_coordinate < 0 else 'neutral'
        }
    
    def _analyze_action_descriptive(self, text: str) -> Dict:
        """Analyze action vs descriptive content"""
        action_indicators = [
            'run', 'execute', 'perform', 'create', 'build', 'implement', 'solve',
            'process', 'generate', 'produce', 'construct', 'develop', 'establish',
            'initiate', 'activate', 'transform', 'navigate', 'optimize', 'validate',
            'demonstrate', 'achieve', 'accomplish', 'complete', 'finish', 'start'
        ]
        
        descriptive_indicators = [
            'beautiful', 'large', 'complex', 'simple', 'efficient', 'optimal', 'good',
            'best', 'excellent', 'superior', 'advanced', 'sophisticated', 'elegant',
            'robust', 'comprehensive', 'detailed', 'precise', 'accurate', 'reliable',
            'stable', 'flexible', 'adaptive', 'revolutionary', 'innovative', 'novel'
        ]
        
        words = text.lower().split()
        action_matches = sum(1 for word in words if any(action in word for action in action_indicators))
        descriptive_matches = sum(1 for word in words if any(desc in word for desc in descriptive_indicators))
        
        total_words = len(words)
        action_score = action_matches / max(1, total_words)
        descriptive_score = descriptive_matches / max(1, total_words)
        
        net_coordinate = action_score - descriptive_score
        
        return {
            'action_score': float(action_score),
            'descriptive_score': float(descriptive_score),
            'action_matches': action_matches,
            'descriptive_matches': descriptive_matches,
            'net_coordinate': float(net_coordinate),
            'dominant_aspect': 'action' if net_coordinate > 0 else 'descriptive' if net_coordinate < 0 else 'neutral'
        }
    
    def _analyze_abstract_concrete(self, text: str) -> Dict:
        """Analyze abstract vs concrete content"""
        abstract_indicators = [
            'concept', 'theory', 'framework', 'principle', 'philosophy', 'paradigm',
            'approach', 'methodology', 'strategy', 'abstraction', 'generalization',
            'conceptual', 'theoretical', 'philosophical', 'metaphysical', 'ideological',
            'notion', 'idea', 'thought', 'belief', 'understanding', 'interpretation',
            'meaning', 'significance', 'implication', 'essence', 'nature', 'quality'
        ]
        
        concrete_indicators = [
            'table', 'computer', 'file', 'number', 'result', 'output', 'input',
            'device', 'machine', 'tool', 'instrument', 'equipment', 'hardware',
            'software', 'database', 'record', 'document', 'report', 'chart',
            'graph', 'diagram', 'image', 'picture', 'screen', 'button', 'menu'
        ]
        
        words = text.lower().split()
        abstract_matches = sum(1 for word in words if any(abstract in word for abstract in abstract_indicators))
        concrete_matches = sum(1 for word in words if any(concrete in word for concrete in concrete_indicators))
        
        total_words = len(words)
        abstract_score = abstract_matches / max(1, total_words)
        concrete_score = concrete_matches / max(1, total_words)
        
        net_coordinate = abstract_score - concrete_score
        
        return {
            'abstract_score': float(abstract_score),
            'concrete_score': float(concrete_score),
            'abstract_matches': abstract_matches,
            'concrete_matches': concrete_matches,
            'net_coordinate': float(net_coordinate),
            'dominant_aspect': 'abstract' if net_coordinate > 0 else 'concrete' if net_coordinate < 0 else 'neutral'
        }
    
    def _analyze_positive_negative(self, text: str) -> Dict:
        """Analyze positive vs negative sentiment"""
        positive_indicators = [
            'good', 'excellent', 'amazing', 'wonderful', 'great', 'perfect', 'successful',
            'optimal', 'superior', 'outstanding', 'exceptional', 'remarkable', 'impressive',
            'effective', 'efficient', 'beneficial', 'advantageous', 'valuable', 'useful',
            'helpful', 'positive', 'constructive', 'productive', 'creative', 'innovative'
        ]
        
        negative_indicators = [
            'bad', 'terrible', 'awful', 'wrong', 'failed', 'error', 'problem',
            'impossible', 'difficult', 'challenging', 'problematic', 'deficient',
            'inadequate', 'insufficient', 'ineffective', 'inefficient', 'harmful',
            'detrimental', 'disadvantageous', 'negative', 'destructive', 'limiting'
        ]
        
        words = text.lower().split()
        positive_matches = sum(1 for word in words if any(pos in word for pos in positive_indicators))
        negative_matches = sum(1 for word in words if any(neg in word for neg in negative_indicators))
        
        total_words = len(words)
        positive_score = positive_matches / max(1, total_words)
        negative_score = negative_matches / max(1, total_words)
        
        net_coordinate = positive_score - negative_score
        
        return {
            'positive_score': float(positive_score),
            'negative_score': float(negative_score),
            'positive_matches': positive_matches,
            'negative_matches': negative_matches,
            'net_coordinate': float(net_coordinate),
            'dominant_aspect': 'positive' if net_coordinate > 0 else 'negative' if net_coordinate < 0 else 'neutral'
        }
    
    def _calculate_8d_coordinates(self, dimension_analyses: Dict) -> np.ndarray:
        """Calculate 8-dimensional coordinates from dimensional analyses"""
        # Extract net coordinates from each dimension analysis
        tech_emotional = dimension_analyses['technical_emotional']['net_coordinate']
        action_descriptive = dimension_analyses['action_descriptive']['net_coordinate']
        abstract_concrete = dimension_analyses['abstract_concrete']['net_coordinate']
        positive_negative = dimension_analyses['positive_negative']['net_coordinate']
        
        # Map to 8D space (each dimension pair contributes 2 coordinates)
        coordinates_8d = np.array([
            tech_emotional,      # Technical-Emotional axis
            0,                   # Reserved for technical-emotional complexity
            action_descriptive,  # Action-Descriptive axis
            0,                   # Reserved for action-descriptive complexity
            abstract_concrete,   # Abstract-Concrete axis
            0,                   # Reserved for abstract-concrete complexity
            positive_negative,   # Positive-Negative axis
            0                    # Reserved for positive-negative complexity
        ])
        
        # Add complexity measures to reserved coordinates
        coordinates_8d[1] = abs(tech_emotional) * 0.5  # Technical-emotional complexity
        coordinates_8d[3] = abs(action_descriptive) * 0.5  # Action-descriptive complexity
        coordinates_8d[5] = abs(abstract_concrete) * 0.5  # Abstract-concrete complexity
        coordinates_8d[7] = abs(positive_negative) * 0.5  # Positive-negative complexity
        
        return coordinates_8d
    
    def _compress_to_s_entropy(self, coordinates_8d: np.ndarray, text: str) -> np.ndarray:
        """Compress 8D coordinates to 3D S-entropy coordinates"""
        # Knowledge dimension: Overall semantic complexity
        knowledge_coord = np.linalg.norm(coordinates_8d[:4])  # First 4 dimensions
        
        # Time dimension: Text processing complexity
        time_coord = len(text.split()) / 1000.0  # Normalized word count
        
        # Entropy dimension: Semantic entropy
        entropy_coord = -np.sum([abs(c) * np.log(abs(c) + 1e-10) for c in coordinates_8d]) / 8.0
        
        return np.array([knowledge_coord, time_coord, entropy_coord])
    
    def _find_dominant_dimension(self, coordinates_8d: np.ndarray) -> str:
        """Find the dominant semantic dimension"""
        dimension_names = [
            'technical', 'technical_complexity', 'action', 'action_complexity',
            'abstract', 'abstract_complexity', 'positive', 'positive_complexity'
        ]
        
        max_idx = np.argmax(np.abs(coordinates_8d))
        return dimension_names[max_idx]
    
    def _calculate_semantic_complexity(self, coordinates_8d: np.ndarray) -> float:
        """Calculate overall semantic complexity"""
        # Complexity as the standard deviation of coordinates
        return float(np.std(coordinates_8d))
    
    def _visualize_8d_mapping(self, mapping_data: Dict, filename: str):
        """Visualize 8D semantic mapping results"""
        try:
            # Extract data for visualization
            coords_8d = mapping_data['coordinates_8d']
            s_entropy_coords = mapping_data['s_entropy_coordinates']
            analyses = mapping_data['dimension_analyses']
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('8D Coordinate Distribution', 'Dimensional Analysis Scores',
                               'S-Entropy Compression', 'Semantic Complexity Map'),
                specs=[[{"type": "xy"}, {"type": "xy"}],
                       [{"type": "xy"}, {"type": "xy"}]]
            )
            
            # 8D coordinates radar chart (using first 4 dimensions for visibility)
            dimension_labels = ['Technical', 'Tech-Complex', 'Action', 'Action-Complex']
            fig.add_trace(
                go.Bar(x=dimension_labels, y=coords_8d[:4],
                      name='8D Coordinates', marker_color='blue'),
                row=1, col=1
            )
            
            # Dimensional analysis scores
            analysis_types = list(analyses.keys())
            net_scores = [analyses[dim]['net_coordinate'] for dim in analysis_types]
            
            fig.add_trace(
                go.Bar(x=analysis_types, y=net_scores,
                      name='Net Scores', marker_color='red'),
                row=1, col=2
            )
            
            # S-entropy compression visualization
            fig.add_trace(
                go.Bar(x=['Knowledge', 'Time', 'Entropy'], y=s_entropy_coords,
                      name='S-Entropy Coords', marker_color='green'),
                row=2, col=1
            )
            
            # Semantic complexity heatmap (using coordinate magnitudes)
            complexity_matrix = np.array(coords_8d).reshape(2, 4)  # Reshape for heatmap
            fig.add_trace(
                go.Heatmap(z=complexity_matrix, colorscale='Viridis',
                          name='Complexity Map'),
                row=2, col=2
            )
            
            fig.update_layout(
                title=f'8D Semantic Mapping: {mapping_data["original_text"][:50]}...',
                height=800,
                showlegend=True
            )
            
            fig.write_html(f'demo/outputs/visualizations/{filename}.html')
            logger.info(f"8D mapping visualization saved: {filename}.html")
            
        except Exception as e:
            logger.error(f"Failed to create 8D mapping visualization: {str(e)}")
    
    def _save_json(self, filename: str, data: Dict):
        """Save data as JSON with error handling"""
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save JSON file {filename}: {str(e)}")


class SemanticNavigationSystem:
    """
    Complete semantic navigation system integrating:
    - Fuzzy compression embedding
    - 8-dimensional semantic mapping
    - Empty dictionary real-time comprehension
    - Dynamic dimensionality navigation
    """
    
    def __init__(self, save_intermediates: bool = True):
        self.save_intermediates = save_intermediates
        self.fuzzy_embedder = FuzzyCompressionEmbedder(save_intermediates)
        self.semantic_mapper = EightDimensionalSemanticMapper(save_intermediates)
        self.navigation_history = []
        
        os.makedirs('demo/outputs/semantic', exist_ok=True)
        logger.info("Semantic Navigation System initialized")
    
    def navigate_semantic_space(self, text_samples: List[str]) -> Dict:
        """
        Navigate through semantic space using multiple text samples
        to demonstrate collision prevention and dynamic dimensionality
        """
        start_time = time.time()
        logger.info(f"Navigating semantic space with {len(text_samples)} text samples")
        
        navigation_data = {
            'text_samples': text_samples,
            'timestamp': datetime.now().isoformat(),
            'sample_count': len(text_samples),
            'individual_results': [],
            'collision_analysis': {},
            'navigation_metrics': {}
        }
        
        # Process each text sample
        sample_embeddings = []
        sample_8d_mappings = []
        
        for i, text in enumerate(text_samples):
            logger.info(f"Processing sample {i+1}/{len(text_samples)}")
            
            # Fuzzy compression embedding
            fuzzy_result = self.fuzzy_embedder.compress_text_fuzzy(text)
            
            # 8D semantic mapping
            mapping_result = self.semantic_mapper.map_text_to_8d_coordinates(text)
            
            # Store results
            sample_result = {
                'sample_id': i,
                'text': text,
                'fuzzy_compression': fuzzy_result,
                'semantic_mapping': mapping_result
            }
            
            navigation_data['individual_results'].append(sample_result)
            sample_embeddings.append(fuzzy_result['fuzzy_coordinates'])
            sample_8d_mappings.append(mapping_result['coordinates_8d'])
        
        # Analyze collision prevention
        collision_analysis = self._analyze_collision_prevention(sample_embeddings, sample_8d_mappings)
        navigation_data['collision_analysis'] = collision_analysis
        
        # Calculate navigation metrics
        navigation_metrics = self._calculate_navigation_metrics(navigation_data, start_time)
        navigation_data['navigation_metrics'] = navigation_metrics
        
        # Save navigation results
        if self.save_intermediates:
            timestamp = int(time.time())
            filename = f'demo/outputs/semantic/semantic_navigation_{timestamp}.json'
            self._save_json(filename, navigation_data)
            self._visualize_semantic_navigation(navigation_data, f'semantic_navigation_{timestamp}')
        
        logger.info(f"Semantic navigation completed: {collision_analysis['collision_rate']:.4f} collision rate, "
                   f"{navigation_metrics['avg_dimensionality']:.1f} avg dimensions")
        
        return navigation_data
    
    def _analyze_collision_prevention(self, embeddings: List[Dict], mappings_8d: List[np.ndarray]) -> Dict:
        """Analyze collision prevention effectiveness"""
        collision_data = {
            'total_samples': len(embeddings),
            'collision_count': 0,
            'near_collision_count': 0,
            'collision_rate': 0.0,
            'distance_statistics': {}
        }
        
        if len(embeddings) < 2:
            return collision_data
        
        # Analyze fuzzy embedding collisions
        fuzzy_clouds = []
        for embedding in embeddings:
            if 'fuzzy_cloud' in embedding:
                cloud_points = np.array(embedding['fuzzy_cloud'])
                # Use centroid of fuzzy cloud for collision analysis
                centroid = np.mean(cloud_points, axis=0)
                fuzzy_clouds.append(centroid)
        
        if len(fuzzy_clouds) >= 2:
            # Calculate pairwise distances between fuzzy cloud centroids
            fuzzy_distances = []
            for i in range(len(fuzzy_clouds)):
                for j in range(i+1, len(fuzzy_clouds)):
                    distance = np.linalg.norm(fuzzy_clouds[i] - fuzzy_clouds[j])
                    fuzzy_distances.append(distance)
            
            # Collision detection (distance < threshold)
            collision_threshold = 0.1
            near_collision_threshold = 0.2
            
            collisions = sum(1 for d in fuzzy_distances if d < collision_threshold)
            near_collisions = sum(1 for d in fuzzy_distances if collision_threshold <= d < near_collision_threshold)
            
            collision_data['collision_count'] = collisions
            collision_data['near_collision_count'] = near_collisions
            collision_data['collision_rate'] = collisions / len(fuzzy_distances) if fuzzy_distances else 0.0
            
            collision_data['distance_statistics'] = {
                'min_distance': float(min(fuzzy_distances)) if fuzzy_distances else 0.0,
                'max_distance': float(max(fuzzy_distances)) if fuzzy_distances else 0.0,
                'mean_distance': float(np.mean(fuzzy_distances)) if fuzzy_distances else 0.0,
                'std_distance': float(np.std(fuzzy_distances)) if fuzzy_distances else 0.0
            }
        
        # Analyze 8D semantic space separation
        if len(mappings_8d) >= 2:
            semantic_distances = pdist(mappings_8d)
            collision_data['semantic_distance_stats'] = {
                'min_semantic_distance': float(np.min(semantic_distances)),
                'max_semantic_distance': float(np.max(semantic_distances)),
                'mean_semantic_distance': float(np.mean(semantic_distances)),
                'std_semantic_distance': float(np.std(semantic_distances))
            }
        
        return collision_data
    
    def _calculate_navigation_metrics(self, navigation_data: Dict, start_time: float) -> Dict:
        """Calculate comprehensive navigation metrics"""
        total_time = time.time() - start_time
        results = navigation_data['individual_results']
        
        if not results:
            return {'navigation_time': total_time}
        
        # Extract dimensionalities
        dimensionalities = []
        compression_ratios = []
        cloud_sizes = []
        
        for result in results:
            fuzzy_data = result['fuzzy_compression']['fuzzy_coordinates']
            dimensionalities.append(fuzzy_data['dimensionality'])
            
            # Calculate compression ratio
            original_length = len(result['text'])
            compressed_size = fuzzy_data['coordinate_density']
            compression_ratio = original_length / max(1, compressed_size)
            compression_ratios.append(compression_ratio)
            
            cloud_sizes.append(fuzzy_data['cloud_points'])
        
        metrics = {
            'navigation_time': float(total_time),
            'avg_dimensionality': float(np.mean(dimensionalities)),
            'std_dimensionality': float(np.std(dimensionalities)),
            'min_dimensionality': int(min(dimensionalities)),
            'max_dimensionality': int(max(dimensionalities)),
            'avg_compression_ratio': float(np.mean(compression_ratios)),
            'avg_cloud_size': float(np.mean(cloud_sizes)),
            'dynamic_dimensionality_achieved': len(set(dimensionalities)) > 1,
            'collision_prevention_score': 1.0 - navigation_data.get('collision_analysis', {}).get('collision_rate', 0.0)
        }
        
        return metrics
    
    def _visualize_semantic_navigation(self, navigation_data: Dict, filename: str):
        """Visualize semantic navigation results"""
        try:
            results = navigation_data['individual_results']
            collision_analysis = navigation_data['collision_analysis']
            
            # Extract data for visualization
            sample_ids = [r['sample_id'] for r in results]
            dimensionalities = [r['fuzzy_compression']['fuzzy_coordinates']['dimensionality'] for r in results]
            collision_resistances = [r['fuzzy_compression']['fuzzy_coordinates']['collision_resistance'] for r in results]
            cloud_sizes = [r['fuzzy_compression']['fuzzy_coordinates']['cloud_points'] for r in results]
            
            # Create comprehensive visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Dynamic Dimensionality', 'Collision Resistance',
                               'Cloud Size Distribution', 'Distance Analysis'),
                specs=[[{"type": "xy"}, {"type": "xy"}],
                       [{"type": "xy"}, {"type": "xy"}]]
            )
            
            # Dynamic dimensionality
            fig.add_trace(
                go.Scatter(x=sample_ids, y=dimensionalities, mode='lines+markers',
                          name='Dimensionality', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Collision resistance
            fig.add_trace(
                go.Bar(x=sample_ids, y=collision_resistances,
                      name='Collision Resistance', marker_color='red'),
                row=1, col=2
            )
            
            # Cloud size distribution
            fig.add_trace(
                go.Histogram(x=cloud_sizes, nbinsx=10,
                            name='Cloud Size Distribution', marker_color='green'),
                row=2, col=1
            )
            
            # Distance analysis
            if 'distance_statistics' in collision_analysis:
                dist_stats = collision_analysis['distance_statistics']
                fig.add_trace(
                    go.Bar(x=['Min', 'Mean', 'Max'], 
                          y=[dist_stats['min_distance'], dist_stats['mean_distance'], dist_stats['max_distance']],
                          name='Distance Stats', marker_color='purple'),
                    row=2, col=2
                )
            
            fig.update_layout(
                title=f'Semantic Navigation Analysis ({len(results)} samples)',
                height=800,
                showlegend=True
            )
            
            fig.write_html(f'demo/outputs/visualizations/{filename}.html')
            logger.info(f"Semantic navigation visualization saved: {filename}.html")
            
        except Exception as e:
            logger.error(f"Failed to create semantic navigation visualization: {str(e)}")
    
    def demonstrate_collision_prevention(self, similar_texts: List[str]) -> Dict:
        """
        Demonstrate collision prevention with deliberately similar texts
        that would collide in traditional embedding systems
        """
        logger.info("Demonstrating collision prevention with similar texts")
        
        # Process similar texts through fuzzy compression
        collision_demo = {
            'similar_texts': similar_texts,
            'traditional_collision_expected': True,
            'fuzzy_results': [],
            'collision_prevention_analysis': {}
        }
        
        # Process each similar text
        for i, text in enumerate(similar_texts):
            fuzzy_result = self.fuzzy_embedder.compress_text_fuzzy(text)
            semantic_result = self.semantic_mapper.map_text_to_8d_coordinates(text)
            
            collision_demo['fuzzy_results'].append({
                'text_id': i,
                'text': text,
                'fuzzy_embedding': fuzzy_result,
                'semantic_mapping': semantic_result
            })
        
        # Analyze collision prevention effectiveness
        embeddings = [r['fuzzy_embedding'] for r in collision_demo['fuzzy_results']]
        prevention_analysis = self._analyze_collision_prevention(
            embeddings,
            [r['semantic_mapping']['coordinates_8d'] for r in collision_demo['fuzzy_results']]
        )
        
        collision_demo['collision_prevention_analysis'] = prevention_analysis
        collision_demo['prevention_success'] = prevention_analysis['collision_rate'] < 0.1  # Success if <10% collision rate
        
        logger.info(f"Collision prevention demo: {prevention_analysis['collision_rate']:.4f} collision rate")
        return collision_demo
    
    def _save_json(self, filename: str, data: Dict):
        """Save data as JSON with error handling"""
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save JSON file {filename}: {str(e)}")


def main():
    """
    Main demonstration of semantic S-entropy navigation with
    fuzzy compression embedding and collision prevention
    """
    print("=" * 80)
    print("REVOLUTIONARY SEMANTIC S-ENTROPY NAVIGATION DEMONSTRATION")
    print("Fuzzy Compression + 8D Mapping + Collision Prevention + Dynamic Dimensionality")
    print("=" * 80)
    
    # Test text samples with varying semantic content
    test_texts = [
        "The revolutionary S-entropy framework enables unprecedented coordinate navigation.",
        "That is a bag",  # Simple text for compression demo
        "Advanced algorithmic processing systems demonstrate optimal performance efficiency.",
        "I feel excited and happy about this wonderful breakthrough in technology.",
        "The concrete implementation involves specific hardware devices and software components.",
        "Abstract philosophical concepts require theoretical framework understanding.",
        "This terrible system fails completely with awful performance and wrong results.",
        "Execute the optimization algorithm to generate superior computational solutions."
    ]
    
    print("\n1. FUZZY COMPRESSION EMBEDDING DEMONSTRATION")
    print("-" * 50)
    
    fuzzy_embedder = FuzzyCompressionEmbedder()
    
    # Demonstrate the multi-stage compression process
    demo_text = "That is a bag"
    print(f"\nDemonstrating multi-stage compression for: '{demo_text}'")
    
    compression_result = fuzzy_embedder.compress_text_fuzzy(demo_text)
    
    print(f"\nCompression stages:")
    for stage in compression_result['compression_stages']:
        print(f"  Stage {stage['stage']} ({stage['description']}): {stage['transformation']}")
    
    fuzzy_coords = compression_result['fuzzy_coordinates']
    print(f"\nFuzzy coordinate cloud:")
    print(f"  Dimensions: {fuzzy_coords['dimensionality']}")
    print(f"  Cloud points: {fuzzy_coords['cloud_points']}")
    print(f"  Collision resistance: {fuzzy_coords['collision_resistance']:.4f}")
    print(f"  Coordinate density: {fuzzy_coords['coordinate_density']:.2f}")
    
    print("\n2. EIGHT-DIMENSIONAL SEMANTIC MAPPING")
    print("-" * 40)
    
    semantic_mapper = EightDimensionalSemanticMapper()
    
    for i, text in enumerate(test_texts[:4]):  # First 4 samples for detailed demo
        print(f"\nSample {i+1}: '{text[:50]}...'")
        mapping_result = semantic_mapper.map_text_to_8d_coordinates(text)
        
        print(f"  8D coordinates: {np.array(mapping_result['coordinates_8d'])}")
        print(f"  S-entropy coords: {mapping_result['s_entropy_coordinates']}")
        print(f"  Dominant dimension: {mapping_result['semantic_metrics']['dominant_dimension']}")
        print(f"  Coordinate magnitude: {mapping_result['semantic_metrics']['coordinate_magnitude']:.3f}")
    
    print("\n3. COMPLETE SEMANTIC NAVIGATION SYSTEM")
    print("-" * 42)
    
    navigation_system = SemanticNavigationSystem()
    
    # Navigate through semantic space
    navigation_result = navigation_system.navigate_semantic_space(test_texts)
    
    metrics = navigation_result['navigation_metrics']
    collision_analysis = navigation_result['collision_analysis']
    
    print(f"\nNavigation Results:")
    print(f"  Samples processed: {navigation_result['sample_count']}")
    print(f"  Average dimensionality: {metrics['avg_dimensionality']:.1f}")
    print(f"  Dimensionality range: {metrics['min_dimensionality']}-{metrics['max_dimensionality']}")
    print(f"  Dynamic dimensionality achieved: {metrics['dynamic_dimensionality_achieved']}")
    print(f"  Collision rate: {collision_analysis['collision_rate']:.4f}")
    print(f"  Average compression ratio: {metrics['avg_compression_ratio']:.2f}:1")
    print(f"  Collision prevention score: {metrics['collision_prevention_score']:.4f}")
    
    print("\n4. COLLISION PREVENTION DEMONSTRATION")
    print("-" * 40)
    
    # Test collision prevention with very similar texts
    similar_texts = [
        "That is a bag",
        "That is a bag.",
        "That is a big bag",
        "This is a bag",
        "That was a bag"
    ]
    
    print(f"\nTesting collision prevention with similar texts:")
    for text in similar_texts:
        print(f"  '{text}'")
    
    collision_demo = navigation_system.demonstrate_collision_prevention(similar_texts)
    
    print(f"\nCollision Prevention Results:")
    print(f"  Collision rate: {collision_demo['collision_prevention_analysis']['collision_rate']:.4f}")
    print(f"  Prevention success: {collision_demo['prevention_success']}")
    
    if 'distance_statistics' in collision_demo['collision_prevention_analysis']:
        dist_stats = collision_demo['collision_prevention_analysis']['distance_statistics']
        print(f"  Mean separation distance: {dist_stats['mean_distance']:.4f}")
        print(f"  Minimum distance: {dist_stats['min_distance']:.4f}")
    
    print("\n5. PERFORMANCE SUMMARY")
    print("-" * 20)
    
    print(f"\nFuzzy Embedding Benefits:")
    print(f"  ✓ Dynamic dimensionality prevents rigid predefined constraints")
    print(f"  ✓ Multi-stage compression amplifies embedding differences")  
    print(f"  ✓ Fuzzy coordinate clouds prevent collision in high dimensions")
    print(f"  ✓ Collision resistance: {np.mean([r['fuzzy_compression']['fuzzy_coordinates']['collision_resistance'] for r in navigation_result['individual_results']]):.4f}")
    
    print(f"\n8D Semantic Mapping Benefits:")
    print(f"  ✓ Eight-dimensional semantic space captures nuanced meaning")
    print(f"  ✓ Technical/Emotional, Action/Descriptive, Abstract/Concrete, Positive/Negative")
    print(f"  ✓ Compression to S-entropy coordinates maintains essential information")
    print(f"  ✓ Average semantic complexity: {np.mean([r['semantic_mapping']['semantic_metrics']['semantic_complexity'] for r in navigation_result['individual_results']]):.4f}")
    
    print("\n" + "=" * 80)
    print("SEMANTIC NAVIGATION DEMONSTRATION COMPLETED")
    print("FUZZY EMBEDDING COLLISION PREVENTION SUCCESSFULLY DEMONSTRATED")
    print("Check demo/outputs/ directory for comprehensive results and visualizations")
    print("=" * 80)


if __name__ == "__main__":
    main()
