#!/usr/bin/env python3
"""
Process Visualization Examples

This script processes the visualization examples from various libraries (D3, React, etc.)
to create a training dataset for the visualization model. It:
1. Extracts zip files
2. Processes examples to create structured training data
3. Creates a specialized dataset for visualization generation

Usage:
    python -m purpose.examples.visualization.process_viz_examples
"""

import os
import json
import logging
import zipfile
import glob
from pathlib import Path
import shutil
from typing import List, Dict, Any, Optional
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("viz_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("viz-processing")

class VisualizationProcessor:
    """Process visualization examples to create training data."""
    
    def __init__(
        self,
        input_dir: str = "content/visualisation",
        output_dir: str = "data/visualization",
        extract_dir: str = "temp/viz_examples"
    ):
        """Initialize the processor."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.extract_dir = Path(extract_dir)
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.extract_dir, exist_ok=True)
        
        # Define visualization categories
        self.viz_categories = {
            "statistical": ["histogram", "boxplot", "violin", "scatter", "correlation"],
            "temporal": ["line", "area", "timeline", "calendar"],
            "hierarchical": ["tree", "treemap", "sunburst", "circle-packing"],
            "network": ["force", "sankey", "chord", "arc"],
            "geographic": ["map", "choropleth", "projection"],
            "custom": ["sports", "athlete", "performance", "biomechanics"]
        }
    
    def extract_zip_files(self):
        """Extract all zip files in the input directory."""
        zip_files = list(self.input_dir.glob("*.zip"))
        logger.info(f"Found {len(zip_files)} zip files to extract")
        
        for zip_path in zip_files:
            try:
                extract_path = self.extract_dir / zip_path.stem
                logger.info(f"Extracting {zip_path.name} to {extract_path}")
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                
            except Exception as e:
                logger.error(f"Error extracting {zip_path.name}: {str(e)}")
    
    def _get_file_content(self, file_path: Path) -> Optional[str]:
        """Safely read file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
            return None
    
    def _categorize_visualization(self, content: str, file_path: str) -> List[str]:
        """Categorize visualization based on content and path."""
        categories = []
        
        # Check content for keywords
        content_lower = content.lower()
        for category, keywords in self.viz_categories.items():
            if any(keyword in content_lower or keyword in str(file_path).lower() 
                  for keyword in keywords):
                categories.append(category)
        
        return categories or ["uncategorized"]
    
    def _extract_dependencies(self, content: str) -> Dict[str, List[str]]:
        """Extract dependencies from visualization code."""
        deps = {
            "d3": [],
            "react": [],
            "other": []
        }
        
        # D3 imports
        d3_imports = re.findall(r'import \* as d3 from ["\']d3["\']|require\(["\']d3["\']\)', content)
        if d3_imports:
            deps["d3"].append("d3")
        
        # D3 module imports
        d3_modules = re.findall(r'import \{([^}]+)\} from ["\']d3-[^"\']+["\']', content)
        for modules in d3_modules:
            deps["d3"].extend([m.strip() for m in modules.split(",")])
        
        # React imports
        react_imports = re.findall(r'import [^;]+ from ["\']react[^"\']*["\']', content)
        if react_imports:
            deps["react"].append("react")
        
        # Other imports
        other_imports = re.findall(r'import [^;]+ from ["\']([^"\']+)["\']', content)
        deps["other"].extend([imp for imp in other_imports 
                            if not imp.startswith(("d3", "react"))])
        
        return deps
    
    def _extract_css_styles(self, content: str) -> Optional[str]:
        """Extract CSS styles from visualization code."""
        css_matches = re.findall(r'<style[^>]*>(.*?)</style>', content, re.DOTALL)
        if css_matches:
            return "\n".join(css_matches)
        
        css_matches = re.findall(r'const styles = {([^}]+)}', content, re.DOTALL)
        if css_matches:
            return "\n".join(css_matches)
        
        return None
    
    def process_visualization_files(self):
        """Process extracted visualization files."""
        examples = []
        
        # File patterns to look for
        patterns = [
            "**/*.html",
            "**/*.js",
            "**/*.jsx",
            "**/*.ts",
            "**/*.tsx"
        ]
        
        for pattern in patterns:
            files = list(self.extract_dir.glob(pattern))
            logger.info(f"Found {len(files)} {pattern} files")
            
            for file_path in files:
                content = self._get_file_content(file_path)
                if not content:
                    continue
                
                # Skip files that don't contain visualization code
                if not any(keyword in content.lower() 
                          for keyword in ["d3", "svg", "canvas", "chart", "plot", "map"]):
                    continue
                
                try:
                    example = {
                        "file_path": str(file_path.relative_to(self.extract_dir)),
                        "file_type": file_path.suffix,
                        "categories": self._categorize_visualization(content, str(file_path)),
                        "dependencies": self._extract_dependencies(content),
                        "code": content,
                        "css": self._extract_css_styles(content)
                    }
                    
                    examples.append(example)
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
        
        return examples
    
    def create_training_dataset(self, examples: List[Dict[str, Any]]):
        """Create training dataset from processed examples."""
        # Save full dataset
        dataset_path = self.output_dir / "visualization_examples.json"
        with open(dataset_path, "w") as f:
            json.dump(examples, f, indent=2)
        
        # Create category-specific datasets
        for category in self.viz_categories:
            category_examples = [
                ex for ex in examples 
                if category in ex["categories"]
            ]
            
            if category_examples:
                category_path = self.output_dir / f"{category}_examples.json"
                with open(category_path, "w") as f:
                    json.dump(category_examples, f, indent=2)
        
        # Create training corpus
        corpus_path = self.output_dir / "training_corpus.jsonl"
        with open(corpus_path, "w") as f:
            for example in examples:
                training_example = {
                    "instruction": f"Create a {', '.join(example['categories'])} visualization",
                    "input": "",
                    "output": example["code"],
                    "metadata": {
                        "categories": example["categories"],
                        "dependencies": example["dependencies"],
                        "file_type": example["file_type"]
                    }
                }
                f.write(json.dumps(training_example) + "\n")
        
        logger.info(f"Created training dataset with {len(examples)} examples")
        logger.info(f"Full dataset saved to {dataset_path}")
        logger.info(f"Training corpus saved to {corpus_path}")
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.extract_dir.exists():
            shutil.rmtree(self.extract_dir)
            logger.info("Cleaned up temporary files")
    
    def run(self):
        """Run the complete visualization processing pipeline."""
        try:
            logger.info("Starting visualization processing pipeline")
            
            # Extract zip files
            self.extract_zip_files()
            
            # Process visualization files
            examples = self.process_visualization_files()
            
            # Create training dataset
            self.create_training_dataset(examples)
            
            # Cleanup
            self.cleanup()
            
            logger.info("Visualization processing pipeline complete!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


if __name__ == "__main__":
    processor = VisualizationProcessor()
    processor.run() 