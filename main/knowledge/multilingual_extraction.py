"""
Multilingual Knowledge Extraction Module

This module implements knowledge extraction capabilities for multiple languages,
enabling extraction from non-English scientific papers and documents.
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
    import spacy
    from transformers import (
        AutoTokenizer, 
        AutoModelForTokenClassification,
        pipeline,
        MarianMTModel, 
        MarianTokenizer
    )
    from langdetect import detect
    from deep_translator import (
        GoogleTranslator, 
        DeeplTranslator, 
        PonsTranslator
    )
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

from purpose.knowledge.enhanced_extraction import EnhancedKnowledgeExtractor

logger = logging.getLogger(__name__)

class MultilingualKnowledgeExtractor:
    """
    Implements knowledge extraction capabilities for multiple languages,
    enabling the processing of non-English scientific papers.
    """
    
    def __init__(
        self, 
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        output_dir: str = "output/multilingual",
        supported_languages: Optional[List[str]] = None,
        translator_type: str = "google",
        marian_model_dir: Optional[str] = None,
        deepl_api_key: Optional[str] = None,
        use_enhanced_extractor: bool = True
    ):
        """
        Initialize the multilingual knowledge extractor.
        
        Args:
            openai_api_key: OpenAI API key (if None, tries to get from environment)
            anthropic_api_key: Anthropic API key (if None, tries to get from environment)
            output_dir: Directory to save extracted knowledge
            supported_languages: List of supported language codes (ISO 639-1)
            translator_type: Type of translator to use (google, deepl, marian, pons)
            marian_model_dir: Directory to store Marian MT models (if using marian)
            deepl_api_key: DeepL API key (if using DeepL)
            use_enhanced_extractor: Whether to use enhanced extraction pipeline
        """
        # Initialize outputs directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize API keys
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.deepl_api_key = deepl_api_key or os.environ.get("DEEPL_API_KEY")
        
        # Initialize OpenAI client
        if self.openai_api_key:
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
            logger.warning("No OpenAI API key provided.")
        
        # Initialize Anthropic client
        self.anthropic_client = None
        if self.anthropic_api_key:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic client: {e}")
        
        # Set up supported languages
        if supported_languages:
            self.supported_languages = supported_languages
        else:
            # Default supported languages
            self.supported_languages = [
                "en", "fr", "de", "es", "it", "pt", "nl", 
                "ru", "zh", "ja", "ko", "ar"
            ]
            
        logger.info(f"Configured with {len(self.supported_languages)} supported languages")
        
        # Set up translators
        self.translator_type = translator_type
        self.marian_model_dir = Path(marian_model_dir) if marian_model_dir else None
        self.translators = {}
        
        if DEPS_AVAILABLE:
            self._initialize_translators()
            
        # Initialize NLP components for language detection and processing
        self.nlp_models = {}
        self.language_detectors = {}
        
        if DEPS_AVAILABLE:
            self._initialize_nlp_components()
            
        # Initialize enhanced extractor if requested
        self.enhanced_extractor = None
        if use_enhanced_extractor:
            self.enhanced_extractor = EnhancedKnowledgeExtractor(
                openai_api_key=openai_api_key,
                anthropic_api_key=anthropic_api_key,
                output_dir=str(self.output_dir / "enhanced")
            )
    
    def _initialize_translators(self) -> None:
        """Initialize translation components for supported languages."""
        try:
            if self.translator_type == "google":
                # Use Google Translator for each language pair
                for lang in self.supported_languages:
                    if lang != "en":
                        self.translators[lang] = {
                            "to_english": GoogleTranslator(source=lang, target='en'),
                            "from_english": GoogleTranslator(source='en', target=lang)
                        }
                        
            elif self.translator_type == "deepl" and self.deepl_api_key:
                # Use DeepL for each language pair
                for lang in self.supported_languages:
                    if lang != "en":
                        self.translators[lang] = {
                            "to_english": DeeplTranslator(
                                api_key=self.deepl_api_key,
                                source=lang,
                                target='en'
                            ),
                            "from_english": DeeplTranslator(
                                api_key=self.deepl_api_key,
                                source='en',
                                target=lang
                            )
                        }
                        
            elif self.translator_type == "marian" and self.marian_model_dir:
                # Use Marian MT models for each language pair
                self.marian_model_dir.mkdir(parents=True, exist_ok=True)
                
                for lang in self.supported_languages:
                    if lang != "en":
                        self._initialize_marian_translators(lang)
                        
            elif self.translator_type == "pons":
                # Use PONS translator for supported language pairs
                supported_pons_langs = ["en", "de", "fr", "es", "it", "pt"]
                for lang in self.supported_languages:
                    if lang in supported_pons_langs and lang != "en":
                        self.translators[lang] = {
                            "to_english": PonsTranslator(source=lang, target='en'),
                            "from_english": PonsTranslator(source='en', target=lang)
                        }
            
            logger.info(f"Initialized translators for {len(self.translators)} languages using {self.translator_type}")
            
        except Exception as e:
            logger.error(f"Error initializing translators: {e}")
    
    def _initialize_marian_translators(self, language: str) -> None:
        """Initialize Marian MT models for a specific language pair."""
        if not DEPS_AVAILABLE:
            return
            
        try:
            # Model naming convention in Hugging Face: opus-mt-{source}-{target}
            to_english_model_name = f"Helsinki-NLP/opus-mt-{language}-en"
            from_english_model_name = f"Helsinki-NLP/opus-mt-en-{language}"
            
            # Initialize to_english translator
            to_english_tokenizer = MarianTokenizer.from_pretrained(to_english_model_name)
            to_english_model = MarianMTModel.from_pretrained(to_english_model_name)
            
            # Initialize from_english translator
            from_english_tokenizer = MarianTokenizer.from_pretrained(from_english_model_name)
            from_english_model = MarianMTModel.from_pretrained(from_english_model_name)
            
            # Store translators
            self.translators[language] = {
                "to_english": {
                    "tokenizer": to_english_tokenizer,
                    "model": to_english_model
                },
                "from_english": {
                    "tokenizer": from_english_tokenizer,
                    "model": from_english_model
                }
            }
            
            logger.info(f"Initialized Marian translators for language: {language}")
            
        except Exception as e:
            logger.error(f"Error initializing Marian translators for {language}: {e}")
    
    def _initialize_nlp_components(self) -> None:
        """Initialize NLP components for language detection and processing."""
        try:
            # Initialize spaCy models for supported languages when available
            available_spacy_models = {
                "en": "en_core_web_sm",
                "fr": "fr_core_news_sm",
                "de": "de_core_news_sm",
                "es": "es_core_news_sm",
                "it": "it_core_news_sm",
                "pt": "pt_core_news_sm",
                "nl": "nl_core_news_sm"
            }
            
            for lang, model_name in available_spacy_models.items():
                if lang in self.supported_languages:
                    try:
                        self.nlp_models[lang] = spacy.load(model_name)
                        logger.info(f"Loaded spaCy model for {lang}: {model_name}")
                    except:
                        logger.warning(f"Could not load spaCy model for {lang}")
                        
            logger.info(f"Initialized NLP components for {len(self.nlp_models)} languages")
            
        except Exception as e:
            logger.error(f"Error initializing NLP components: {e}")
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of a text.
        
        Args:
            text: Text to detect language of
            
        Returns:
            ISO 639-1 language code (e.g., 'en', 'fr')
        """
        if not DEPS_AVAILABLE:
            return "en"  # Default to English if dependencies not available
            
        try:
            # Use langdetect for language detection
            language = detect(text[:1000])  # Use first 1000 chars for efficiency
            
            # Return language if supported, otherwise default to English
            if language in self.supported_languages:
                return language
            else:
                logger.warning(f"Detected unsupported language: {language}, defaulting to English")
                return "en"
                
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return "en"  # Default to English on error
    
    def translate_to_english(self, text: str, source_language: Optional[str] = None) -> str:
        """
        Translate text to English.
        
        Args:
            text: Text to translate
            source_language: Source language code (if None, will be auto-detected)
            
        Returns:
            Translated text in English
        """
        if not DEPS_AVAILABLE:
            return text
            
        # Detect language if not provided
        if not source_language:
            source_language = self.detect_language(text)
            
        # No translation needed for English
        if source_language == "en":
            return text
            
        # Check if translator available for this language
        if source_language not in self.translators:
            logger.warning(f"No translator available for {source_language}, returning original text")
            return text
            
        try:
            translator = self.translators[source_language]["to_english"]
            
            if self.translator_type == "marian":
                # Use Marian MT for translation
                tokenizer = translator["tokenizer"]
                model = translator["model"]
                
                # Translate in chunks to avoid OOM
                max_length = 512
                chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
                translated_chunks = []
                
                for chunk in chunks:
                    inputs = tokenizer([chunk], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
                    translated = model.generate(**inputs)
                    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
                    translated_chunks.append(translated_text)
                    
                return " ".join(translated_chunks)
            else:
                # Use other translator APIs
                # Translate in chunks to respect API limits
                chunk_size = 1000  # Characters
                chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
                translated_chunks = []
                
                for chunk in chunks:
                    translated_chunk = translator.translate(chunk)
                    translated_chunks.append(translated_chunk)
                    
                return " ".join(translated_chunks)
                
        except Exception as e:
            logger.error(f"Error translating to English: {e}")
            return text  # Return original text on error
    
    def translate_from_english(self, text: str, target_language: str) -> str:
        """
        Translate text from English to target language.
        
        Args:
            text: English text to translate
            target_language: Target language code
            
        Returns:
            Translated text in target language
        """
        if not DEPS_AVAILABLE:
            return text
            
        # No translation needed for English
        if target_language == "en":
            return text
            
        # Check if translator available for this language
        if target_language not in self.translators:
            logger.warning(f"No translator available for {target_language}, returning original text")
            return text
            
        try:
            translator = self.translators[target_language]["from_english"]
            
            if self.translator_type == "marian":
                # Use Marian MT for translation
                tokenizer = translator["tokenizer"]
                model = translator["model"]
                
                # Translate in chunks to avoid OOM
                max_length = 512
                chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
                translated_chunks = []
                
                for chunk in chunks:
                    inputs = tokenizer([chunk], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
                    translated = model.generate(**inputs)
                    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
                    translated_chunks.append(translated_text)
                    
                return " ".join(translated_chunks)
            else:
                # Use other translator APIs
                # Translate in chunks to respect API limits
                chunk_size = 1000  # Characters
                chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
                translated_chunks = []
                
                for chunk in chunks:
                    translated_chunk = translator.translate(chunk)
                    translated_chunks.append(translated_chunk)
                    
                return " ".join(translated_chunks)
                
        except Exception as e:
            logger.error(f"Error translating from English: {e}")
            return text  # Return original text on error
    
    def extract_knowledge(self, text: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract knowledge from text in any supported language.
        
        Args:
            text: Text to extract knowledge from
            language: Language code (if None, will be auto-detected)
            
        Returns:
            Dictionary of extracted knowledge
        """
        # Detect language if not provided
        if not language:
            language = self.detect_language(text)
            
        logger.info(f"Extracting knowledge from text in {language} language")
        
        # For non-English text, translate to English first
        if language != "en":
            english_text = self.translate_to_english(text, language)
        else:
            english_text = text
        
        # Extract knowledge using the enhanced extractor if available
        if self.enhanced_extractor:
            extracted_knowledge = self._extract_with_enhanced_extractor(english_text)
        else:
            extracted_knowledge = self._extract_with_llm(english_text)
        
        # Add language metadata
        extracted_knowledge["source_language"] = language
        
        # If original language wasn't English, translate key elements back
        if language != "en":
            extracted_knowledge = self._translate_knowledge_to_original_language(
                extracted_knowledge, language
            )
        
        # Save extraction results
        self._save_extraction(extracted_knowledge, language)
        
        return extracted_knowledge
    
    def _extract_with_enhanced_extractor(self, text: str) -> Dict[str, Any]:
        """Extract knowledge using the enhanced extractor."""
        if not self.enhanced_extractor:
            return self._extract_with_llm(text)
            
        try:
            # Create a temporary file for the extraction
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as temp:
                temp.write(text)
                temp_path = temp.name
                
            # Extract knowledge
            knowledge = self.enhanced_extractor.extract_knowledge_from_paper(temp_path)
            
            # Clean up
            try:
                os.unlink(temp_path)
            except:
                pass
                
            return knowledge
        except Exception as e:
            logger.error(f"Error using enhanced extractor: {e}")
            return self._extract_with_llm(text)
    
    def _extract_with_llm(self, text: str) -> Dict[str, Any]:
        """Extract knowledge using LLM-based approach."""
        # Extract with OpenAI if available
        if self.openai_client:
            return self._extract_with_openai(text)
        
        # Extract with Anthropic if available
        if self.anthropic_client:
            return self._extract_with_anthropic(text)
            
        # Fallback to basic extraction
        logger.warning("No LLM clients available for extraction, using basic approach")
        return self._extract_basic(text)
    
    def _extract_with_openai(self, text: str) -> Dict[str, Any]:
        """Extract knowledge using OpenAI."""
        prompt = """
        Extract structured knowledge from the following scientific text. 
        Include:
        
        1. Core concepts and their definitions
        2. Key research questions or hypotheses
        3. Methodologies and approaches used
        4. Main findings and results
        5. Important measurements or statistics
        
        Format the response as a JSON object with these keys:
        "core_concepts", "terminology", "research_questions", "methodologies", "key_findings", "measurements"
        
        For each item, include a relevance score from 0.0 to 1.0 indicating how central it is to the text.
        
        Text:
        {text}
        """
        
        try:
            # Process in chunks if text is too long
            max_chunk_size = 4000
            chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            chunk_extractions = []
            
            for i, chunk in enumerate(chunks[:5]):  # Limit to first 5 chunks
                logger.info(f"Processing chunk {i+1}/{min(len(chunks), 5)}")
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You extract structured knowledge from scientific texts."},
                        {"role": "user", "content": prompt.format(text=chunk)}
                    ],
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                extraction = json.loads(content)
                extraction["chunk_index"] = i
                chunk_extractions.append(extraction)
            
            # Combine extractions from all chunks
            combined = self._combine_chunk_extractions(chunk_extractions)
            return combined
            
        except Exception as e:
            logger.error(f"Error extracting with OpenAI: {e}")
            return self._extract_basic(text)
    
    def _extract_with_anthropic(self, text: str) -> Dict[str, Any]:
        """Extract knowledge using Anthropic Claude."""
        prompt = """
        Extract structured knowledge from the following scientific text. 
        Include:
        
        1. Core concepts and their definitions
        2. Key research questions or hypotheses
        3. Methodologies and approaches used
        4. Main findings and results
        5. Important measurements or statistics
        
        Format the response as a JSON object with these keys:
        "core_concepts", "terminology", "research_questions", "methodologies", "key_findings", "measurements"
        
        For each item, include a relevance score from 0.0 to 1.0 indicating how central it is to the text.
        
        Text:
        {text}
        """
        
        try:
            # Process in chunks if text is too long
            max_chunk_size = 10000
            chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            chunk_extractions = []
            
            for i, chunk in enumerate(chunks[:3]):  # Limit to first 3 chunks
                logger.info(f"Processing chunk {i+1}/{min(len(chunks), 3)}")
                
                response = self.anthropic_client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=2000,
                    messages=[
                        {"role": "user", "content": prompt.format(text=chunk)}
                    ],
                    system="You extract structured knowledge from scientific texts."
                )
                
                content = response.content[0].text
                
                # Extract JSON from Claude's response
                import re
                json_match = re.search(r'{[\s\S]*}', content)
                if json_match:
                    json_str = json_match.group(0)
                    extraction = json.loads(json_str)
                    extraction["chunk_index"] = i
                    chunk_extractions.append(extraction)
            
            # Combine extractions from all chunks
            combined = self._combine_chunk_extractions(chunk_extractions)
            return combined
            
        except Exception as e:
            logger.error(f"Error extracting with Anthropic: {e}")
            return self._extract_basic(text)
    
    def _extract_basic(self, text: str) -> Dict[str, Any]:
        """Basic knowledge extraction without LLMs."""
        # Use spaCy for basic extraction if available
        if "en" in self.nlp_models:
            nlp = self.nlp_models["en"]
            
            try:
                # Process in chunks to avoid memory issues
                max_chunk_size = 100000  # Characters
                chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
                
                all_entities = []
                for chunk in chunks:
                    doc = nlp(chunk)
                    all_entities.extend([(ent.text, ent.label_) for ent in doc.ents])
                
                # Group entities by type
                entity_groups = defaultdict(list)
                for entity, label in all_entities:
                    entity_groups[label].append(entity)
                
                # Create basic extraction structure
                extraction = {
                    "core_concepts": [{"text": e, "relevance": 0.7} for e in set(entity_groups.get("CONCEPT", []))],
                    "terminology": {},
                    "research_questions": [],
                    "methodologies": [{"text": e, "relevance": 0.7} for e in set(entity_groups.get("METHOD", []))],
                    "key_findings": [],
                    "measurements": [{"text": e, "relevance": 0.7} for e in set(entity_groups.get("QUANTITY", []))]
                }
                
                return extraction
                
            except Exception as e:
                logger.error(f"Error in basic extraction: {e}")
        
        # Fallback to minimal extraction
        return {
            "core_concepts": [],
            "terminology": {},
            "research_questions": [],
            "methodologies": [],
            "key_findings": [],
            "measurements": []
        }
    
    def _combine_chunk_extractions(self, extractions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine extractions from multiple chunks."""
        if not extractions:
            return {
                "core_concepts": [],
                "terminology": {},
                "research_questions": [],
                "methodologies": [],
                "key_findings": [],
                "measurements": []
            }
            
        combined = {
            "core_concepts": [],
            "terminology": {},
            "research_questions": [],
            "methodologies": [],
            "key_findings": [],
            "measurements": []
        }
        
        # Track seen items to avoid duplicates
        seen_items = {
            "core_concepts": set(),
            "research_questions": set(),
            "methodologies": set(),
            "key_findings": set(),
            "measurements": set()
        }
        
        # Combine list fields
        for extraction in extractions:
            for field in ["core_concepts", "research_questions", "methodologies", "key_findings", "measurements"]:
                if field in extraction:
                    for item in extraction[field]:
                        if isinstance(item, dict) and "text" in item:
                            item_text = item["text"].lower()
                            if item_text not in seen_items[field]:
                                seen_items[field].add(item_text)
                                combined[field].append(item)
                        elif isinstance(item, str):
                            item_text = item.lower()
                            if item_text not in seen_items[field]:
                                seen_items[field].add(item_text)
                                combined[field].append({"text": item, "relevance": 0.7})
            
            # Combine terminology
            if "terminology" in extraction:
                for term, definition in extraction["terminology"].items():
                    if term not in combined["terminology"]:
                        # Handle both string and dict definitions
                        if isinstance(definition, dict) and "definition" in definition:
                            combined["terminology"][term] = definition["definition"]
                        else:
                            combined["terminology"][term] = definition
        
        return combined
    
    def _translate_knowledge_to_original_language(
        self, 
        knowledge: Dict[str, Any], 
        target_language: str
    ) -> Dict[str, Any]:
        """Translate key knowledge elements back to original language."""
        if not DEPS_AVAILABLE or target_language == "en":
            return knowledge
            
        try:
            # Create bilingual knowledge structure
            bilingual = {
                "core_concepts": [],
                "terminology": {},
                "research_questions": [],
                "methodologies": [],
                "key_findings": [],
                "measurements": [],
                "source_language": target_language
            }
            
            # Translate list fields
            for field in ["core_concepts", "research_questions", "methodologies", "key_findings", "measurements"]:
                if field in knowledge:
                    for item in knowledge[field]:
                        if isinstance(item, dict) and "text" in item:
                            # Translate item text
                            translated_text = self.translate_from_english(item["text"], target_language)
                            
                            # Create bilingual item
                            bilingual_item = item.copy()
                            bilingual_item["text_original"] = translated_text
                            bilingual_item["text_english"] = item["text"]
                            
                            # Use translated text as primary
                            bilingual_item["text"] = translated_text
                            
                            bilingual[field].append(bilingual_item)
                        elif isinstance(item, str):
                            translated_text = self.translate_from_english(item, target_language)
                            bilingual[field].append({
                                "text": translated_text,
                                "text_original": translated_text,
                                "text_english": item,
                                "relevance": 0.7
                            })
            
            # Translate terminology
            if "terminology" in knowledge:
                for term, definition in knowledge["terminology"].items():
                    # Translate both term and definition
                    translated_term = self.translate_from_english(term, target_language)
                    
                    if isinstance(definition, dict) and "definition" in definition:
                        translated_def = self.translate_from_english(definition["definition"], target_language)
                        bilingual["terminology"][translated_term] = {
                            "definition": translated_def,
                            "definition_original": translated_def,
                            "definition_english": definition["definition"],
                            "term_english": term
                        }
                    else:
                        translated_def = self.translate_from_english(definition, target_language)
                        bilingual["terminology"][translated_term] = {
                            "definition": translated_def,
                            "definition_original": translated_def,
                            "definition_english": definition,
                            "term_english": term
                        }
            
            return bilingual
            
        except Exception as e:
            logger.error(f"Error translating knowledge back to {target_language}: {e}")
            return knowledge
    
    def _save_extraction(self, extraction: Dict[str, Any], language: str) -> None:
        """Save extraction results to file."""
        # Generate a timestamp-based filename
        import time
        timestamp = int(time.time())
        
        # Create language-specific directory
        language_dir = self.output_dir / language
        language_dir.mkdir(exist_ok=True)
        
        # Save to file
        output_file = language_dir / f"extraction_{timestamp}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(extraction, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved extraction results for {language} to {output_file}")
    
    def process_paper(self, paper_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a paper in any supported language.
        
        Args:
            paper_path: Path to paper file (PDF or text)
            
        Returns:
            Dictionary of extracted knowledge
        """
        # Read paper content
        text_content = self._read_paper_content(paper_path)
        if not text_content:
            logger.error(f"Failed to read content from {paper_path}")
            return {}
            
        # Detect language
        language = self.detect_language(text_content)
        logger.info(f"Detected language for {paper_path}: {language}")
        
        # Extract knowledge
        knowledge = self.extract_knowledge(text_content, language)
        
        # Add paper metadata
        result = {
            "paper_path": str(paper_path),
            "paper_name": Path(paper_path).name,
            "language": language,
            "knowledge": knowledge
        }
        
        # Save paper-specific results
        output_file = self.output_dir / f"paper_{Path(paper_path).stem}_{language}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Processed paper {paper_path} in {language}")
        
        return result
    
    def _read_paper_content(self, paper_path: Union[str, Path]) -> str:
        """Read content from a paper file."""
        path = Path(paper_path)
        
        try:
            if path.suffix.lower() == '.pdf':
                # Read PDF content
                try:
                    import PyPDF2
                    with open(path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text() + "\n"
                        return text
                except ImportError:
                    logger.error("PyPDF2 not installed for PDF reading")
                    return ""
            else:
                # Read text file with appropriate encoding detection
                try:
                    import chardet
                    with open(path, 'rb') as file:
                        raw_data = file.read()
                        encoding = chardet.detect(raw_data)['encoding']
                    
                    with open(path, 'r', encoding=encoding) as file:
                        return file.read()
                except ImportError:
                    # Fallback to utf-8
                    with open(path, 'r', encoding='utf-8') as file:
                        return file.read()
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            return ""
    
    def process_papers(self, papers_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Process all papers in a directory, supporting multiple languages.
        
        Args:
            papers_dir: Directory containing papers
            
        Returns:
            Dictionary of results grouped by language
        """
        papers_dir = Path(papers_dir)
        
        # Find all PDF and text files
        paper_files = list(papers_dir.glob("*.pdf")) + list(papers_dir.glob("*.txt"))
        
        if not paper_files:
            logger.warning(f"No papers found in {papers_dir}")
            return {}
            
        logger.info(f"Processing {len(paper_files)} papers with multilingual support")
        
        # Process each paper
        all_results = []
        
        for paper_path in paper_files:
            try:
                result = self.process_paper(paper_path)
                if result:
                    all_results.append(result)
            except Exception as e:
                logger.error(f"Error processing paper {paper_path}: {e}")
        
        # Group results by language
        results_by_language = defaultdict(list)
        for result in all_results:
            language = result.get("language", "unknown")
            results_by_language[language].append(result)
        
        # Create summary
        summary = {
            "papers_processed": len(all_results),
            "languages_found": dict(sorted(
                [(lang, len(papers)) for lang, papers in results_by_language.items()],
                key=lambda x: x[1],
                reverse=True
            )),
            "results_by_language": dict(results_by_language)
        }
        
        # Save summary
        summary_file = self.output_dir / "multilingual_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            # Save a simplified summary without the full results
            simplified_summary = {
                "papers_processed": summary["papers_processed"],
                "languages_found": summary["languages_found"],
                "paper_counts": {
                    lang: len(papers) for lang, papers in results_by_language.items()
                }
            }
            json.dump(simplified_summary, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Processed {len(all_results)} papers in {len(results_by_language)} languages")
        
        return summary 