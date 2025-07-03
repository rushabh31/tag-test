# -*- coding: utf-8 -*-
"""Ultra-Accurate-Custom-RAG-with-Full-Page-Images.ipynb

Ultra-accurate RAG system with:
1. Custom MMR reranking (no external models)
2. VertexAI embeddings only
3. HNSW with cosine similarity optimization
4. Full PDF page images when content is used
5. Maximum accuracy focus
"""

import os
import re
import tempfile
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Set, Union
from dataclasses import dataclass, field
import json
from tqdm import tqdm
import time
import logging
import base64
from io import BytesIO
import hashlib
from collections import defaultdict
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import pypdf
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import cv2

# Only HNSW and BM25 - no external models
import hnswlib
from rank_bm25 import BM25Okapi

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
import gradio as gr

# VertexAI - our only embedding source
import vertexai
from google.oauth2.credentials import Credentials
from helpers import get_coin_token
from vertexai.generative_models import GenerativeModel

class VertexGenAI:
    def __init__(self):
        credentials = Credentials(get_coin_token())
        vertexai.init(
            project="pri-gen-ai",
            api_transport="rest",
            api_endpoint="https://xyz/vertex",
            credentials=credentials,
        )
        self.metadata = [("x-user", os.getenv("USERNAME"))]

    def generate_content(self, prompt: str = "Provide interesting trivia"):
        """Generate content based on the provided prompt."""
        model = GenerativeModel("gemini-1.5-pro-002")
        resp = model.generate_content(prompt, metadata=self.metadata)
        return resp.text if resp else None

    def get_embeddings(self, texts: list[str], model_name: str = "text-embedding-004"):
        """Get embeddings for a list of texts using Vertex AI."""
        from vertexai.language_models import TextEmbeddingModel
        
        model = TextEmbeddingModel.from_pretrained(model_name, metadata=self.metadata)
        
        if not texts:
            return []
        
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = model.get_embeddings(texts, metadata=self.metadata)
        return embeddings

@dataclass
class PageImageData:
    """Full page image data."""
    page_number: int
    page_image: Image.Image
    base64_string: str
    extracted_regions: List[Dict] = field(default_factory=list)
    ocr_text: str = ""
    confidence: float = 0.0

@dataclass
class EnhancedImageData:
    """Enhanced image data with custom analysis."""
    image: Image.Image
    page_number: int
    bbox: Optional[Tuple[int, int, int, int]] = None
    ocr_text: str = ""
    base64_string: str = ""
    image_type: str = ""
    analysis: str = ""
    confidence: float = 0.0
    semantic_keywords: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    context_text: str = ""
    visual_features: Dict[str, float] = field(default_factory=dict)

@dataclass
class UltraChunk:
    """Ultra-enhanced chunk with maximum context."""
    text: str
    page_numbers: List[int]
    start_char_idx: int
    end_char_idx: int
    filename: str
    section_info: Dict[str, str] = field(default_factory=dict)
    image_refs: List[int] = field(default_factory=list)
    chunk_hash: str = ""
    context_before: str = ""
    context_after: str = ""
    semantic_density: float = 0.0
    embedding: Optional[List[float]] = None
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    page_coverage: float = 0.0  # What % of page this chunk covers
    
    def __post_init__(self):
        content = f"{self.text}{self.filename}{self.page_numbers}"
        self.chunk_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        
        # Calculate enhanced metrics
        sentences = len([s for s in self.text.split('.') if s.strip()])
        words = len(self.text.split())
        self.semantic_density = words / max(sentences, 1)

    def to_document(self) -> Document:
        # Include maximum context
        enhanced_content = f"{self.context_before}\n\n{self.text}\n\n{self.context_after}".strip()
        
        return Document(
            page_content=enhanced_content,
            metadata={
                "page_numbers": self.page_numbers,
                "source": self.filename,
                "start_idx": self.start_char_idx,
                "end_idx": self.end_char_idx,
                "section_info": self.section_info,
                "image_refs": self.image_refs,
                "chunk_hash": self.chunk_hash,
                "semantic_density": self.semantic_density,
                "original_text": self.text,
                "page_coverage": self.page_coverage,
                "relevance_scores": self.relevance_scores
            }
        )

@dataclass
class UltraDocument:
    """Ultra-enhanced PDF document with full page images."""
    filename: str
    content: str = ""
    pages: Dict[int, str] = field(default_factory=dict)
    char_to_page_map: List[int] = field(default_factory=list)
    chunks: List[UltraChunk] = field(default_factory=list)
    images: List[EnhancedImageData] = field(default_factory=list)
    page_images: Dict[int, PageImageData] = field(default_factory=dict)  # Full page images
    page_to_chunks: Dict[int, List[int]] = field(default_factory=dict)
    page_to_images: Dict[int, List[int]] = field(default_factory=dict)

    @property
    def langchain_documents(self) -> List[Document]:
        return [chunk.to_document() for chunk in self.chunks]

class UltraAccurateMMRReranker:
    """Ultra-accurate MMR reranker using only VertexAI embeddings."""
    
    def __init__(self, vertex_gen_ai: VertexGenAI):
        self.vertex_gen_ai = vertex_gen_ai
        # Optimized parameters for maximum accuracy
        self.lambda_diversity = 0.6  # Balance relevance vs diversity
        self.similarity_threshold = 0.85  # High similarity threshold for diversity
        self.relevance_weight = 0.7
        self.diversity_weight = 0.3
        
    def get_embedding_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings in batch for efficiency."""
        try:
            if not texts:
                return []
                
            # Batch process for efficiency
            batch_size = 10
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                embeddings_response = self.vertex_gen_ai.get_embeddings(batch)
                
                batch_embeddings = []
                for emb in embeddings_response:
                    if hasattr(emb, 'values'):
                        vector = np.array(emb.values, dtype=np.float32)
                        # L2 normalize for cosine similarity
                        vector = vector / (np.linalg.norm(vector) + 1e-12)
                        batch_embeddings.append(vector)
                    else:
                        # Zero vector fallback
                        batch_embeddings.append(np.zeros(768, dtype=np.float32))
                
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error getting embeddings batch: {e}")
            return [np.zeros(768, dtype=np.float32) for _ in texts]
    
    def calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Ultra-accurate cosine similarity calculation."""
        try:
            # Ensure vectors are normalized
            vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-12)
            vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-12)
            
            # Calculate cosine similarity
            similarity = np.dot(vec1_norm, vec2_norm)
            
            # Clamp to valid range
            similarity = np.clip(similarity, -1.0, 1.0)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def calculate_semantic_similarity(self, doc1: Document, doc2: Document) -> float:
        """Calculate semantic similarity between documents."""
        try:
            # Use original text for better comparison
            text1 = doc1.metadata.get('original_text', doc1.page_content)
            text2 = doc2.metadata.get('original_text', doc2.page_content)
            
            # Get embeddings
            embeddings = self.get_embedding_batch([text1, text2])
            
            if len(embeddings) == 2:
                return self.calculate_cosine_similarity(embeddings[0], embeddings[1])
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def calculate_relevance_score(self, query: str, document: Document) -> float:
        """Calculate ultra-accurate relevance score."""
        try:
            # Get original text for analysis
            doc_text = document.metadata.get('original_text', document.page_content)
            
            # Get embeddings
            embeddings = self.get_embedding_batch([query, doc_text])
            
            if len(embeddings) == 2:
                # Semantic similarity component
                semantic_sim = self.calculate_cosine_similarity(embeddings[0], embeddings[1])
                
                # Additional relevance factors
                relevance_factors = self._calculate_relevance_factors(query, document)
                
                # Combine scores
                final_relevance = (
                    0.7 * semantic_sim +
                    0.3 * relevance_factors
                )
                
                return max(0.0, min(1.0, final_relevance))
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating relevance score: {e}")
            return 0.0
    
    def _calculate_relevance_factors(self, query: str, document: Document) -> float:
        """Calculate additional relevance factors."""
        try:
            metadata = document.metadata
            score = 0.0
            
            # Section relevance
            section_info = metadata.get('section_info', {})
            if section_info:
                query_lower = query.lower()
                for section_key, section_title in section_info.items():
                    if any(word in section_title.lower() for word in query_lower.split()):
                        score += 0.3
                        break
            
            # Semantic density bonus
            semantic_density = metadata.get('semantic_density', 0)
            if semantic_density > 15:  # High information content
                score += 0.2
            
            # Page coverage bonus (larger chunks from important pages)
            page_coverage = metadata.get('page_coverage', 0)
            if page_coverage > 0.3:  # Significant portion of page
                score += 0.15
            
            # Image reference bonus
            image_refs = metadata.get('image_refs', [])
            if image_refs and any(keyword in query.lower() for keyword in ['chart', 'graph', 'table', 'figure', 'image']):
                score += 0.25
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating relevance factors: {e}")
            return 0.0
    
    def ultra_mmr_rerank(self, query: str, documents: List[Document], k: int = 5) -> List[Tuple[Document, float]]:
        """Ultra-accurate MMR reranking algorithm."""
        try:
            if not documents or k <= 0:
                return []
            
            logger.info(f"Ultra MMR reranking {len(documents)} documents to select top {k}")
            
            # Step 1: Calculate relevance scores for all documents
            relevance_scores = []
            for doc in documents:
                relevance = self.calculate_relevance_score(query, doc)
                relevance_scores.append(relevance)
            
            # Step 2: Initialize MMR selection
            selected_docs = []
            selected_indices = set()
            remaining_indices = list(range(len(documents)))
            
            # Step 3: Iterative MMR selection
            for selection_round in range(min(k, len(documents))):
                if not remaining_indices:
                    break
                
                best_score = -float('inf')
                best_idx = -1
                
                # Evaluate each remaining document
                for idx in remaining_indices:
                    current_doc = documents[idx]
                    current_relevance = relevance_scores[idx]
                    
                    # Calculate diversity score (similarity to already selected)
                    max_similarity = 0.0
                    if selected_indices:
                        similarities = []
                        for selected_idx in selected_indices:
                            selected_doc = documents[selected_idx]
                            similarity = self.calculate_semantic_similarity(current_doc, selected_doc)
                            similarities.append(similarity)
                        
                        max_similarity = max(similarities) if similarities else 0.0
                    
                    # MMR score calculation
                    mmr_score = (
                        self.relevance_weight * current_relevance - 
                        self.diversity_weight * max_similarity
                    )
                    
                    # Additional penalties for very similar content
                    if max_similarity > self.similarity_threshold:
                        mmr_score *= 0.5  # Significant penalty for redundancy
                    
                    # Bonus for first selection (highest relevance)
                    if selection_round == 0:
                        mmr_score = current_relevance  # Pure relevance for first pick
                    
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = idx
                
                # Select the best document
                if best_idx != -1:
                    selected_doc = documents[best_idx]
                    final_score = relevance_scores[best_idx]  # Use relevance as final score
                    
                    # Store MMR information in metadata
                    selected_doc.metadata['mmr_score'] = best_score
                    selected_doc.metadata['mmr_round'] = selection_round
                    selected_doc.metadata['relevance_score'] = final_score
                    
                    selected_docs.append((selected_doc, final_score))
                    selected_indices.add(best_idx)
                    remaining_indices.remove(best_idx)
            
            # Step 4: Final ranking by relevance (MMR ensures diversity)
            selected_docs.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Ultra MMR selected {len(selected_docs)} documents with diversity optimization")
            return selected_docs
            
        except Exception as e:
            logger.error(f"Error in ultra MMR reranking: {e}")
            # Fallback: return by relevance only
            try:
                scored_docs = []
                for doc in documents[:k]:
                    relevance = self.calculate_relevance_score(query, doc)
                    scored_docs.append((doc, relevance))
                return sorted(scored_docs, key=lambda x: x[1], reverse=True)
            except:
                return [(doc, 0.5) for doc in documents[:k]]

class CustomImageMatcher:
    """Custom image-text matching using only VertexAI."""
    
    def __init__(self, vertex_gen_ai: VertexGenAI):
        self.vertex_gen_ai = vertex_gen_ai
        
    def extract_visual_features(self, image: Image.Image) -> Dict[str, float]:
        """Extract comprehensive visual features."""
        try:
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            height, width = gray.shape
            features = {}
            
            # Basic metrics
            features['aspect_ratio'] = width / height
            features['area'] = width * height
            
            # Edge and complexity analysis
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = np.sum(edges > 0) / (width * height)
            
            # Contrast and brightness
            features['contrast'] = np.std(gray) / 255.0
            features['brightness'] = np.mean(gray) / 255.0
            
            # Line detection for structure analysis
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
            
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
            
            features['horizontal_structure'] = np.sum(horizontal_lines > 0) / (width * height)
            features['vertical_structure'] = np.sum(vertical_lines > 0) / (width * height)
            
            # Text area estimation
            binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            features['text_ratio'] = np.sum(binary == 0) / (width * height)
            
            # Histogram analysis for data visualization detection
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            features['histogram_peaks'] = len([i for i, v in enumerate(hist) if v > np.mean(hist) * 2])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting visual features: {e}")
            return {}
    
    def extract_semantic_keywords(self, text: str, context: str = "") -> List[str]:
        """Extract semantic keywords with context awareness."""
        keywords = []
        text_lower = text.lower()
        context_lower = context.lower()
        
        # Enhanced domain-specific indicators
        visual_indicators = {
            'chart': ['chart', 'graph', 'plot', 'data', 'trend', 'axis', 'value', 'percent', 'rate', 'statistics', 'visualization'],
            'table': ['table', 'row', 'column', 'cell', 'data', 'list', 'entry', 'comparison', 'matrix'],
            'diagram': ['diagram', 'flow', 'process', 'step', 'arrow', 'connection', 'structure', 'workflow', 'procedure'],
            'figure': ['figure', 'image', 'illustration', 'visual', 'picture', 'photo']
        }
        
        # Check both text and context
        combined_text = f"{text_lower} {context_lower}"
        
        for category, words in visual_indicators.items():
            found_words = [word for word in words if word in combined_text]
            keywords.extend(found_words)
        
        # Extract numerical values and percentages
        numbers = re.findall(r'\d+\.?\d*%?', text)
        keywords.extend(numbers[:7])
        
        # Extract important terms (capitalized, technical terms)
        important_terms = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
        keywords.extend(important_terms[:8])
        
        # Extract quoted or emphasized terms
        quoted_terms = re.findall(r'"([^"]+)"', text)
        keywords.extend(quoted_terms[:3])
        
        return list(set(keywords))
    
    def calculate_advanced_similarity(self, query: str, image: EnhancedImageData) -> float:
        """Calculate advanced similarity using VertexAI embeddings."""
        try:
            # Prepare comprehensive image representation
            image_text_parts = []
            
            if image.ocr_text:
                image_text_parts.append(image.ocr_text)
            
            if image.analysis:
                image_text_parts.append(image.analysis)
            
            if image.semantic_keywords:
                image_text_parts.append(" ".join(image.semantic_keywords))
            
            if image.context_text:
                image_text_parts.append(image.context_text[:200])
            
            image_representation = " ".join(image_text_parts)
            
            if not image_representation.strip():
                return 0.0
            
            # Get VertexAI embeddings
            try:
                embeddings_response = self.vertex_gen_ai.get_embeddings([query, image_representation])
                
                if len(embeddings_response) == 2:
                    query_emb = np.array(embeddings_response[0].values)
                    image_emb = np.array(embeddings_response[1].values)
                    
                    # Normalize
                    query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-12)
                    image_emb = image_emb / (np.linalg.norm(image_emb) + 1e-12)
                    
                    # Cosine similarity
                    semantic_similarity = np.dot(query_emb, image_emb)
                else:
                    semantic_similarity = 0.0
                    
            except Exception as e:
                logger.warning(f"Error getting embeddings for image similarity: {e}")
                semantic_similarity = 0.0
            
            # Keyword overlap similarity
            query_words = set(query.lower().split())
            image_words = set(image.semantic_keywords) | set(image.ocr_text.lower().split())
            
            if image_words:
                keyword_similarity = len(query_words & image_words) / len(query_words | image_words)
            else:
                keyword_similarity = 0.0
            
            # Type-specific relevance
            type_relevance = self._calculate_type_specific_relevance(query, image)
            
            # Visual feature relevance
            visual_relevance = self._calculate_visual_feature_relevance(query, image.visual_features)
            
            # Context relevance
            context_relevance = self._calculate_context_relevance(query, image.context_text)
            
            # Weighted combination optimized for accuracy
            final_similarity = (
                0.4 * semantic_similarity +
                0.25 * keyword_similarity +
                0.2 * type_relevance +
                0.1 * visual_relevance +
                0.05 * context_relevance
            )
            
            # Apply confidence multiplier
            final_similarity *= image.confidence
            
            return min(final_similarity, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating advanced similarity: {e}")
            return 0.0
    
    def _calculate_type_specific_relevance(self, query: str, image: EnhancedImageData) -> float:
        """Calculate type-specific relevance score."""
        query_lower = query.lower()
        image_type = image.image_type.lower()
        
        type_query_mapping = {
            'chart': ['chart', 'graph', 'data', 'trend', 'statistics', 'percentage', 'analysis', 'plot'],
            'table': ['table', 'data', 'list', 'comparison', 'values', 'entries', 'rows', 'columns'],
            'diagram': ['diagram', 'process', 'flow', 'structure', 'workflow', 'steps', 'procedure'],
            'figure': ['figure', 'image', 'illustration', 'visual', 'show', 'display']
        }
        
        relevant_terms = type_query_mapping.get(image_type, [])
        matches = sum(1 for term in relevant_terms if term in query_lower)
        
        max_possible = len(relevant_terms)
        return matches / max_possible if max_possible > 0 else 0.0
    
    def _calculate_visual_feature_relevance(self, query: str, visual_features: Dict[str, float]) -> float:
        """Calculate relevance based on visual features."""
        if not visual_features:
            return 0.0
        
        query_lower = query.lower()
        relevance = 0.0
        
        # Chart/graph queries
        if any(term in query_lower for term in ['chart', 'graph', 'plot', 'data']):
            relevance += visual_features.get('edge_density', 0) * 0.3
            relevance += visual_features.get('histogram_peaks', 0) * 0.1
        
        # Table queries
        if any(term in query_lower for term in ['table', 'data', 'rows', 'columns']):
            relevance += visual_features.get('horizontal_structure', 0) * 0.4
            relevance += visual_features.get('vertical_structure', 0) * 0.4
        
        # Text-heavy queries
        if any(term in query_lower for term in ['text', 'content', 'description']):
            relevance += visual_features.get('text_ratio', 0) * 0.5
        
        return min(relevance, 1.0)
    
    def _calculate_context_relevance(self, query: str, context: str) -> float:
        """Calculate context relevance."""
        if not context:
            return 0.0
        
        query_words = set(query.lower().split())
        context_words = set(context.lower().split())
        
        if not context_words:
            return 0.0
        
        overlap = len(query_words & context_words)
        return overlap / len(query_words) if query_words else 0.0

class UltraDocumentProcessor:
    """Ultra-enhanced document processor with full page images."""

    def __init__(self, vertex_gen_ai: VertexGenAI):
        self.vertex_gen_ai = vertex_gen_ai
        self.chunk_size = 500  # Smaller for precision
        self.chunk_overlap = 100  # Balanced overlap
        self.context_window = 300  # Extended context
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )
        
        self.image_matcher = CustomImageMatcher(vertex_gen_ai)
        self.mmr_reranker = UltraAccurateMMRReranker(vertex_gen_ai)
        
        # Enhanced patterns for section detection
        self.section_patterns = [
            r'(?:Section|SECTION|Chapter|CHAPTER)\s+(\d+(?:\.\d+)*)[:\s]+(.*?)(?=\n|$)',
            r'^(\d+(?:\.\d+)*)\s+(.*?)$',
            r'^([A-Z][A-Z\s]{2,20})$',
            r'^(Abstract|Introduction|Methodology|Methods|Results|Discussion|Conclusion|References)$',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):?\s*$'
        ]

    def extract_full_page_images(self, file_path: str) -> Dict[int, PageImageData]:
        """Extract full page images for reference."""
        page_images = {}
        
        try:
            logger.info("Extracting full page images...")
            pdf_images = convert_from_path(file_path, dpi=200)
            
            for page_num, page_image in enumerate(pdf_images, 1):
                try:
                    # Convert to base64
                    buffered = BytesIO()
                    page_image.save(buffered, format="PNG", quality=85)
                    page_base64 = base64.b64encode(buffered.getvalue()).decode()
                    
                    # Extract text for this page
                    page_ocr = ""
                    try:
                        page_ocr = pytesseract.image_to_string(page_image, config='--psm 3')
                    except Exception as e:
                        logger.warning(f"OCR failed for page {page_num}: {e}")
                    
                    page_data = PageImageData(
                        page_number=page_num,
                        page_image=page_image,
                        base64_string=page_base64,
                        ocr_text=page_ocr,
                        confidence=0.9  # High confidence for full pages
                    )
                    
                    page_images[page_num] = page_data
                    
                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {e}")
                    continue
            
            logger.info(f"Extracted {len(page_images)} full page images")
            return page_images
            
        except Exception as e:
            logger.error(f"Error extracting full page images: {e}")
            return {}

    def extract_region_images(self, file_path: str) -> List[EnhancedImageData]:
        """Extract specific image regions with enhanced analysis."""
        images = []
        
        try:
            pdf_text = self._extract_raw_text(file_path)
            pdf_images = convert_from_path(file_path, dpi=200)
            
            for page_num, page_image in enumerate(pdf_images, 1):
                page_context = self._get_page_context(pdf_text, page_num)
                page_images = self._extract_page_regions(page_image, page_num, page_context)
                images.extend(page_images)
                    
        except Exception as e:
            logger.error(f"Error extracting region images: {e}")
            
        return images

    def _extract_raw_text(self, file_path: str) -> Dict[int, str]:
        """Extract raw text by page."""
        pages_text = {}
        try:
            with open(file_path, "rb") as file:
                pdf = pypdf.PdfReader(file)
                for i, page in enumerate(pdf.pages):
                    pages_text[i + 1] = page.extract_text() or ""
        except Exception as e:
            logger.error(f"Error extracting raw text: {e}")
        return pages_text

    def _get_page_context(self, pdf_text: Dict[int, str], page_num: int) -> str:
        """Get comprehensive text context around a page."""
        context_pages = []
        
        # Include current page and adjacent pages
        for p in range(max(1, page_num - 1), min(len(pdf_text) + 1, page_num + 2)):
            if p in pdf_text:
                context_pages.append(pdf_text[p][:600])  # Extended context
        
        return " ".join(context_pages)

    def _extract_page_regions(self, page_image: Image.Image, page_num: int, 
                            page_context: str) -> List[EnhancedImageData]:
        """Extract image regions with ultra-accurate analysis."""
        images = []
        
        try:
            opencv_image = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            
            # Enhanced preprocessing
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Multiple edge detection approaches
            edges1 = cv2.Canny(blurred, 30, 80)
            edges2 = cv2.Canny(blurred, 50, 150)
            edges = cv2.bitwise_or(edges1, edges2)
            
            # Enhanced morphological operations
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Enhanced filtering
            min_area = 4000
            max_area = (page_image.width * page_image.height) * 0.8  # Max 80% of page
            
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area <= area <= max_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Filter by aspect ratio and position
                    if 0.1 <= aspect_ratio <= 10 and y > 50:  # Not in header
                        valid_contours.append((area, contour))
            
            # Sort by area and process top regions
            valid_contours.sort(reverse=True)
            
            processed_count = 0
            for area, contour in valid_contours[:4]:  # Top 4 regions
                try:
                    image_data = self._process_image_region_ultra(
                        page_image, contour, page_num, page_context, processed_count
                    )
                    
                    if image_data and image_data.confidence > 0.5:
                        # Get VertexAI embedding for image
                        image_text = f"{image_data.ocr_text} {image_data.analysis}"
                        if image_text.strip():
                            try:
                                emb_response = self.vertex_gen_ai.get_embeddings([image_text])
                                if emb_response and len(emb_response) > 0:
                                    image_data.embedding = emb_response[0].values
                            except Exception as e:
                                logger.warning(f"Could not get embedding for image: {e}")
                        
                        images.append(image_data)
                        processed_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error processing region: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error extracting page regions: {e}")
            
        return images

    def _process_image_region_ultra(self, page_image: Image.Image, contour, 
                                  page_num: int, page_context: str, 
                                  region_idx: int) -> Optional[EnhancedImageData]:
        """Ultra-accurate image region processing."""
        try:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Enhanced filtering
            aspect_ratio = w / h
            if aspect_ratio > 15 or aspect_ratio < 0.05:  # Very thin regions
                return None
                
            # Extract region
            roi = page_image.crop((x, y, x + w, y + h))
            
            # Ultra-accurate OCR with multiple approaches
            ocr_text = self._ultra_accurate_ocr(roi)
            
            # Extract visual features
            visual_features = self.image_matcher.extract_visual_features(roi)
            
            # Ultra-accurate analysis
            image_type, analysis, confidence = self._analyze_image_ultra(
                roi, ocr_text, page_context, visual_features
            )
            
            # Quality threshold
            if confidence < 0.5:
                return None
            
            # Extract enhanced keywords
            keywords = self.image_matcher.extract_semantic_keywords(
                f"{ocr_text} {analysis}", page_context
            )
            
            # Convert to base64
            buffered = BytesIO()
            roi.save(buffered, format="PNG", quality=90)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            return EnhancedImageData(
                image=roi,
                page_number=page_num,
                bbox=(x, y, x + w, y + h),
                ocr_text=ocr_text.strip(),
                base64_string=img_base64,
                image_type=image_type,
                analysis=analysis,
                confidence=confidence,
                semantic_keywords=keywords,
                context_text=page_context[:300],
                visual_features=visual_features
            )
            
        except Exception as e:
            logger.error(f"Error in ultra processing: {e}")
            return None

    def _ultra_accurate_ocr(self, image: Image.Image) -> str:
        """Ultra-accurate OCR with multiple approaches."""
        ocr_results = []
        
        try:
            # Multiple PSM modes for different content types
            psm_configs = [
                '--psm 6',  # Single uniform block
                '--psm 8',  # Single word
                '--psm 3',  # Auto page segmentation
                '--psm 11', # Sparse text
                '--psm 13'  # Raw line (for simple text)
            ]
            
            for config in psm_configs:
                try:
                    result = pytesseract.image_to_string(image, config=config)
                    if result.strip():
                        ocr_results.append(result.strip())
                except:
                    continue
            
            # Choose the longest meaningful result
            if ocr_results:
                best_result = max(ocr_results, key=lambda x: len(x.split()))
                return best_result
            else:
                return ""
                
        except Exception as e:
            logger.warning(f"Ultra OCR failed: {e}")
            return ""

    def _analyze_image_ultra(self, image: Image.Image, ocr_text: str, 
                           context: str, visual_features: Dict[str, float]) -> Tuple[str, str, float]:
        """Ultra-accurate image analysis."""
        try:
            text_lower = ocr_text.lower()
            context_lower = context.lower()
            
            # Initialize
            confidence = 0.4
            image_type = "figure"
            analysis = ""
            
            # Get visual metrics
            h_structure = visual_features.get('horizontal_structure', 0)
            v_structure = visual_features.get('vertical_structure', 0)
            edge_density = visual_features.get('edge_density', 0)
            text_ratio = visual_features.get('text_ratio', 0)
            histogram_peaks = visual_features.get('histogram_peaks', 0)
            
            # Ultra-accurate type detection
            
            # Table detection (enhanced)
            table_indicators = ['table', 'row', 'column', 'cell', '|', 'data']
            table_score = sum(1 for indicator in table_indicators if indicator in text_lower)
            
            if (h_structure > 0.015 and v_structure > 0.015) or table_score >= 2:
                image_type = "table"
                confidence = 0.85 + min(h_structure + v_structure, 0.15)
                
                # Enhanced table analysis
                rows = len([line for line in ocr_text.split('\n') if line.strip()])
                numbers = re.findall(r'\d+\.?\d*', ocr_text)
                
                analysis = f"Data table with {rows} rows containing {len(numbers)} numerical values. "
                
                if numbers:
                    sample_numbers = numbers[:5]
                    analysis += f"Sample values: {', '.join(sample_numbers)}. "
                
                analysis += f"Structure density: H={h_structure:.3f}, V={v_structure:.3f}."
            
            # Chart/Graph detection (enhanced)
            elif (edge_density > 0.12 or histogram_peaks > 15) or \
                 any(indicator in text_lower for indicator in ['chart', 'graph', '%', 'axis', 'plot']):
                image_type = "chart"
                confidence = 0.9
                
                percentages = re.findall(r'\d+\.?\d*%', text_lower)
                numbers = re.findall(r'\d+\.?\d*', text_lower)
                
                analysis = f"Data visualization chart with {len(numbers)} numeric elements"
                
                if percentages:
                    analysis += f" including {len(percentages)} percentages: {', '.join(percentages[:4])}"
                
                if histogram_peaks > 10:
                    analysis += f". Complex visualization with {histogram_peaks} data peaks"
                
                analysis += f". Visual complexity: edge density {edge_density:.3f}."
            
            # Diagram detection (enhanced)
            elif any(indicator in text_lower for indicator in ['flow', 'process', 'step', 'diagram', 'arrow']) or \
                 any(indicator in context_lower for indicator in ['process', 'workflow', 'procedure', 'method']):
                image_type = "diagram"
                confidence = 0.75 + min(edge_density, 0.2)
                
                analysis = f"Process diagram or workflow illustration. "
                
                if ocr_text:
                    steps = len(re.findall(r'\bstep\b|\b\d+\b|→|⇒', text_lower))
                    analysis += f"Contains {steps} process elements. "
                
                analysis += f"Structural complexity: {edge_density:.3f} edge density."
            
            # Text-heavy figure
            elif text_ratio > 0.4:
                image_type = "figure"
                confidence = 0.65
                
                word_count = len(ocr_text.split())
                analysis = f"Text-rich figure with {word_count} words ({text_ratio:.1%} text coverage). "
                
                if ocr_text:
                    analysis += f"Content preview: '{ocr_text[:80]}...'" if len(ocr_text) > 80 else f"Content: '{ocr_text}'"
            
            # General figure
            else:
                confidence = 0.55
                analysis = f"Visual figure or illustration"
                
                if ocr_text:
                    analysis += f" with text: '{ocr_text[:100]}...'" if len(ocr_text) > 100 else f": '{ocr_text}'"
                else:
                    analysis += " (primarily visual content)"
            
            # Context relevance boost
            context_boost = 0.0
            if any(word in context_lower for word in [image_type, 'figure', 'table', 'chart']):
                context_boost = 0.1
            
            # Quality assessment
            brightness = visual_features.get('brightness', 0.5)
            contrast = visual_features.get('contrast', 0.5)
            
            quality_score = 0.0
            if 0.2 < brightness < 0.8 and contrast > 0.25:
                quality_score = 0.05
                quality_label = "High"
            else:
                quality_label = "Medium"
            
            final_confidence = min(confidence + context_boost + quality_score, 1.0)
            
            analysis += f" [Quality: {quality_label}, Confidence: {final_confidence:.2f}]"
            
            return image_type, analysis, final_confidence
            
        except Exception as e:
            logger.error(f"Ultra image analysis failed: {e}")
            return "figure", f"Analysis error: {str(e)}", 0.3

    def extract_text_from_pdf(self, file_path: str) -> UltraDocument:
        """Extract text with ultra-enhanced processing."""
        doc = UltraDocument(filename=os.path.basename(file_path))
        
        try:
            # Extract full page images first
            doc.page_images = self.extract_full_page_images(file_path)
            
            # Extract region images
            doc.images = self.extract_region_images(file_path)
            
            # Build mappings
            for idx, img in enumerate(doc.images):
                page = img.page_number
                if page not in doc.page_to_images:
                    doc.page_to_images[page] = []
                doc.page_to_images[page].append(idx)
            
            # Extract text with enhanced processing
            full_text = ""
            char_to_page = []
            
            with open(file_path, "rb") as file:
                pdf = pypdf.PdfReader(file)
                
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    page_text = page.extract_text() or ""
                    page_text = self._ultra_clean_text(page_text)
                    
                    # Add ultra-high-quality OCR from images
                    if page_num in doc.page_to_images:
                        image_texts = []
                        for img_idx in doc.page_to_images[page_num]:
                            img = doc.images[img_idx]
                            if img.ocr_text and img.confidence > 0.7:  # High threshold
                                image_texts.append(f"[{img.image_type.title()}]: {img.ocr_text}")
                        
                        if image_texts:
                            page_text += f"\n\n=== Enhanced Visual Content on Page {page_num} ===\n" + "\n".join(image_texts)
                    
                    if page_text.strip():
                        doc.pages[page_num] = page_text
                        full_text += page_text + "\n\n"
                        char_to_page.extend([page_num] * len(page_text + "\n\n"))
            
            doc.content = full_text
            doc.char_to_page_map = char_to_page
            
            return doc
            
        except Exception as e:
            logger.error(f"Error in ultra text extraction: {e}")
            raise

    def _ultra_clean_text(self, text: str) -> str:
        """Ultra-enhanced text cleaning."""
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s*\n\s*', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Enhanced hyphenation fixes
        text = re.sub(r'([a-z])-\s*\n\s*([a-z])', r'\1\2', text)
        text = re.sub(r'([a-z])- ?([a-z])', r'\1\2', text)
        
        # OCR error corrections
        text = re.sub(r'\b(\w)\s+(\w)\b', r'\1\2', text)
        text = re.sub(r'(\w)\s*\.\s*(\w)', r'\1.\2', text)
        
        # Number formatting fixes
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
        text = re.sub(r'(\d)\s*,\s*(\d)', r'\1,\2', text)
        
        return text.strip()

    def chunk_document_ultra(self, doc: UltraDocument) -> UltraDocument:
        """Ultra-enhanced chunking with maximum context preservation."""
        if not doc.content:
            return doc
        
        try:
            raw_chunks = self.text_splitter.create_documents([doc.content])
            
            for i, chunk in enumerate(raw_chunks):
                chunk_text = chunk.page_content
                
                # Enhanced position finding
                start_pos = doc.content.find(chunk_text)
                if start_pos == -1:
                    # Fuzzy matching for position
                    words = chunk_text.split()[:7]
                    search_text = " ".join(words)
                    start_pos = doc.content.find(search_text)
                    if start_pos == -1:
                        start_pos = i * (self.chunk_size - self.chunk_overlap)
                
                end_pos = start_pos + len(chunk_text) - 1
                
                # Enhanced context extraction
                context_before = ""
                context_after = ""
                
                if start_pos > self.context_window:
                    context_start = start_pos - self.context_window
                    context_before = doc.content[context_start:start_pos].strip()
                
                if end_pos + self.context_window < len(doc.content):
                    context_end = end_pos + self.context_window
                    context_after = doc.content[end_pos:context_end].strip()
                
                # Find pages with enhanced mapping
                chunk_pages = set()
                for pos in range(max(0, start_pos), min(end_pos + 1, len(doc.char_to_page_map))):
                    if pos < len(doc.char_to_page_map):
                        chunk_pages.add(doc.char_to_page_map[pos])
                
                if not chunk_pages:
                    chunk_pages = {1}
                
                # Calculate page coverage
                page_coverage = len(chunk_text) / max([len(doc.pages.get(p, "")) for p in chunk_pages], 1)
                
                # Ultra-accurate image matching
                relevant_images = self._find_relevant_images_ultra(
                    chunk_text, list(chunk_pages), doc.images, doc.page_to_images
                )
                
                # Enhanced section extraction
                section_info = self._extract_section_info_ultra(chunk_text)
                
                # Create ultra chunk
                ultra_chunk = UltraChunk(
                    text=chunk_text,
                    page_numbers=sorted(list(chunk_pages)),
                    start_char_idx=start_pos,
                    end_char_idx=end_pos,
                    filename=doc.filename,
                    section_info=section_info,
                    image_refs=relevant_images,
                    context_before=context_before,
                    context_after=context_after,
                    page_coverage=page_coverage
                )
                
                # Get VertexAI embedding for chunk
                try:
                    emb_response = self.vertex_gen_ai.get_embeddings([chunk_text])
                    if emb_response and len(emb_response) > 0:
                        ultra_chunk.embedding = emb_response[0].values
                except Exception as e:
                    logger.warning(f"Could not get embedding for chunk: {e}")
                
                doc.chunks.append(ultra_chunk)
                
                # Build page to chunks mapping
                for page in chunk_pages:
                    if page not in doc.page_to_chunks:
                        doc.page_to_chunks[page] = []
                    doc.page_to_chunks[page].append(len(doc.chunks) - 1)
            
            logger.info(f"Created {len(doc.chunks)} ultra-enhanced chunks")
            return doc
            
        except Exception as e:
            logger.error(f"Error in ultra chunking: {e}")
            return doc

    def _find_relevant_images_ultra(self, chunk_text: str, chunk_pages: List[int], 
                                   images: List[EnhancedImageData], 
                                   page_to_images: Dict[int, List[int]]) -> List[int]:
        """Ultra-accurate image relevance matching."""
        relevant_images = []
        
        try:
            # Get candidate images from extended page range
            candidate_images = []
            for page in chunk_pages:
                # Current page
                if page in page_to_images:
                    candidate_images.extend(page_to_images[page])
                # Adjacent pages for context
                for adj_page in [page - 1, page + 1]:
                    if adj_page in page_to_images:
                        candidate_images.extend(page_to_images[adj_page])
            
            candidate_images = list(set(candidate_images))
            
            # Ultra-accurate scoring
            image_scores = []
            for img_idx in candidate_images:
                if img_idx < len(images):
                    img = images[img_idx]
                    
                    # Use advanced similarity calculation
                    relevance_score = self.image_matcher.calculate_advanced_similarity(chunk_text, img)
                    
                    # Enhanced factors
                    page_proximity = 1.0 if img.page_number in chunk_pages else 0.7
                    confidence_factor = img.confidence
                    
                    # Type-specific bonuses
                    type_bonus = 1.0
                    if any(keyword in chunk_text.lower() for keyword in ['chart', 'graph', 'table', 'figure']):
                        if img.image_type in ['chart', 'table', 'diagram']:
                            type_bonus = 1.2
                    
                    final_score = relevance_score * page_proximity * confidence_factor * type_bonus
                    
                    # Higher threshold for ultra accuracy
                    if final_score > 0.4:
                        image_scores.append((img_idx, final_score))
            
            # Sort and select top relevant images
            image_scores.sort(key=lambda x: x[1], reverse=True)
            relevant_images = [img_idx for img_idx, score in image_scores[:3]]  # Top 3
            
            logger.debug(f"Found {len(relevant_images)} ultra-relevant images for chunk")
            
        except Exception as e:
            logger.error(f"Error in ultra image matching: {e}")
        
        return relevant_images

    def _extract_section_info_ultra(self, text: str) -> Dict[str, str]:
        """Ultra-enhanced section extraction."""
        section_info = {}
        
        try:
            lines = text.split('\n')
            
            for pattern in self.section_patterns:
                for line_idx, line in enumerate(lines):
                    line = line.strip()
                    if not line or len(line) < 3:
                        continue
                    
                    match = re.match(pattern, line, re.IGNORECASE)
                    if match:
                        groups = match.groups()
                        
                        if len(groups) >= 2:
                            section_num = groups[0].strip()
                            section_title = groups[1].strip()
                            if section_title:
                                section_info[section_num] = section_title
                        elif len(groups) == 1:
                            section_title = groups[0].strip()
                            if len(section_title) > 2 and section_title.isupper():
                                section_info[f"section_{len(section_info)}"] = section_title
            
            # Additional pattern: Look for emphasized text
            emphasized_patterns = [
                r'\*\*([^*]+)\*\*',  # Bold markdown
                r'__([^_]+)__',      # Bold underscore
                r'([A-Z][A-Z\s]{4,})', # All caps
            ]
            
            for pattern in emphasized_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0] if match else ""
                    
                    if len(match) > 3 and len(match) < 50:
                        section_info[f"emphasis_{len(section_info)}"] = match.strip()
        
        except Exception as e:
            logger.error(f"Error in ultra section extraction: {e}")
        
        return section_info

    def process_pdf(self, file_path: str) -> UltraDocument:
        """Process PDF with ultra-enhanced accuracy."""
        doc = self.extract_text_from_pdf(file_path)
        return self.chunk_document_ultra(doc)

class VertexAIEmbeddings(Embeddings):
    """VertexAI embeddings - our only embedding source."""
    
    def __init__(self, vertex_gen_ai: VertexGenAI, model_name: str = "text-embedding-004"):
        self.vertex_gen_ai = vertex_gen_ai
        self.model_name = model_name
        self._embedding_dimension = None
        super().__init__()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with ultra-accurate preprocessing."""
        if not texts:
            return []
        
        try:
            processed_texts = [self._ultra_preprocess_text(text) for text in texts]
            
            logger.info(f"Embedding {len(processed_texts)} documents with VertexAI")
            embeddings_response = self.vertex_gen_ai.get_embeddings(processed_texts, self.model_name)
            
            embeddings = []
            for embedding in embeddings_response:
                if hasattr(embedding, 'values'):
                    vector = np.array(embedding.values, dtype=np.float32)
                    # Ultra-accurate normalization
                    vector = vector / (np.linalg.norm(vector) + 1e-12)
                    embeddings.append(vector.tolist())
                else:
                    embeddings.append(self._get_zero_vector())
            
            if embeddings and self._embedding_dimension is None:
                self._embedding_dimension = len(embeddings[0])
                logger.info(f"Set embedding dimension to {self._embedding_dimension}")
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting document embeddings: {str(e)}")
            return [self._get_zero_vector() for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query with ultra-accurate preprocessing."""
        if not text:
            return self._get_zero_vector()
        
        try:
            processed_text = self._ultra_preprocess_text(text)
            embeddings_response = self.vertex_gen_ai.get_embeddings([processed_text], self.model_name)
            
            if embeddings_response and len(embeddings_response) > 0:
                embedding = embeddings_response[0]
                if hasattr(embedding, 'values'):
                    vector = np.array(embedding.values, dtype=np.float32)
                    vector = vector / (np.linalg.norm(vector) + 1e-12)
                    
                    if self._embedding_dimension is None:
                        self._embedding_dimension = len(vector)
                    return vector.tolist()
            
            return self._get_zero_vector()
            
        except Exception as e:
            logger.error(f"Error getting query embedding: {str(e)}")
            return self._get_zero_vector()
    
    def _ultra_preprocess_text(self, text: str) -> str:
        """Ultra-accurate text preprocessing for embeddings."""
        # Enhanced cleaning
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove excessive punctuation
        text = re.sub(r'[^\w\s.,;:!?()-]', ' ', text)
        
        # Fix common OCR errors
        text = re.sub(r'\b(\w)\s+(\w)\s+(\w)\b', r'\1\2\3', text)
        
        # Truncate with sentence boundary awareness
        if len(text) > 7500:
            # Find last complete sentence
            truncated = text[:7500]
            last_period = truncated.rfind('.')
            if last_period > 6000:  # Keep if reasonable length
                text = truncated[:last_period + 1]
            else:
                text = truncated + "..."
        
        return text
    
    def _get_zero_vector(self) -> List[float]:
        """Get normalized zero vector."""
        dim = self._embedding_dimension if self._embedding_dimension else 768
        return [0.0] * dim

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)

class UltraHybridRetriever:
    """Ultra-accurate hybrid retriever with custom MMR."""
    
    def __init__(self, documents: List[Document] = None, vertex_gen_ai: VertexGenAI = None):
        logger.info("Initializing UltraHybridRetriever...")
        
        self.vertex_gen_ai = vertex_gen_ai
        self.embeddings = VertexAIEmbeddings(vertex_gen_ai) if vertex_gen_ai else None
        
        # Storage
        self.documents = []
        self.document_embeddings = []
        
        # Dense retrieval (HNSW) - optimized for accuracy
        self.hnsw_index = None
        self.dimension = None
        
        # Sparse retrieval (BM25) - enhanced
        self.bm25 = None
        self.tokenized_docs = []
        
        # Ultra-accurate MMR reranker
        self.mmr_reranker = UltraAccurateMMRReranker(vertex_gen_ai) if vertex_gen_ai else None
        
        if documents and len(documents) > 0:
            self.add_documents(documents)
    
    def add_documents(self, documents: List[Document]):
        """Add documents to ultra-accurate hybrid index."""
        try:
            logger.info(f"Adding {len(documents)} documents to ultra hybrid index")
            
            texts = [doc.page_content for doc in documents]
            
            # Get ultra-accurate embeddings
            embeddings = self.embeddings.embed_documents(texts)
            
            if embeddings:
                # Initialize HNSW with ultra-accurate parameters
                if self.hnsw_index is None:
                    self.dimension = len(embeddings[0])
                    self.hnsw_index = hnswlib.Index(space='cosine', dim=self.dimension)
                    # Ultra-accurate HNSW parameters
                    self.hnsw_index.init_index(
                        max_elements=100000, 
                        ef_construction=500,  # Higher for accuracy
                        M=48  # Higher connectivity for accuracy
                    )
                    logger.info(f"Initialized ultra-accurate HNSW with dimension {self.dimension}")
                
                # Add to HNSW
                start_idx = len(self.documents)
                ids = list(range(start_idx, start_idx + len(documents)))
                embeddings_array = np.array(embeddings, dtype=np.float32)
                self.hnsw_index.add_items(embeddings_array, ids)
                self.hnsw_index.set_ef(200)  # Higher ef for better recall
            
            # Enhanced sparse indexing
            new_tokenized = [self._ultra_tokenize_text(text) for text in texts]
            self.tokenized_docs.extend(new_tokenized)
            
            if self.tokenized_docs:
                self.bm25 = BM25Okapi(self.tokenized_docs)
            
            # Store everything
            self.documents.extend(documents)
            self.document_embeddings.extend(embeddings)
            
            logger.info(f"Successfully added {len(documents)} documents to ultra hybrid index")
            
        except Exception as e:
            logger.error(f"Error adding documents to ultra hybrid index: {str(e)}")
            raise
    
    def _ultra_tokenize_text(self, text: str) -> List[str]:
        """Ultra-enhanced tokenization for maximum accuracy."""
        text = text.lower()
        
        # Enhanced preprocessing
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        tokens = text.split()
        
        # Enhanced stop words (more comprehensive)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'it', 'its', 'he', 'she', 'they', 'them', 'their', 'there', 'where', 'when', 'how', 'why',
            'what', 'who', 'which', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now'
        }
        
        # Keep meaningful tokens
        filtered_tokens = []
        for token in tokens:
            if (len(token) > 2 and 
                token not in stop_words and 
                not token.isdigit() and 
                len(token) < 20):  # Remove very long tokens (likely OCR errors)
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """Ultra-accurate hybrid retrieval with MMR reranking."""
        try:
            if not self.documents:
                return []
            
            logger.info(f"Ultra hybrid retrieval for query: {query[:50]}...")
            
            # Step 1: Dense retrieval with higher recall
            dense_results = self._ultra_dense_retrieval(query, k * 6)
            
            # Step 2: Enhanced sparse retrieval
            sparse_results = self._ultra_sparse_retrieval(query, k * 6)
            
            # Step 3: Ultra-accurate combination
            combined_results = self._ultra_combine_results(dense_results, sparse_results)
            
            # Step 4: Ultra-accurate MMR reranking
            if self.mmr_reranker and len(combined_results) > k:
                final_results = self.mmr_reranker.ultra_mmr_rerank(query, combined_results, k)
                return [doc for doc, score in final_results]
            else:
                return combined_results[:k]
            
        except Exception as e:
            logger.error(f"Error in ultra hybrid retrieval: {str(e)}")
            return []
    
    def _ultra_dense_retrieval(self, query: str, k: int) -> List[Document]:
        """Ultra-accurate dense retrieval."""
        results = []
        
        try:
            if not self.hnsw_index:
                return results
            
            query_embedding = self.embeddings.embed_query(query)
            if not query_embedding:
                return results
            
            # Use higher k for better recall
            search_k = min(k, len(self.documents))
            query_array = np.array([query_embedding], dtype=np.float32)
            labels, distances = self.hnsw_index.knn_query(query_array, k=search_k)
            
            for label, distance in zip(labels[0], distances[0]):
                if label < len(self.documents):
                    doc = self.documents[label].copy()
                    similarity = 1 - distance
                    doc.metadata['dense_score'] = similarity
                    doc.metadata['retrieval_method'] = 'dense'
                    results.append(doc)
                    
        except Exception as e:
            logger.error(f"Error in ultra dense retrieval: {e}")
        
        return results
    
    def _ultra_sparse_retrieval(self, query: str, k: int) -> List[Document]:
        """Ultra-accurate sparse retrieval."""
        results = []
        
        try:
            if not self.bm25:
                return results
            
            query_tokens = self._ultra_tokenize_text(query)
            if not query_tokens:
                return results
            
            scores = self.bm25.get_scores(query_tokens)
            
            # Get top results with minimum score threshold
            min_score_threshold = 0.1
            top_indices = np.argsort(scores)[::-1]
            
            count = 0
            for idx in top_indices:
                if count >= k:
                    break
                if idx < len(self.documents) and scores[idx] > min_score_threshold:
                    doc = self.documents[idx].copy()
                    doc.metadata['sparse_score'] = float(scores[idx])
                    doc.metadata['retrieval_method'] = 'sparse'
                    results.append(doc)
                    count += 1
                    
        except Exception as e:
            logger.error(f"Error in ultra sparse retrieval: {e}")
        
        return results
    
    def _ultra_combine_results(self, dense_results: List[Document], 
                             sparse_results: List[Document]) -> List[Document]:
        """Ultra-accurate combination algorithm."""
        seen_hashes = set()
        combined = []
        
        # Enhanced score normalization
        def ultra_normalize_scores(docs, score_key):
            scores = [doc.metadata.get(score_key, 0) for doc in docs]
            if not scores or len(set(scores)) <= 1:
                return docs
            
            # Use robust normalization
            scores_array = np.array(scores)
            q75, q25 = np.percentile(scores_array, [75, 25])
            iqr = q75 - q25
            
            if iqr > 0:
                # Robust scaling
                for doc in docs:
                    original_score = doc.metadata.get(score_key, 0)
                    normalized = (original_score - q25) / iqr
                    normalized = max(0, min(1, normalized))  # Clamp to [0, 1]
                    doc.metadata[f'{score_key}_normalized'] = normalized
            else:
                # Fallback to simple normalization
                min_score, max_score = min(scores), max(scores)
                for doc in docs:
                    original_score = doc.metadata.get(score_key, 0)
                    normalized = (original_score - min_score) / (max_score - min_score) if max_score > min_score else 0.5
                    doc.metadata[f'{score_key}_normalized'] = normalized
            
            return docs
        
        # Normalize scores
        dense_results = ultra_normalize_scores(dense_results, 'dense_score')
        sparse_results = ultra_normalize_scores(sparse_results, 'sparse_score')
        
        # Ultra-accurate combination with adaptive weighting
        all_docs = []
        
        # Process dense results (primary weight: 0.7)
        for doc in dense_results:
            doc_hash = doc.metadata.get('chunk_hash', hash(doc.page_content))
            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)
                dense_score = doc.metadata.get('dense_score_normalized', 0)
                
                # Quality bonuses
                quality_bonus = 0.0
                
                # Semantic density bonus
                semantic_density = doc.metadata.get('semantic_density', 0)
                if semantic_density > 12:
                    quality_bonus += 0.1
                
                # Section info bonus
                if doc.metadata.get('section_info'):
                    quality_bonus += 0.05
                
                # Image reference bonus
                if doc.metadata.get('image_refs'):
                    quality_bonus += 0.05
                
                final_score = 0.7 * dense_score + quality_bonus
                doc.metadata['combined_score'] = final_score
                all_docs.append(doc)
        
        # Process sparse results (secondary weight: 0.3)
        for doc in sparse_results:
            doc_hash = doc.metadata.get('chunk_hash', hash(doc.page_content))
            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)
                sparse_score = doc.metadata.get('sparse_score_normalized', 0)
                doc.metadata['combined_score'] = 0.3 * sparse_score
                all_docs.append(doc)
            else:
                # Boost existing documents found by both methods
                for existing_doc in all_docs:
                    existing_hash = existing_doc.metadata.get('chunk_hash', hash(existing_doc.page_content))
                    if existing_hash == doc_hash:
                        sparse_score = doc.metadata.get('sparse_score_normalized', 0)
                        boost = 0.3 * sparse_score
                        existing_doc.metadata['combined_score'] += boost
                        existing_doc.metadata['found_by_both'] = True
                        # Additional bonus for being found by both methods
                        existing_doc.metadata['combined_score'] *= 1.15
                        break
        
        # Ultra-accurate final ranking
        all_docs.sort(key=lambda x: x.metadata.get('combined_score', 0), reverse=True)
        
        return all_docs

class UltraAccurateRAGSystem:
    """Ultra-accurate RAG system with full page image support."""
    
    def __init__(self, vertex_gen_ai: VertexGenAI = None):
        logger.info("Initializing UltraAccurateRAGSystem...")
        
        self.vertex_gen_ai = vertex_gen_ai
        self.processor = UltraDocumentProcessor(vertex_gen_ai) if vertex_gen_ai else None
        self.documents = {}
        self.ultra_retriever = None
        self.conversation_history = []
        self.reference_counter = 1
        
        if vertex_gen_ai:
            self.ultra_retriever = UltraHybridRetriever(vertex_gen_ai=vertex_gen_ai)
    
    def upload_pdf(self, file_path: str) -> str:
        """Process and index PDF with ultra-accurate analysis."""
        try:
            logger.info(f"Ultra-accurate processing: {file_path}")
            
            doc = self.processor.process_pdf(file_path)
            self.documents[doc.filename] = doc
            
            if self.ultra_retriever:
                self.ultra_retriever.add_documents(doc.langchain_documents)
            
            # Ultra-detailed statistics
            high_conf_images = sum(1 for img in doc.images if img.confidence > 0.7)
            ultra_semantic_chunks = sum(1 for chunk in doc.chunks if chunk.semantic_density > 15)
            embedded_chunks = sum(1 for chunk in doc.chunks if chunk.embedding is not None)
            page_images_count = len(doc.page_images)
            
            avg_confidence = np.mean([img.confidence for img in doc.images]) if doc.images else 0
            avg_semantic_density = np.mean([chunk.semantic_density for chunk in doc.chunks]) if doc.chunks else 0
            
            result = f"""✅ **Ultra-Accurate Processing Complete: {doc.filename}**

**Content Analysis:**
- 📄 Pages: {len(doc.pages)} (with {page_images_count} full page images)
- 🧩 Chunks: {len(doc.chunks)} ({embedded_chunks} with VertexAI embeddings)
- 🖼️ Region Images: {len(doc.images)} ({high_conf_images} ultra-high confidence)
- 🎯 Ultra-semantic chunks: {ultra_semantic_chunks}

**Ultra-Quality Metrics:**
- Average image confidence: {avg_confidence:.3f}
- Average semantic density: {avg_semantic_density:.1f}
- Processing method: Custom MMR + Ultra-HNSW + Enhanced BM25
- Page image coverage: 100%
"""
            
            logger.info(result.replace('\n', ' '))
            return result
            
        except Exception as e:
            error_msg = f"❌ **Ultra-accurate processing error:** {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def ask(self, query: str) -> Dict[str, Any]:
        """Ultra-accurate query processing with full page images."""
        if not self.ultra_retriever or not self.documents:
            return {
                "answer": "Please upload documents first.",
                "references": [],
                "images": [],
                "page_images": []
            }
        
        try:
            logger.info(f"Ultra-accurate query processing: {query[:50]}...")
            
            # Enhanced query preprocessing
            processed_query = self._ultra_enhance_query(query)
            
            # Ultra-accurate retrieval with MMR
            relevant_docs = self.ultra_retriever.get_relevant_documents(processed_query, k=6)
            
            if not relevant_docs:
                return {
                    "answer": "No relevant information found. Please verify the information exists in your documents or try rephrasing with different terms.",
                    "references": [],
                    "images": [],
                    "page_images": []
                }
            
            # Enhanced context preparation with page tracking
            context_parts = []
            references = []
            all_images = []
            used_pages = set()  # Track which pages are used for context
            
            for i, doc in enumerate(relevant_docs):
                metadata = doc.metadata
                doc_name = metadata.get('source', 'Unknown')
                pages = metadata.get('page_numbers', [])
                
                # Track used pages
                used_pages.update(pages)
                
                # Enhanced scoring information
                dense_score = metadata.get('dense_score', 0.0)
                sparse_score = metadata.get('sparse_score', 0.0)
                combined_score = metadata.get('combined_score', 0.0)
                mmr_score = metadata.get('mmr_score', 0.0)
                relevance_score = metadata.get('relevance_score', 0.0)
                semantic_density = metadata.get('semantic_density', 0.0)
                found_by_both = metadata.get('found_by_both', False)
                
                # Create ultra-enhanced reference
                ref_id = self.reference_counter
                reference = {
                    "id": ref_id,
                    "document": doc_name,
                    "pages": pages,
                    "dense_score": dense_score,
                    "sparse_score": sparse_score,
                    "combined_score": combined_score,
                    "mmr_score": mmr_score,
                    "relevance_score": relevance_score,
                    "semantic_density": semantic_density,
                    "found_by_both": found_by_both,
                    "content": metadata.get('original_text', doc.page_content)[:250] + "...",
                    "type": "text",
                    "retrieval_method": metadata.get('retrieval_method', 'hybrid')
                }
                references.append(reference)
                self.reference_counter += 1
                
                # Ultra-enhanced context
                section_info = metadata.get('section_info', {})
                section_text = ""
                if section_info:
                    sections = [f"{k}: {v}" for k, v in section_info.items()]
                    section_text = f" [Sections: {', '.join(sections[:3])}]"
                
                context_text = f"[{ref_id}] {doc.page_content}{section_text}"
                context_parts.append(context_text)
                
                # Ultra-accurate image retrieval
                image_refs = metadata.get('image_refs', [])
                if image_refs and doc_name in self.documents:
                    pdf_doc = self.documents[doc_name]
                    for img_idx in image_refs:
                        if img_idx < len(pdf_doc.images):
                            img_data = pdf_doc.images[img_idx]
                            
                            # Calculate ultra-accurate image relevance
                            img_relevance = self.processor.image_matcher.calculate_advanced_similarity(
                                processed_query, img_data
                            )
                            
                            if img_relevance > 0.35:  # High threshold for ultra accuracy
                                all_images.append({
                                    "document": doc_name,
                                    "page": img_data.page_number,
                                    "type": img_data.image_type,
                                    "analysis": img_data.analysis,
                                    "confidence": img_data.confidence,
                                    "base64": img_data.base64_string,
                                    "reference_id": self.reference_counter,
                                    "relevance_score": img_relevance,
                                    "ocr_text": img_data.ocr_text[:200],
                                    "keywords": img_data.semantic_keywords,
                                    "visual_features": img_data.visual_features
                                })
                                
                                # Add image reference
                                img_ref = {
                                    "id": self.reference_counter,
                                    "document": doc_name,
                                    "pages": [img_data.page_number],
                                    "combined_score": img_relevance,
                                    "content": f"{img_data.image_type}: {img_data.analysis[:200]}...",
                                    "type": "image",
                                    "image_confidence": img_data.confidence,
                                    "ultra_relevance": img_relevance
                                }
                                references.append(img_ref)
                                self.reference_counter += 1
            
            # Get full page images for used pages
            page_images = self._get_full_page_images(used_pages)
            
            # Select top 3 most relevant images
            top_images = sorted(
                all_images, 
                key=lambda x: x['relevance_score'] * x['confidence'], 
                reverse=True
            )[:3]
            
            # Ultra-enhanced context preparation
            context = self._prepare_ultra_context(context_parts, top_images)
            
            # Create reference links
            ref_links = {ref["id"]: f"Ref {ref['id']}" for ref in references}
            
            # Ultra-enhanced prompt
            prompt = self._create_ultra_prompt(processed_query, context, ref_links)
            
            # Generate response
            response = self.vertex_gen_ai.generate_content(prompt)
            
            if not response:
                response = "I couldn't generate a comprehensive response. Please try rephrasing your question with more specific terms or verify the information exists in your documents."
            
            # Add clickable reference links
            for ref_id, ref_text in ref_links.items():
                response = response.replace(f"[{ref_id}]", f"[[{ref_id}]](#ref{ref_id})")
            
            self.conversation_history.append((query, response))
            
            logger.info(f"Generated ultra-accurate response with {len(references)} references, {len(top_images)} images, and {len(page_images)} full pages")
            
            return {
                "answer": response,
                "references": references,
                "images": top_images,
                "page_images": page_images
            }
            
        except Exception as e:
            logger.error(f"Error in ultra-accurate processing: {str(e)}")
            return {
                "answer": f"Error in ultra-accurate processing: {str(e)}",
                "references": [],
                "images": [],
                "page_images": []
            }
    
    def _get_full_page_images(self, used_pages: Set[int]) -> List[Dict[str, Any]]:
        """Get full page images for pages used in context."""
        page_images = []
        
        try:
            for doc_name, doc in self.documents.items():
                for page_num in used_pages:
                    if page_num in doc.page_images:
                        page_data = doc.page_images[page_num]
                        page_images.append({
                            "document": doc_name,
                            "page": page_num,
                            "base64": page_data.base64_string,
                            "ocr_text": page_data.ocr_text[:300] if page_data.ocr_text else "",
                            "confidence": page_data.confidence
                        })
            
            # Sort by page number for consistency
            page_images.sort(key=lambda x: (x['document'], x['page']))
            
        except Exception as e:
            logger.error(f"Error getting full page images: {e}")
        
        return page_images
    
    def _ultra_enhance_query(self, query: str) -> str:
        """Ultra-enhanced query preprocessing."""
        enhanced_terms = []
        query_lower = query.lower()
        
        # Ultra-comprehensive enhancement mapping
        enhancement_mapping = {
            'chart': ['visualization', 'graph', 'data', 'statistics', 'plot', 'trend'],
            'graph': ['chart', 'visualization', 'data', 'plot', 'trend'],
            'table': ['data', 'comparison', 'structure', 'rows', 'columns', 'matrix'],
            'process': ['steps', 'methodology', 'workflow', 'procedure', 'flow'],
            'analysis': ['examination', 'study', 'evaluation', 'assessment', 'findings'],
            'results': ['findings', 'outcomes', 'conclusions', 'data', 'evidence'],
            'method': ['approach', 'technique', 'process', 'procedure'],
            'data': ['information', 'statistics', 'numbers', 'values'],
            'figure': ['image', 'illustration', 'visual', 'diagram'],
            'explain': ['describe', 'detail', 'clarify', 'elaborate'],
            'show': ['display', 'present', 'demonstrate', 'illustrate'],
            'compare': ['contrast', 'difference', 'similarity', 'relation']
        }
        
        # Add contextually relevant terms
        for key, terms in enhancement_mapping.items():
            if key in query_lower:
                enhanced_terms.extend(terms[:2])  # Limit to top 2 terms
        
        # Add enhanced terms without making query too long
        if enhanced_terms:
            unique_terms = list(set(enhanced_terms))[:3]  # Max 3 additional terms
            enhanced_query = f"{query} {' '.join(unique_terms)}"
        else:
            enhanced_query = query
        
        return enhanced_query
    
    def _prepare_ultra_context(self, context_parts: List[str], images: List[Dict]) -> str:
        """Prepare ultra-enhanced context."""
        context = "\n\n".join(context_parts)
        
        if images:
            image_context = "\n\n=== ULTRA-ACCURATE VISUAL CONTENT ANALYSIS ===\n"
            for img in images:
                visual_features = img.get('visual_features', {})
                
                # Top 3 visual features
                top_features = sorted(visual_features.items(), key=lambda x: x[1], reverse=True)[:3]
                feature_summary = ", ".join([f"{k}: {v:.3f}" for k, v in top_features])
                
                img_context = f"""
Ultra Image Reference [{img['reference_id']}] - {img['type'].title()} from {img['document']}, Page {img['page']}:
- Enhanced Analysis: {img['analysis']}
- OCR Content: {img['ocr_text']}
- Semantic Keywords: {', '.join(img['keywords'])}
- Ultra Relevance Score: {img['relevance_score']:.3f}
- Visual Features: {feature_summary}
- Confidence Level: {img['confidence']:.3f}
"""
                image_context += img_context
            
            context += image_context
        
        return context
    
    def _create_ultra_prompt(self, query: str, context: str, ref_links: Dict) -> str:
        """Create ultra-enhanced prompt for maximum accuracy."""
        
        prompt = f"""You are an expert research assistant with ULTRA-ACCURATE analysis capabilities. Your mission is to provide the most precise, comprehensive, and accurate answer using ONLY the provided context.

ULTRA-ACCURACY REQUIREMENTS (CRITICAL):
1. Use ONLY information explicitly present in the provided context - NO external knowledge
2. Add numbered citations [1], [2], etc. after EVERY factual statement - NO exceptions
3. For numerical data, charts, tables: provide EXACT values, precise measurements, specific statistics
4. For visual content: provide detailed, technical descriptions with specific observations
5. If ANY information cannot be determined from context, explicitly state: "This information is not available in the provided context"
6. Cross-reference multiple sources when the same information appears in different chunks
7. Never make assumptions, estimates, or add interpretative content not explicitly stated

RESPONSE STRUCTURE (MANDATORY):
1. Direct, precise answer with immediate citations
2. Detailed supporting evidence with comprehensive explanations  
3. Technical visual content analysis (if applicable) with specific details
4. Cross-validation summary when multiple sources confirm the same information
5. Explicit limitations and gaps in available information

CONTEXT WITH ULTRA-ACCURATE ANALYSIS:
{context}

AVAILABLE REFERENCE IDs: {list(ref_links.keys())}

USER QUESTION: {query}

VISUAL CONTENT ANALYSIS REQUIREMENTS:
- For charts/graphs: Specify EXACT values, scales, trends, axes labels, data points
- For tables: Detail PRECISE structure, specific values, exact relationships, complete comparisons  
- For diagrams: Explain components step-by-step, exact connections, specific processes
- Always cite the specific image reference number with confidence scores

Provide an ULTRA-ACCURATE, comprehensively researched response with precise citations for EVERY factual claim:"""

        return prompt
    
    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
        self.reference_counter = 1
        return "Conversation reset successfully."

# Ultra-accurate interface
class UltraAccurateInterface:
    """Interface for ultra-accurate RAG system with full page images."""

    def __init__(self):
        self.rag_system = None
        self.vertex_gen_ai = None
        self.current_references = []
        self.current_images = []
        self.current_page_images = []
        logger.info("Initialized UltraAccurateInterface")
        self.setup_interface()

    def setup_interface(self):
        """Set up the ultra-accurate interface."""
        
        custom_css = """
        .logo-container {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .logo-text {
            font-size: 28px;
            font-weight: bold;
            color: white;
            text-align: center;
        }
        .ultra-badge {
            background: linear-gradient(45deg, #ff6b6b, #feca57);
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 12px;
            margin-left: 10px;
        }
        .reference-card {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            background: linear-gradient(135deg, #f8f9ff 0%, #f1f3ff 100%);
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .image-card {
            border: 1px solid #ddd;
            border-radius: 12px;
            padding: 15px;
            margin: 10px 0;
            background: linear-gradient(135deg, #fff8f0 0%, #fff4e6 100%);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .score-badge {
            display: inline-block;
            background: #007bff;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            margin-left: 5px;
        }
        .ultra-score {
            background: #ff6b6b;
        }
        .mmr-score {
            background: #28a745;
        }
        .page-image-card {
            border: 2px solid #007bff;
            border-radius: 12px;
            padding: 15px;
            margin: 10px 0;
            background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        """

        with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Ultra-Accurate RAG with Full Pages") as self.interface:
            
            # Enhanced header
            gr.HTML("""
            <div class="logo-container">
                <div style="margin-right: 15px; font-size: 32px;">🎯</div>
                <div class="logo-text">
                    Ultra-Accurate RAG System
                    <div class="ultra-badge">Custom MMR + HNSW + BM25 + Full Page Images</div>
                </div>
            </div>
            """)
            
            # Enhanced chatbot
            chatbot = gr.Chatbot(
                height=500,
                show_label=False,
                avatar_images=("👤", "🔬"),
                bubble_full_width=False,
                show_copy_button=True,
                likeable=True
            )
            
            # Input with enhanced placeholder
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask ultra-detailed questions (e.g., 'What are the exact values in the revenue chart on page 5?', 'Explain each step in the process diagram')",
                    show_label=False,
                    lines=2,
                    scale=5
                )
            
            # Enhanced three buttons
            with gr.Row():
                upload_btn = gr.Button("📤 Upload Documents", variant="primary", size="lg", scale=1)
                ask_btn = gr.Button("🎯 Ask (Ultra-Accurate)", variant="secondary", size="lg", scale=1) 
                reset_btn = gr.Button("🔄 Reset", variant="stop", size="lg", scale=1)
            
            upload_files = gr.File(
                label="Select PDF Documents for Ultra-Accurate Analysis",
                file_types=[".pdf"],
                file_count="multiple",
                visible=False
            )
            
            status_display = gr.Markdown("🤖 **Status:** Ready to initialize ultra-accurate system (VertexAI embeddings only)")
            
            # Enhanced references section
            gr.Markdown("## 📚 Ultra-Accurate References (with MMR Scores)")
            text_references = gr.HTML(value="<p>No references yet. Upload documents and ask questions to see ultra-accurate MMR-ranked references.</p>")
            
            # Enhanced images section
            gr.Markdown("## 🖼️ Ultra-Accurate Visual Analysis (Top 3 by Advanced Similarity)")
            
            with gr.Row():
                image1 = gr.Image(label="Highest Relevance", visible=False, height=250)
                image2 = gr.Image(label="Second Highest", visible=False, height=250)
                image3 = gr.Image(label="Third Highest", visible=False, height=250)
            
            image_info = gr.HTML(value="<p>No visual content found yet. Upload documents with charts, diagrams, or tables for ultra-accurate analysis.</p>")
            
            # Full page images section
            gr.Markdown("## 📄 Full Page Images (Pages Used in Response)")
            
            with gr.Row():
                page_image1 = gr.Image(label="Referenced Page 1", visible=False, height=400)
                page_image2 = gr.Image(label="Referenced Page 2", visible=False, height=400)
                page_image3 = gr.Image(label="Referenced Page 3", visible=False, height=400)
            
            page_image_info = gr.HTML(value="<p>Full page images will appear here when content from specific pages is used to generate responses.</p>")

            initialized = gr.State(False)

            # Enhanced event handlers
            def handle_upload_click():
                return gr.update(visible=True)
            
            def handle_upload_files(files, is_initialized):
                if not is_initialized:
                    try:
                        self.vertex_gen_ai = VertexGenAI()
                        test_response = self.vertex_gen_ai.generate_content("Test ultra-accurate system")
                        if test_response:
                            self.rag_system = UltraAccurateRAGSystem(vertex_gen_ai=self.vertex_gen_ai)
                            is_initialized = True
                        else:
                            return "❌ Failed to initialize VertexAI for ultra-accurate processing", False, gr.update(visible=False)
                    except Exception as e:
                        return f"❌ Ultra-accurate initialization error: {str(e)}", False, gr.update(visible=False)
                
                if not files:
                    return "❌ No files selected for ultra-accurate processing", is_initialized, gr.update(visible=False)
                
                results = []
                for file in files:
                    try:
                        result = self.rag_system.upload_pdf(file.name)
                        results.append(result)
                    except Exception as e:
                        results.append(f"❌ Error in ultra-accurate processing {os.path.basename(file.name)}: {str(e)}")
                
                status = "## 📄 Ultra-Accurate Processing Results\n" + "\n".join(results)
                return status, is_initialized, gr.update(visible=False)
            
            def handle_ask(message, history, is_initialized):
                if not is_initialized or not self.rag_system:
                    error_msg = "❌ Please upload documents first to enable ultra-accurate processing"
                    if history is None:
                        history = []
                    return history + [[message, error_msg]], "", "", "", "", "", "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                
                if not message.strip():
                    if history is None:
                        history = []
                    return history, message, "", "", "", "", "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                
                try:
                    # Get ultra-accurate response
                    result = self.rag_system.ask(message)
                    answer = result["answer"]
                    self.current_references = result["references"]
                    self.current_images = result["images"]
                    self.current_page_images = result["page_images"]
                    
                    # Update chat history
                    if history is None:
                        history = []
                    updated_history = history + [[message, answer]]
                    
                    # Format ultra-accurate references
                    text_refs_html = self._format_ultra_references()
                    
                    # Format ultra-accurate image info
                    images_info_html = self._format_ultra_images()
                    
                    # Format page image info
                    page_images_info_html = self._format_page_images()
                    
                    # Prepare images for display
                    img1, img2, img3 = self._prepare_ultra_images()
                    
                    # Prepare page images for display
                    page_img1, page_img2, page_img3 = self._prepare_page_images()
                    
                    return (updated_history, "", text_refs_html, images_info_html, page_images_info_html, "",
                           img1, img2, img3, page_img1, page_img2, page_img3)
                    
                except Exception as e:
                    error_msg = f"❌ Ultra-accurate processing error: {str(e)}"
                    if history is None:
                        history = []
                    return (history + [[message, error_msg]], "", "", "", "", "",
                           gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                           gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
            
            def handle_reset():
                if self.rag_system:
                    self.rag_system.reset_conversation()
                self.current_references = []
                self.current_images = []
                self.current_page_images = []
                return ([], "🔄 Ultra-accurate conversation reset", 
                       "<p>No references yet.</p>", 
                       "<p>No images yet.</p>",
                       "<p>No page images yet.</p>",
                       gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                       gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
            
            # Wire up events
            upload_btn.click(fn=handle_upload_click, outputs=upload_files)
            upload_files.upload(fn=handle_upload_files, inputs=[upload_files, initialized], outputs=[status_display, initialized, upload_files])
            ask_btn.click(fn=handle_ask, inputs=[msg_input, chatbot, initialized], outputs=[chatbot, msg_input, text_references, image_info, page_image_info, status_display, image1, image2, image3, page_image1, page_image2, page_image3])
            msg_input.submit(fn=handle_ask, inputs=[msg_input, chatbot, initialized], outputs=[chatbot, msg_input, text_references, image_info, page_image_info, status_display, image1, image2, image3, page_image1, page_image2, page_image3])
            reset_btn.click(fn=handle_reset, outputs=[chatbot, status_display, text_references, image_info, page_image_info, image1, image2, image3, page_image1, page_image2, page_image3])

    def _format_ultra_references(self):
        """Format references with ultra-accurate scoring information."""
        if not self.current_references:
            return "<p>No references available.</p>"
        
        html_parts = []
        text_refs = [ref for ref in self.current_references if ref.get("type") != "image"]
        
        for ref in text_refs:
            ref_id = ref["id"]
            doc = ref["document"]
            pages = ref["pages"]
            combined_score = ref.get("combined_score", 0.0)
            dense_score = ref.get("dense_score", 0.0)
            sparse_score = ref.get("sparse_score", 0.0)
            mmr_score = ref.get("mmr_score", 0.0)
            relevance_score = ref.get("relevance_score", 0.0)
            semantic_density = ref.get("semantic_density", 0.0)
            found_by_both = ref.get("found_by_both", False)
            retrieval_method = ref.get("retrieval_method", "hybrid")
            content = ref["content"]
            
            # Format page numbers
            if len(pages) > 1:
                page_str = f"pp. {min(pages)}-{max(pages)}"
            else:
                page_str = f"p. {pages[0]}" if pages else "p. ?"
            
            ref_html = f"""
            <div class="reference-card" id="ref{ref_id}">
                <strong>[{ref_id}]</strong> 📄 {doc}, {page_str}
                <span class="score-badge ultra-score">Combined: {combined_score:.3f}</span>
                <span class="score-badge">Dense: {dense_score:.3f}</span>
                <span class="score-badge">Sparse: {sparse_score:.3f}</span>
                {f'<span class="score-badge mmr-score">MMR: {mmr_score:.3f}</span>' if mmr_score > 0 else ''}
                {f'<span class="score-badge">Relevance: {relevance_score:.3f}</span>' if relevance_score > 0 else ''}
                {f'<span class="score-badge mmr-score">Both Methods</span>' if found_by_both else ''}
                <br><small>
                    <strong>Method:</strong> {retrieval_method.title()} | 
                    <strong>Semantic Density:</strong> {semantic_density:.1f} | 
                    <strong>Quality:</strong> {'Ultra-High' if combined_score > 0.8 else 'High' if combined_score > 0.6 else 'Medium' if combined_score > 0.4 else 'Low'}
                </small>
                <br><em>"{content}"</em>
            </div>
            """
            html_parts.append(ref_html)
        
        if not html_parts:
            return "<p>No text references found.</p>"
        
        return "".join(html_parts)

    def _format_ultra_images(self):
        """Format image information with ultra-accurate details."""
        if not self.current_images:
            return "<p>No relevant visual content found for this query using ultra-accurate analysis.</p>"
        
        html_parts = []
        for i, img in enumerate(self.current_images):
            doc = img["document"]
            page = img["page"]
            img_type = img["type"]
            analysis = img["analysis"]
            confidence = img["confidence"]
            relevance_score = img["relevance_score"]
            ref_id = img.get("reference_id", "")
            ocr_preview = img.get("ocr_text", "")[:150]
            keywords = ", ".join(img.get("keywords", []))
            visual_features = img.get("visual_features", {})
            
            # Top visual features
            top_features = sorted(visual_features.items(), key=lambda x: x[1], reverse=True)[:3]
            feature_summary = ", ".join([f"{k}: {v:.3f}" for k, v in top_features])
            
            # Calculate combined score
            combined_score = relevance_score * confidence
            
            img_html = f"""
            <div class="image-card">
                <strong>Image {i+1} {f'[{ref_id}]' if ref_id else ''}</strong> 🖼️ {img_type.title()}
                <span class="score-badge ultra-score">Relevance: {relevance_score:.3f}</span>
                <span class="score-badge">Confidence: {confidence:.3f}</span>
                <span class="score-badge mmr-score">Final: {combined_score:.3f}</span>
                <br><strong>Source:</strong> {doc}, Page {page}
                <br><strong>Ultra Analysis:</strong> {analysis}
                {f'<br><strong>OCR Content:</strong> "{ocr_preview}..."' if ocr_preview else ''}
                {f'<br><strong>Keywords:</strong> {keywords}' if keywords else ''}
                {f'<br><strong>Visual Features:</strong> {feature_summary}' if feature_summary else ''}
            </div>
            """
            html_parts.append(img_html)
        
        return "".join(html_parts)

    def _format_page_images(self):
        """Format full page image information."""
        if not self.current_page_images:
            return "<p>No full page images available. Page images will appear when content from specific pages is used in responses.</p>"
        
        html_parts = []
        for i, page_img in enumerate(self.current_page_images):
            doc = page_img["document"]
            page = page_img["page"]
            confidence = page_img["confidence"]
            ocr_preview = page_img.get("ocr_text", "")[:200]
            
            page_html = f"""
            <div class="page-image-card">
                <strong>📄 Full Page {i+1}</strong> - {doc}, Page {page}
                <span class="score-badge">Confidence: {confidence:.3f}</span>
                <br><strong>Usage:</strong> Content from this page was used to generate the response
                {f'<br><strong>Page Content Preview:</strong> "{ocr_preview}..."' if ocr_preview else ''}
            </div>
            """
            html_parts.append(page_html)
        
        return "".join(html_parts)

    def _prepare_ultra_images(self):
        """Prepare ultra-accurate images for display."""
        img1 = gr.update(visible=False)
        img2 = gr.update(visible=False) 
        img3 = gr.update(visible=False)
        
        if not self.current_images:
            return img1, img2, img3
        
        display_images = []
        for img_data in self.current_images:
            try:
                img_bytes = base64.b64decode(img_data["base64"])
                pil_image = Image.open(BytesIO(img_bytes))
                display_images.append(pil_image)
            except Exception as e:
                logger.error(f"Error converting image: {str(e)}")
                continue
        
        # Enhanced labels with ultra-accurate scores
        if len(display_images) > 0:
            img_data = self.current_images[0]
            score = img_data['relevance_score'] * img_data['confidence']
            img1 = gr.update(
                value=display_images[0], 
                visible=True,
                label=f"🥇 {img_data['type'].title()} (Ultra Score: {score:.3f}) - {img_data['document']}, Page {img_data['page']}"
            )
        
        if len(display_images) > 1:
            img_data = self.current_images[1]
            score = img_data['relevance_score'] * img_data['confidence']
            img2 = gr.update(
                value=display_images[1],
                visible=True,
                label=f"🥈 {img_data['type'].title()} (Ultra Score: {score:.3f}) - {img_data['document']}, Page {img_data['page']}"
            )
        
        if len(display_images) > 2:
            img_data = self.current_images[2]
            score = img_data['relevance_score'] * img_data['confidence']
            img3 = gr.update(
                value=display_images[2],
                visible=True,
                label=f"🥉 {img_data['type'].title()} (Ultra Score: {score:.3f}) - {img_data['document']}, Page {img_data['page']}"
            )
        
        return img1, img2, img3

    def _prepare_page_images(self):
        """Prepare full page images for display."""
        page_img1 = gr.update(visible=False)
        page_img2 = gr.update(visible=False)
        page_img3 = gr.update(visible=False)
        
        if not self.current_page_images:
            return page_img1, page_img2, page_img3
        
        display_page_images = []
        for page_data in self.current_page_images[:3]:  # Top 3 pages
            try:
                img_bytes = base64.b64decode(page_data["base64"])
                pil_image = Image.open(BytesIO(img_bytes))
                display_page_images.append(pil_image)
            except Exception as e:
                logger.error(f"Error converting page image: {str(e)}")
                continue
        
        # Display full page images
        if len(display_page_images) > 0:
            page_data = self.current_page_images[0]
            page_img1 = gr.update(
                value=display_page_images[0],
                visible=True,
                label=f"📄 Full Page - {page_data['document']}, Page {page_data['page']} (Used in Response)"
            )
        
        if len(display_page_images) > 1:
            page_data = self.current_page_images[1]
            page_img2 = gr.update(
                value=display_page_images[1],
                visible=True,
                label=f"📄 Full Page - {page_data['document']}, Page {page_data['page']} (Used in Response)"
            )
        
        if len(display_page_images) > 2:
            page_data = self.current_page_images[2]
            page_img3 = gr.update(
                value=display_page_images[2],
                visible=True,
                label=f"📄 Full Page - {page_data['document']}, Page {page_data['page']} (Used in Response)"
            )
        
        return page_img1, page_img2, page_img3

def launch_ultra_accurate_rag():
    """Launch the ultra-accurate RAG interface."""
    interface = UltraAccurateInterface()
    return interface.interface

def main():
    """Main function with ultra-accurate system."""
    print("🎯 Starting Ultra-Accurate RAG System with Full Page Images")
    
    try:
        import google.colab
        print("📱 Running in Google Colab")
        from google.colab import auth
        auth.authenticate_user()
        print("✅ Authentication completed")
    except ImportError:
        print("💻 Running in local environment")
    
    print("\n🚀 **Ultra-Accurate Features (VertexAI Only):**")
    print("- 🎯 Custom MMR (Maximal Marginal Relevance) reranking")
    print("- 🔍 Hybrid: Ultra-HNSW + Enhanced BM25")
    print("- 🖼️ Advanced image-text similarity (VertexAI embeddings only)")
    print("- 📄 Full PDF page images for referenced pages")
    print("- 📊 Custom visual feature extraction and analysis")
    print("- 🔗 Multi-stage ultra-accurate scoring and filtering")
    print("- 📈 Comprehensive quality metrics and confidence scoring")
    print("- ⚡ Optimized for maximum accuracy without external dependencies")
    print("- 🎯 Returns complete page images when content is used in responses")
    
    interface = launch_ultra_accurate_rag()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=False,
        show_error=True
    )

if __name__ == "__main__":
    main()
