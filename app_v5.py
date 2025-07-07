# -*- coding: utf-8 -*-
"""Highly-Accurate-RAG-with-Hybrid-Retrieval-and-Reranking.ipynb

Enhanced RAG system with:
1. Hybrid retrieval (Dense HNSW + Sparse BM25)
2. Cross-encoder reranking for accuracy
3. Semantic image-text matching
4. Multi-stage relevance filtering
5. Query expansion and refinement
6. Better chunking and context preservation
7. Inline sentence-level citations
8. Interactive reference modal with complete source display
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

# Configure logging - only show warnings and errors
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import pypdf
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import cv2

# Enhanced retrieval libraries
import hnswlib
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
import gradio as gr

# Your existing VertexAI code
import vertexai
from google.oauth2.credentials import Credentials
from vertexai.generative_models import GenerativeModel

# --- Start of Added Groq Support ---
import groq
# --- End of Added Groq Support ---


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

# --- Start of Added GroqGenAI Class ---
class GroqGenAI:
    """Wrapper for Groq API to be interchangeable with VertexGenAI."""
    def __init__(self):
        try:
            self.api_key = os.environ['GROQ_API_KEY']
        except KeyError:
            raise ValueError("To use Groq, please set the GROQ_API_KEY environment variable.")
        
        self.client = groq.Groq(api_key=self.api_key)
        # For get_embeddings, we'll use a local model as Groq doesn't provide an embedding API.
        # Using a model already present in the script for consistency.
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded SentenceTransformer for Groq embeddings")
        except Exception as e:
            logger.error(f"Could not load SentenceTransformer model for Groq: {e}")
            self.embedding_model = None

    def generate_content(self, prompt: str = "Provide interesting trivia"):
        """Generate content based on the provided prompt using Groq."""
        if not self.client:
            return "Groq client not initialized."
            
        try:
            resp = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama3-70b-8192",  # Using a powerful model for high-quality generation
            )
            return resp.choices[0].message.content if resp.choices else None
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            return None

    def get_embeddings(self, texts: list[str], model_name: str = "local_sentence_transformer"):
        """Get embeddings for a list of texts using a local SentenceTransformer."""
        if self.embedding_model is None:
            logger.error("SentenceTransformer model not available for embeddings.")
            # Return zero vectors of a dimension that matches vertex typically (768)
            return [[0.0] * 768 for _ in texts]

        if not texts:
            return []
        
        if isinstance(texts, str):
            texts = [texts]

        try:
            # The output of sentence-transformer can be directly used.
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            # The existing VertexAIEmbeddings class handles normalization.
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Failed to generate embeddings with SentenceTransformer: {e}")
            return [[0.0] * 768 for _ in texts]
# --- End of Added GroqGenAI Class ---


@dataclass
class EnhancedImageData:
    """Enhanced image data with semantic analysis."""
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
    context_text: str = ""  # Surrounding text context

@dataclass
class EnhancedChunk:
    """Enhanced chunk with better context preservation."""
    text: str
    page_numbers: List[int]
    start_char_idx: int
    end_char_idx: int
    filename: str
    section_info: Dict[str, str] = field(default_factory=dict)
    image_refs: List[int] = field(default_factory=list)
    chunk_hash: str = ""
    context_before: str = ""  # Context before this chunk
    context_after: str = ""   # Context after this chunk
    semantic_density: float = 0.0  # Measure of information density
    
    def __post_init__(self):
        content = f"{self.text}{self.filename}{self.page_numbers}"
        self.chunk_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        
        # Calculate semantic density (keywords per sentence ratio)
        sentences = len([s for s in self.text.split('.') if s.strip()])
        words = len(self.text.split())
        self.semantic_density = words / max(sentences, 1)

    def to_document(self) -> Document:
        # Include more context in the document
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
                "original_text": self.text  # Keep original for reference
            }
        )

@dataclass
class PDFDocument:
    """Enhanced PDF document."""
    filename: str
    content: str = ""
    pages: Dict[int, str] = field(default_factory=dict)
    char_to_page_map: List[int] = field(default_factory=list)
    chunks: List[EnhancedChunk] = field(default_factory=list)
    images: List[EnhancedImageData] = field(default_factory=list)
    page_to_images: Dict[int, List[int]] = field(default_factory=dict)
    semantic_sections: Dict[str, List[int]] = field(default_factory=dict)
    page_images: Dict[int, str] = field(default_factory=dict)  # Store full page images as base64

    @property
    def langchain_documents(self) -> List[Document]:
        return [chunk.to_document() for chunk in self.chunks]

class SemanticImageMatcher:
    """Semantic matching between queries and images."""
    
    def __init__(self):
        # Load a lightweight sentence transformer for image-text matching
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence transformer for image matching")
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
            self.sentence_model = None
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract semantic keywords from text."""
        # Simple keyword extraction - could be enhanced with NER
        keywords = []
        
        # Domain-specific keywords
        chart_keywords = ['chart', 'graph', 'plot', 'data', 'trend', 'axis', 'value', 'percent', 'rate']
        table_keywords = ['table', 'row', 'column', 'cell', 'data', 'list', 'entry']
        diagram_keywords = ['diagram', 'flow', 'process', 'step', 'arrow', 'connection', 'structure']
        
        text_lower = text.lower()
        
        for word_list, type_name in [(chart_keywords, 'chart'), (table_keywords, 'table'), (diagram_keywords, 'diagram')]:
            if any(word in text_lower for word in word_list):
                keywords.extend([w for w in word_list if w in text_lower])
        
        # Extract numbers and percentages
        numbers = re.findall(r'\d+\.?\d*%?', text)
        keywords.extend(numbers[:5])  # Limit to first 5 numbers
        
        return list(set(keywords))
    
    def calculate_image_relevance(self, query: str, image: EnhancedImageData) -> float:
        """Calculate semantic relevance between query and image."""
        if not self.sentence_model:
            # Fallback to keyword matching
            return self._keyword_similarity(query, image)
        
        try:
            # Get embeddings
            query_embedding = self.sentence_model.encode([query])[0]
            
            # Combine OCR text and analysis for image representation
            image_text = f"{image.ocr_text} {image.analysis} {' '.join(image.semantic_keywords)}"
            if not image_text.strip():
                return 0.0
                
            image_embedding = self.sentence_model.encode([image_text])[0]
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, image_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(image_embedding)
            )
            
            # Boost score based on image confidence and type relevance
            type_boost = self._get_type_relevance_boost(query, image.image_type)
            confidence_boost = image.confidence * 0.3
            
            final_score = similarity + type_boost + confidence_boost
            return min(final_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating image relevance: {e}")
            return self._keyword_similarity(query, image)
    
    def _keyword_similarity(self, query: str, image: EnhancedImageData) -> float:
        """Fallback keyword-based similarity."""
        query_words = set(query.lower().split())
        image_words = set(image.ocr_text.lower().split()) | set(image.semantic_keywords)
        
        if not image_words:
            return 0.0
            
        intersection = query_words & image_words
        union = query_words | image_words
        
        jaccard_sim = len(intersection) / len(union) if union else 0.0
        return jaccard_sim * image.confidence
    
    def _get_type_relevance_boost(self, query: str, image_type: str) -> float:
        """Boost score based on query-image type relevance."""
        query_lower = query.lower()
        
        type_boosts = {
            'chart': 0.2 if any(word in query_lower for word in ['chart', 'graph', 'data', 'trend', 'statistics']) else 0.0,
            'table': 0.2 if any(word in query_lower for word in ['table', 'data', 'list', 'comparison']) else 0.0,
            'diagram': 0.2 if any(word in query_lower for word in ['diagram', 'process', 'flow', 'structure']) else 0.0,
            'figure': 0.1  # General boost for figures
        }
        
        return type_boosts.get(image_type, 0.0)

class AdvancedPDFProcessor:
    """Advanced PDF processor with enhanced chunking and image analysis."""

    def __init__(self):
        self.chunk_size = 600  # Smaller chunks for better precision
        self.chunk_overlap = 150  # More overlap for context preservation
        self.context_window = 200  # Characters of context before/after chunks
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )
        
        self.image_matcher = SemanticImageMatcher()
        
        # Enhanced section patterns
        self.section_patterns = [
            r'(?:Section|SECTION|Chapter|CHAPTER)\s+(\d+(?:\.\d+)*)[:\s]+(.*?)(?=\n|$)',
            r'^(\d+(?:\.\d+)*)\s+(.*?)$',
            r'^([A-Z][A-Z\s]{2,20})$',
            r'^(Abstract|Introduction|Methodology|Methods|Results|Discussion|Conclusion|References)$',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):?\s*$'
        ]

    def extract_images_from_pdf(self, file_path: str) -> Tuple[List[EnhancedImageData], Dict[int, str]]:
        """Extract and analyze images with enhanced semantic processing, also return page images."""
        images = []
        page_images = {}
        
        try:
            # Read PDF content first to get context
            pdf_text = self._extract_raw_text(file_path)
            pdf_images = convert_from_path(file_path, dpi=200)
            
            for page_num, page_image in enumerate(pdf_images, 1):
                # Store full page image
                buffered = BytesIO()
                page_image.save(buffered, format="PNG", quality=85)
                page_images[page_num] = base64.b64encode(buffered.getvalue()).decode()
                
                # Get page text for context
                page_context = self._get_page_context(pdf_text, page_num)
                
                # Process images on this page
                page_extracted_images = self._extract_page_images(page_image, page_num, page_context)
                images.extend(page_extracted_images)
                        
        except Exception as e:
            logger.error(f"Error extracting images from PDF: {str(e)}")
            
        return images, page_images

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
        """Get text context around a page."""
        context_pages = []
        
        # Include current page and adjacent pages for context
        for p in range(max(1, page_num - 1), min(len(pdf_text) + 1, page_num + 2)):
            if p in pdf_text:
                context_pages.append(pdf_text[p][:500])  # First 500 chars of each page
        
        return " ".join(context_pages)

    def _extract_page_images(self, page_image: Image.Image, page_num: int, 
                             page_context: str) -> List[EnhancedImageData]:
        """Extract images from a single page with context."""
        images = []
        
        try:
            opencv_image = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
            if opencv_image is None or opencv_image.size == 0:
                logger.error("opencv_image is empty or None")
                return images
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            if gray is None or gray.size == 0:
                logger.error("gray image is empty or None")
                return images
            # Enhanced edge detection
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            if blurred is None or blurred.size == 0:
                logger.error("blurred image is empty or None")
                return images
            edges = cv2.Canny(blurred, 30, 80)
            if edges is None or edges.size == 0:
                logger.error("edges image is empty or None")
                return images
            # Morphological operations to connect nearby edges
            kernel = np.ones((3, 3), np.uint8)
            if kernel is None or kernel.size == 0:
                logger.error("kernel is empty or None")
                return images
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours is None or len(contours) == 0:
                logger.warning("No contours found on page %d", page_num)
                return images
            # Filter and sort contours by area
            min_area = 5000
            valid_contours = []
            for c in contours:
                try:
                    area = cv2.contourArea(c)
                    if area > min_area:
                        valid_contours.append((area, c))
                except Exception as e:
                    logger.warning(f"Error calculating contour area: {e}")
                    continue
            if len(valid_contours) == 0:
                logger.info(f"No valid contours found on page {page_num}")
                return images
            valid_contours.sort(reverse=True)  # Largest first
            
            processed_count = 0
            for area, contour in valid_contours[:3]:  # Top 3 largest regions
                try:
                    image_data = self._process_image_region(
                        page_image, contour, page_num, page_context, processed_count
                    )
                    if image_data is not None and hasattr(image_data, 'confidence') and image_data.confidence > 0.4:  # Only high-confidence images
                        images.append(image_data)
                        processed_count += 1
                except Exception as e:
                    logger.warning(f"Error processing image region: {e}")
                    continue
                    
        except Exception as e:
            # Extra debug for ambiguous truth value errors
            import sys
            err_msg = str(e)
            if 'ambiguous' in err_msg:
                logger.error(f"Ambiguous truth value error: {err_msg}")
                logger.error(f"Type of opencv_image: {type(opencv_image)}, shape: {getattr(opencv_image, 'shape', None)}")
                logger.error(f"Type of gray: {type(gray)}, shape: {getattr(gray, 'shape', None)}")
                logger.error(f"Type of blurred: {type(blurred)}, shape: {getattr(blurred, 'shape', None)}")
                logger.error(f"Type of edges: {type(edges)}, shape: {getattr(edges, 'shape', None)}")
                logger.error(f"Type of kernel: {type(kernel)}, shape: {getattr(kernel, 'shape', None)}")
                logger.error(f"Type of contours: {type(contours)}, len: {len(contours) if contours is not None else None}")
            logger.error(f"Error extracting page images: {err_msg}")
            
        return images

    def _process_image_region(self, page_image: Image.Image, contour, 
                              page_num: int, page_context: str, 
                              region_idx: int) -> Optional[EnhancedImageData]:
        """Process a single image region."""
        try:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Ensure we have valid dimensions
            if w <= 0 or h <= 0:
                return None
            
            # Filter out very thin regions (likely text lines)
            aspect_ratio = w / h
            if aspect_ratio > 10 or aspect_ratio < 0.1:
                return None
                
            # Ensure coordinates are within image bounds
            img_width, img_height = page_image.size
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))
            w = min(w, img_width - x)
            h = min(h, img_height - y)
            
            if w <= 0 or h <= 0:
                return None
                
            roi = page_image.crop((x, y, x + w, y + h))
            
            # Enhanced OCR with better configuration
            ocr_text = ""
            try:
                # Try different PSM modes for better results
                for psm in [6, 8, 3]:
                    try:
                        ocr_result = pytesseract.image_to_string(
                            roi, 
                            config=f'--psm {psm} -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,%()-+/: '
                        )
                        if len(ocr_result.strip()) > len(ocr_text.strip()):
                            ocr_text = ocr_result
                    except:
                        continue
                        
            except Exception as e:
                logger.warning(f"OCR failed for region {region_idx}: {e}")
            
            # Enhanced image analysis
            image_type, analysis, confidence = self._analyze_image_enhanced(roi, ocr_text, page_context)
            
            if confidence < 0.4:  # Skip low-confidence detections
                return None
            
            # Extract semantic keywords
            keywords = self.image_matcher.extract_keywords(f"{ocr_text} {analysis}")
            
            # Convert to base64
            buffered = BytesIO()
            roi.save(buffered, format="PNG", quality=85)
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
                context_text=page_context[:200]  # First 200 chars of page context
            )
            
        except Exception as e:
            logger.error(f"Error processing image region: {e}")
            return None

    def _analyze_image_enhanced(self, image: Image.Image, ocr_text: str, 
                                context: str) -> Tuple[str, str, float]:
        """Enhanced image analysis with context."""
        try:
            img_array = np.array(image)
            if img_array is None or img_array.size == 0:
                logger.error("Image array is empty or None")
                return "figure", "Image array is empty", 0.0
            height, width = img_array.shape[:2]
            
            # Calculate image statistics
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
            if gray is None or gray.size == 0:
                logger.error("Gray image array is empty or None")
                return "figure", "Gray image array is empty", 0.0
            # Defensive: never use a numpy array in a boolean context
            # Detect lines (for tables/charts)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            if horizontal_kernel is None or horizontal_kernel.size == 0:
                logger.error("horizontal_kernel is empty or None")
                return "figure", "horizontal_kernel is empty", 0.0
            if vertical_kernel is None or vertical_kernel.size == 0:
                logger.error("vertical_kernel is empty or None")
                return "figure", "vertical_kernel is empty", 0.0
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
            if horizontal_lines is None or horizontal_lines.size == 0:
                logger.error("horizontal_lines is empty or None")
                return "figure", "horizontal_lines is empty", 0.0
            if vertical_lines is None or vertical_lines.size == 0:
                logger.error("vertical_lines is empty or None")
                return "figure", "vertical_lines is empty", 0.0
            
            # Analyze text characteristics
            text_lower = ocr_text.lower()
            context_lower = context.lower()
            
            # Enhanced type detection
            confidence = 0.3  # Base confidence
            image_type = "figure"
            analysis = ""
            
            # Table detection
            if (float(horizontal_lines.sum()) > 100 and float(vertical_lines.sum()) > 100) or \
               any(word in text_lower for word in ['table', 'row', 'column', '|']):
                image_type = "table"
                confidence = 0.8
                analysis = f"Table with structured data containing {len(ocr_text.split())} text elements."
                if ocr_text:
                    # Count potential rows/columns
                    rows = len([line for line in ocr_text.split('\n') if line.strip()])
                    analysis += f" Estimated {rows} rows of data."
            
            # Chart/Graph detection
            elif any(word in text_lower for word in ['%', 'chart', 'graph', 'axis', 'plot']) or \
                 len(re.findall(r'\d+\.?\d*%', text_lower)) > 2:
                image_type = "chart"
                confidence = 0.9
                percentages = re.findall(r'\d+\.?\d*%', text_lower)
                numbers = re.findall(r'\d+\.?\d*', text_lower)
                analysis = f"Chart/Graph with {len(numbers)} numeric values"
                if percentages:
                    analysis += f" including {len(percentages)} percentages"
                analysis += f". Key data: {', '.join(percentages[:3])}" if percentages else ""
            
            # Diagram/Flow detection
            elif any(word in text_lower for word in ['flow', 'process', 'step', 'arrow', 'diagram']) or \
                 any(word in context_lower for word in ['process', 'workflow', 'procedure']):
                image_type = "diagram"
                confidence = 0.7
                analysis = f"Process diagram or flowchart showing workflow steps."
                if ocr_text:
                    steps = len([word for word in text_lower.split() if word in ['step', '1', '2', '3', '4', '5']])
                    if steps > 0:
                        analysis += f" Contains approximately {steps} process steps."
            
            # Figure detection with context
            else:
                confidence = 0.5
                analysis = f"Figure or illustration"
                if ocr_text:
                    analysis += f" with descriptive text: '{ocr_text[:100]}...'" if len(ocr_text) > 100 else f" with text: '{ocr_text}'"
                else:
                    analysis += " (visual content without readable text)"
            
            # Boost confidence based on context relevance
            if any(word in context_lower for word in [image_type, 'figure', 'table', 'chart']):
                confidence = min(confidence + 0.2, 1.0)
            
            # Add dimensional and quality info
            analysis += f" [Size: {width}x{height}px, Quality: {'High' if min(float(width), float(height)) > 150 else 'Medium'}]"
            
            return image_type, analysis, confidence
            
        except Exception as e:
            logger.error(f"Enhanced image analysis failed: {e}")
            return "figure", f"Image analysis error: {str(e)}", 0.2

    def extract_text_from_pdf(self, file_path: str) -> PDFDocument:
        """Extract text with enhanced processing."""
        doc = PDFDocument(filename=os.path.basename(file_path))
        
        try:
            # Extract images and page images
            doc.images, doc.page_images = self.extract_images_from_pdf(file_path)
            
            # Build page-to-images mapping
            for idx, img in enumerate(doc.images):
                page = img.page_number
                if page not in doc.page_to_images:
                    doc.page_to_images[page] = []
                doc.page_to_images[page].append(idx)
            
            # Extract text
            full_text = ""
            char_to_page = []
            
            with open(file_path, "rb") as file:
                pdf = pypdf.PdfReader(file)
                
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    page_text = page.extract_text() or ""
                    page_text = self._clean_pdf_text(page_text)
                    
                    # Add high-quality OCR text from images
                    if page_num in doc.page_to_images:
                        image_texts = []
                        for img_idx in doc.page_to_images[page_num]:
                            img = doc.images[img_idx]
                            if img.ocr_text and img.confidence > 0.6:
                                image_texts.append(f"[{img.image_type.title()}]: {img.ocr_text}")
                        
                        if image_texts:
                            page_text += f"\n\n=== Visual Content on Page {page_num} ===\n" + "\n".join(image_texts)
                    
                    if page_text.strip():
                        doc.pages[page_num] = page_text
                        full_text += page_text + "\n\n"
                        char_to_page.extend([page_num] * len(page_text + "\n\n"))
            
            doc.content = full_text
            doc.char_to_page_map = char_to_page
            
            return doc
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise

    def _clean_pdf_text(self, text: str) -> str:
        """Enhanced text cleaning."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s*\n\s*', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fix hyphenated words
        text = re.sub(r'([a-z])-\s*\n\s*([a-z])', r'\1\2', text)
        text = re.sub(r'([a-z])- ?([a-z])', r'\1\2', text)
        
        # Fix common OCR errors
        text = re.sub(r'\b(\w)\s+(\w)\b', r'\1\2', text)  # Fix spaced letters
        text = re.sub(r'(\w)\s*\.\s*(\w)', r'\1.\2', text)  # Fix spaced periods
        
        return text.strip()

    def chunk_document_enhanced(self, doc: PDFDocument) -> PDFDocument:
        """Enhanced chunking with better context preservation."""
        if not doc.content:
            return doc
        
        try:
            # Create overlapping chunks with context
            raw_chunks = self.text_splitter.create_documents([doc.content])
            
            for i, chunk in enumerate(raw_chunks):
                chunk_text = chunk.page_content
                start_pos = doc.content.find(chunk_text)
                
                if start_pos == -1:
                    # Fallback for exact match failure
                    # Try to find approximate position
                    words = chunk_text.split()[:5]
                    search_text = " ".join(words)
                    start_pos = doc.content.find(search_text)
                    if start_pos == -1:
                        start_pos = i * (self.chunk_size - self.chunk_overlap)
                
                end_pos = start_pos + len(chunk_text) - 1
                
                # Get context before and after
                context_before = ""
                context_after = ""
                
                if start_pos > self.context_window:
                    context_start = start_pos - self.context_window
                    context_before = doc.content[context_start:start_pos].strip()
                
                if end_pos + self.context_window < len(doc.content):
                    context_end = end_pos + self.context_window
                    context_after = doc.content[end_pos:context_end].strip()
                
                # Find pages this chunk spans
                chunk_pages = set()
                for pos in range(max(0, start_pos), min(end_pos + 1, len(doc.char_to_page_map))):
                    if pos < len(doc.char_to_page_map):
                        chunk_pages.add(doc.char_to_page_map[pos])
                
                if not chunk_pages:
                    chunk_pages = {1}
                
                # Enhanced image matching
                relevant_images = self._find_relevant_images(
                    chunk_text, list(chunk_pages), doc.images, doc.page_to_images
                )
                
                # Extract section information
                section_info = self._extract_section_info_enhanced(chunk_text)
                
                enhanced_chunk = EnhancedChunk(
                    text=chunk_text,
                    page_numbers=sorted(list(chunk_pages)),
                    start_char_idx=start_pos,
                    end_char_idx=end_pos,
                    filename=doc.filename,
                    section_info=section_info,
                    image_refs=relevant_images,
                    context_before=context_before,
                    context_after=context_after
                )
                
                doc.chunks.append(enhanced_chunk)
            
            # Created enhanced chunks
            return doc
            
        except Exception as e:
            logger.error(f"Error in enhanced chunking: {e}")
            return doc

    def _find_relevant_images(self, chunk_text: str, chunk_pages: List[int], 
                              images: List[EnhancedImageData], 
                              page_to_images: Dict[int, List[int]]) -> List[int]:
        """Find images relevant to a chunk using semantic matching."""
        relevant_images = []
        
        try:
            # Get candidate images from chunk pages and adjacent pages
            candidate_images = []
            for page in chunk_pages:
                # Current page
                if page in page_to_images:
                    candidate_images.extend(page_to_images[page])
                # Adjacent pages
                for adj_page in [page - 1, page + 1]:
                    if adj_page in page_to_images:
                        candidate_images.extend(page_to_images[adj_page])
            
            # Remove duplicates
            candidate_images = list(set(candidate_images))
            
            # Score each candidate image
            image_scores = []
            for img_idx in candidate_images:
                if img_idx < len(images):
                    img = images[img_idx]
                    
                    # Calculate relevance score
                    relevance_score = self.image_matcher.calculate_image_relevance(chunk_text, img)
                    
                    # Additional scoring factors
                    page_proximity = 1.0 if img.page_number in chunk_pages else 0.5
                    confidence_factor = img.confidence
                    
                    final_score = relevance_score * page_proximity * confidence_factor
                    
                    if final_score > 0.3:  # Threshold for relevance
                        image_scores.append((img_idx, final_score))
            
            # Sort by score and take top relevant images
            image_scores.sort(key=lambda x: x[1], reverse=True)
            relevant_images = [img_idx for img_idx, score in image_scores[:2]]  # Max 2 images per chunk
            
        except Exception as e:
            logger.error(f"Error finding relevant images: {e}")
        
        return relevant_images

    def _extract_section_info_enhanced(self, text: str) -> Dict[str, str]:
        """Enhanced section extraction."""
        section_info = {}
        
        lines = text.split('\n')
        for pattern in self.section_patterns:
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    if len(groups) >= 2:
                        section_num = groups[0].strip()
                        section_title = groups[1].strip()
                        section_info[section_num] = section_title
                    elif len(groups) == 1:
                        section_title = groups[0].strip()
                        if len(section_title) > 3:  # Avoid single words
                            section_info[f"section_{len(section_info)}"] = section_title
        
        return section_info

    def process_pdf(self, file_path: str) -> PDFDocument:
        """Process PDF with all enhancements."""
        doc = self.extract_text_from_pdf(file_path)
        return self.chunk_document_enhanced(doc)

class VertexAIEmbeddings(Embeddings):
    """Embeddings class that can use either VertexAI or another compatible provider."""
    
    def __init__(self, gen_ai_provider: Union[VertexGenAI, GroqGenAI], model_name: str = "text-embedding-004"):
        self.gen_ai_provider = gen_ai_provider
        self.model_name = model_name
        self._embedding_dimension = None
        super().__init__()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with preprocessing."""
        if not texts:
            return []
        
        try:
            # Preprocess texts for better embeddings
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # Use the generic provider to get embeddings
            embeddings_response = self.gen_ai_provider.get_embeddings(processed_texts, self.model_name)
            
            embeddings = []
            
            # Handle different response formats (VertexAI vs. local SentenceTransformer)
            for embedding_data in embeddings_response:
                if hasattr(embedding_data, 'values'): # VertexAI format
                    vector = np.array(embedding_data.values, dtype=np.float32)
                elif isinstance(embedding_data, list): # List format from GroqGenAI
                    vector = np.array(embedding_data, dtype=np.float32)
                else:
                    embeddings.append(self._get_zero_vector())
                    continue

                # L2 normalize for cosine similarity
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                embeddings.append(vector.tolist())

            if embeddings and self._embedding_dimension is None:
                self._embedding_dimension = len(embeddings[0])
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            return [self._get_zero_vector() for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query with preprocessing."""
        if not text:
            return self._get_zero_vector()
        
        try:
            processed_text = self._preprocess_text(text)
            embeddings_response = self.gen_ai_provider.get_embeddings([processed_text], self.model_name)
            
            if embeddings_response and len(embeddings_response) > 0:
                embedding_data = embeddings_response[0]
                if hasattr(embedding_data, 'values'): # VertexAI format
                    vector = np.array(embedding_data.values, dtype=np.float32)
                elif isinstance(embedding_data, list): # List format from GroqGenAI
                    vector = np.array(embedding_data, dtype=np.float32)
                else:
                    return self._get_zero_vector()
                
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                
                if self._embedding_dimension is None:
                    self._embedding_dimension = len(vector)
                return vector.tolist()
            
            return self._get_zero_vector()
            
        except Exception as e:
            logger.error(f"Error getting query embedding: {str(e)}")
            return self._get_zero_vector()
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better embeddings."""
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Truncate if too long (most embedding models have limits)
        if len(text) > 8000:  # Conservative limit
            text = text[:8000] + "..."
        
        return text
    
    def _get_zero_vector(self) -> List[float]:
        """Get normalized zero vector."""
        dim = self._embedding_dimension if self._embedding_dimension else 768
        return [0.0] * dim

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)

class CustomReranker:
    """Custom reranker using semantic similarity and keyword matching."""
    
    def __init__(self):
        try:
            # Use a lightweight sentence transformer for semantic similarity
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence transformer for custom reranking")
        except Exception as e:
            logger.warning(f"Could not load sentence transformer for reranking: {e}")
            self.sentence_model = None
    
    def rerank(self, query: str, documents: List[Document], top_k: int = None) -> List[Tuple[Document, float]]:
        """Rerank documents using custom scoring."""
        if not documents:
            return []
        
        if top_k is None:
            top_k = len(documents)
        
        try:
            # Get query embedding if model is available
            query_embedding = None
            if self.sentence_model:
                query_embedding = self.sentence_model.encode([query])[0]
            
            # Score each document
            scored_docs = []
            for doc in documents:
                score = self._calculate_custom_score(query, doc, query_embedding)
                scored_docs.append((doc, score))
            
            # Sort by score and return top_k
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return scored_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Error in custom reranking: {e}")
            # Fallback: return documents with original scores
            return [(doc, doc.metadata.get('combined_score', 0.5)) for doc in documents[:top_k]]
    
    def _calculate_custom_score(self, query: str, doc: Document, query_embedding: np.ndarray = None) -> float:
        """Calculate custom relevance score for a document."""
        score = 0.0
        
        # Get document text
        doc_text = doc.metadata.get('original_text', doc.page_content)
        
        # 1. Semantic similarity (40% weight)
        if query_embedding is not None and self.sentence_model:
            try:
                doc_embedding = self.sentence_model.encode([doc_text])[0]
                semantic_sim = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                score += 0.4 * semantic_sim
            except:
                pass
        
        # 2. Keyword matching (30% weight)
        keyword_score = self._keyword_matching_score(query, doc_text)
        score += 0.3 * keyword_score
        
        # 3. Length and quality factors (20% weight)
        quality_score = self._quality_score(doc_text, doc.metadata)
        score += 0.2 * quality_score
        
        # 4. Original retrieval score (10% weight)
        original_score = doc.metadata.get('combined_score', 0.5)
        score += 0.1 * original_score
        
        return min(score, 1.0)
    
    def _keyword_matching_score(self, query: str, doc_text: str) -> float:
        """Calculate keyword matching score."""
        query_words = set(query.lower().split())
        doc_words = set(doc_text.lower().split())
        
        if not query_words or not doc_words:
            return 0.0
        
        # Exact matches
        exact_matches = len(query_words & doc_words)
        
        # Partial matches (substring)
        partial_matches = 0
        for query_word in query_words:
            if len(query_word) > 2:  # Only consider words longer than 2 chars
                for doc_word in doc_words:
                    if query_word in doc_word or doc_word in query_word:
                        partial_matches += 0.5
        
        total_matches = exact_matches + partial_matches
        return min(total_matches / len(query_words), 1.0)
    
    def _quality_score(self, doc_text: str, metadata: dict) -> float:
        """Calculate quality score based on document characteristics."""
        score = 0.5  # Base score
        
        # Length factor (prefer medium-length documents)
        text_length = len(doc_text)
        if 100 <= text_length <= 1000:
            score += 0.2
        elif text_length > 1000:
            score += 0.1
        
        # Semantic density factor
        semantic_density = metadata.get('semantic_density', 0.0)
        if semantic_density > 10:
            score += 0.2
        
        # Section information factor
        if metadata.get('section_info'):
            score += 0.1
        
        return min(score, 1.0)

class HybridRetriever:
    """Hybrid retriever combining dense (HNSW) and sparse (BM25) retrieval with reranking."""
    
    def __init__(self, documents: List[Document] = None, gen_ai_provider: Union[VertexGenAI, GroqGenAI] = None):
        logger.info("Initializing HybridRetriever with custom reranking...")
        
        self.gen_ai_provider = gen_ai_provider
        self.embeddings = VertexAIEmbeddings(gen_ai_provider) if gen_ai_provider else None
        
        # Storage
        self.documents = []
        self.document_embeddings = []
        
        # Dense retrieval (HNSW)
        self.hnsw_index = None
        self.dimension = None
        
        # Sparse retrieval (BM25)
        self.bm25 = None
        self.tokenized_docs = []
        
        # Custom Reranking
        self.reranker = CustomReranker()
        
        if documents and len(documents) > 0:
            self.add_documents(documents)
    
    def _init_reranker(self):
        """Initialize custom reranker."""
        try:
            self.reranker = CustomReranker()
            logger.info("Initialized custom reranker")
        except Exception as e:
            logger.warning(f"Could not initialize reranker: {e}")
            self.reranker = None
    
    def add_documents(self, documents: List[Document]):
        """Add documents to both dense and sparse indices."""
        try:
            # Add documents to hybrid index
            
            # Extract texts
            texts = [doc.page_content for doc in documents]
            
            # Dense embeddings
            embeddings = self.embeddings.embed_documents(texts)
            
            if embeddings:
                # Initialize HNSW if needed
                if self.hnsw_index is None:
                    self.dimension = len(embeddings[0])
                    self.hnsw_index = hnswlib.Index(space='cosine', dim=self.dimension)
                    self.hnsw_index.init_index(max_elements=50000, ef_construction=400, M=32)
                
                # Add to HNSW
                start_idx = len(self.documents)
                ids = list(range(start_idx, start_idx + len(documents)))
                embeddings_array = np.array(embeddings, dtype=np.float32)
                self.hnsw_index.add_items(embeddings_array, ids)
                self.hnsw_index.set_ef(100)  # Higher ef for better recall
            
            # Sparse indexing (BM25)
            new_tokenized = [self._tokenize_text(text) for text in texts]
            self.tokenized_docs.extend(new_tokenized)
            
            # Rebuild BM25 index
            if self.tokenized_docs:
                self.bm25 = BM25Okapi(self.tokenized_docs)
            
            # Store documents and embeddings
            self.documents.extend(documents)
            self.document_embeddings.extend(embeddings)
            
            # Successfully added documents to hybrid index
            
        except Exception as e:
            logger.error(f"Error adding documents to hybrid index: {str(e)}")
            raise
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        # Simple tokenization - could be enhanced
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        # Remove very short tokens
        tokens = [token for token in tokens if len(token) > 2]
        return tokens
    
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve documents using hybrid approach with reranking."""
        try:
            if not self.documents:
                return []
            
            # Hybrid retrieval for query
            
            # Step 1: Dense retrieval (HNSW)
            dense_results = self._dense_retrieval(query, k * 3)  # Get more candidates
            
            # Step 2: Sparse retrieval (BM25)
            sparse_results = self._sparse_retrieval(query, k * 3)
            
            # Step 3: Combine and deduplicate
            combined_results = self._combine_results(dense_results, sparse_results)
            
            # Step 4: Rerank if available
            if self.reranker and len(combined_results) > k:
                reranked_results = self._rerank_results(query, combined_results, k * 2)
            else:
                reranked_results = combined_results[:k * 2]
            
            # Step 5: Final selection and scoring
            final_results = self._final_selection(reranked_results, k)
            
            # Retrieved documents after hybrid processing
            return final_results
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            return []
    
    def _dense_retrieval(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Dense retrieval using HNSW."""
        results = []
        
        try:
            if not self.hnsw_index:
                return results
            
            query_embedding = self.embeddings.embed_query(query)
            if not query_embedding:
                return results
            
            query_array = np.array([query_embedding], dtype=np.float32)
            labels, distances = self.hnsw_index.knn_query(query_array, k=min(k, len(self.documents)))
            
            for label, distance in zip(labels[0], distances[0]):
                if label < len(self.documents):
                    doc = self.documents[label].copy()
                    similarity = 1 - distance  # Convert distance to similarity
                    doc.metadata['dense_score'] = similarity
                    doc.metadata['retrieval_method'] = 'dense'
                    results.append((doc, similarity))
                    
        except Exception as e:
            logger.error(f"Error in dense retrieval: {e}")
        
        return results
    
    def _sparse_retrieval(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Sparse retrieval using BM25."""
        results = []
        
        try:
            if not self.bm25:
                return results
            
            query_tokens = self._tokenize_text(query)
            if not query_tokens:
                return results
            
            scores = self.bm25.get_scores(query_tokens)
            
            # Get top k results
            top_indices = np.argsort(scores)[::-1][:k]
            
            for idx in top_indices:
                if idx < len(self.documents) and scores[idx] > 0:
                    doc = self.documents[idx].copy()
                    doc.metadata['sparse_score'] = float(scores[idx])
                    doc.metadata['retrieval_method'] = 'sparse'
                    results.append((doc, float(scores[idx])))
                    
        except Exception as e:
            logger.error(f"Error in sparse retrieval: {e}")
        
        return results
    
    def _combine_results(self, dense_results: List[Tuple[Document, float]], 
                         sparse_results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Combine and deduplicate dense and sparse results."""
        seen_hashes = set()
        combined = []
        
        # Normalize scores to [0, 1] range
        def normalize_scores(results):
            if not results:
                return results
            scores = [score for _, score in results]
            max_s, min_s = max(scores), min(scores)
            if max_s > min_s:
                return [(doc, (score - min_s) / (max_s - min_s)) 
                        for doc, score in results]
            return results
        
        dense_norm = normalize_scores(dense_results)
        sparse_norm = normalize_scores(sparse_results)
        
        # Combine with weighted scoring
        all_results = []
        
        # Add dense results with weight
        for doc, score in dense_norm:
            doc_hash = doc.metadata.get('chunk_hash', hash(doc.page_content))
            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)
                doc.metadata['combined_score'] = 0.7 * score  # Dense weight: 0.7
                all_results.append((doc, doc.metadata['combined_score']))
        
        # Add sparse results with weight
        for doc, score in sparse_norm:
            doc_hash = doc.metadata.get('chunk_hash', hash(doc.page_content))
            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)
                doc.metadata['combined_score'] = 0.3 * score  # Sparse weight: 0.3
                all_results.append((doc, doc.metadata['combined_score']))
            else:
                # If document already exists, boost its score
                for i, (existing_doc, existing_score) in enumerate(all_results):
                    existing_hash = existing_doc.metadata.get('chunk_hash', hash(existing_doc.page_content))
                    if existing_hash == doc_hash:
                        boosted_score = existing_score + 0.3 * score
                        existing_doc.metadata['combined_score'] = boosted_score
                        all_results[i] = (existing_doc, boosted_score)
                        break
        
        # Sort by combined score
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results
    
    def _rerank_results(self, query: str, results: List[Tuple[Document, float]], 
                        top_k: int) -> List[Tuple[Document, float]]:
        """Rerank results using custom reranker."""
        if not self.reranker or len(results) <= 1:
            return results
        
        try:
            # Extract documents from results
            documents = [doc for doc, _ in results]
            
            # Use custom reranker
            reranked_docs = self.reranker.rerank(query, documents, top_k)
            
            # Combine with original scores
            reranked_results = []
            for doc, rerank_score in reranked_docs:
                # Find original score
                original_score = next((score for d, score in results if d.metadata.get('chunk_hash') == doc.metadata.get('chunk_hash')), 0.5)
                
                # Weighted combination: 70% rerank, 30% original
                final_score = 0.7 * rerank_score + 0.3 * original_score
                doc.metadata['rerank_score'] = rerank_score
                doc.metadata['final_score'] = final_score
                reranked_results.append((doc, final_score))
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error in custom reranking: {e}")
            return results[:top_k]
    
    def _final_selection(self, results: List[Tuple[Document, float]], k: int) -> List[Document]:
        """Final selection with diversity and quality filtering."""
        if not results:
            return []
        
        selected_docs = []
        seen_pages = set()
        
        for doc, score in results:
            if len(selected_docs) >= k:
                break
            
            # Quality filter: minimum score threshold
            if score < 0.1:
                continue
            
            # Diversity filter: avoid too many chunks from same page
            doc_pages = set(doc.metadata.get('page_numbers', []))
            page_overlap = len(doc_pages & seen_pages)
            
            # Allow if no overlap or score is very high
            if page_overlap == 0 or score > 0.8:
                selected_docs.append(doc)
                seen_pages.update(doc_pages)
        
        return selected_docs

# Add this helper function at the top of the UltraAccurateRAGSystem class

class UltraAccurateRAGSystem:
    """Ultra-accurate RAG system with all enhancements."""
    
    def __init__(self, gen_ai_provider: Union[VertexGenAI, GroqGenAI] = None):
        logger.info("Initializing UltraAccurateRAGSystem...")
        
        self.gen_ai_provider = gen_ai_provider
        self.processor = AdvancedPDFProcessor()
        self.documents = {}
        self.hybrid_retriever = None
        self.conversation_history = []
        self.reference_counter = 1
        
        if gen_ai_provider:
            self.hybrid_retriever = HybridRetriever(gen_ai_provider=gen_ai_provider)
    
    def _convert_to_serializable(self, obj):
        """Convert NumPy types to Python native types for JSON serialization."""
        import numpy as np
        
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj
            
    def upload_pdf(self, file_path: str) -> str:
        """Process and index PDF with all enhancements."""
        try:
            # Processing PDF with enhanced accuracy
            
            doc = self.processor.process_pdf(file_path)
            self.documents[doc.filename] = doc
            
            if self.hybrid_retriever:
                self.hybrid_retriever.add_documents(doc.langchain_documents)
            
            # Detailed stats
            high_conf_images = sum(1 for img in doc.images if float(img.confidence) > 0.6)
            semantic_chunks = sum(1 for chunk in doc.chunks if float(chunk.semantic_density) > 10)
            
            result = f"""✅ **Enhanced Processing Complete: {doc.filename}**

**Content Analysis:**
- 📄 Pages: {len(doc.pages)}
- 🧩 Chunks: {len(doc.chunks)} (with context preservation)
- 🖼️ Images: {len(doc.images)} ({high_conf_images} high-confidence)
- 🎯 Semantic chunks: {semantic_chunks}

**Quality Metrics:**
- Image analysis confidence: {np.mean([float(img.confidence) for img in doc.images if doc.images]):.2f}
- Chunk semantic density: {np.mean([float(chunk.semantic_density) for chunk in doc.chunks if doc.chunks]):.1f}
"""
            
            # Processing complete
            return result
            
        except Exception as e:
            error_msg = f"❌ **Error processing PDF:** {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def ask(self, query: str) -> Dict[str, Any]:
        """Ultra-accurate query processing with enhanced retrieval."""
        if not self.hybrid_retriever or not self.documents:
            return {
                "answer": "Please upload documents first.",
                "references": [],
                "images": []
            }
        
        # Reset reference counter for each new response
        self.reference_counter = 1
        try:
            # Processing query with enhanced accuracy
            
            # Enhanced query preprocessing
            processed_query = self._enhance_query(query)
            
            # Hybrid retrieval with reranking
            relevant_docs = self.hybrid_retriever.get_relevant_documents(processed_query, k=6)
            
            if not relevant_docs:
                return {
                    "answer": "No relevant information found. Try rephrasing your question or check if the information exists in your documents.",
                    "references": [],
                    "images": []
                }
            
            # Enhanced context preparation
            context_parts = []
            references = []
            all_images = []
            
            for i, doc in enumerate(relevant_docs):
                metadata = doc.metadata
                doc_name = metadata.get('source', 'Unknown')
                pages = metadata.get('page_numbers', [])
                
                # Enhanced scoring - Convert to native Python types
                final_score = float(metadata.get('final_score', metadata.get('combined_score', 0.0)))
                dense_score = float(metadata.get('dense_score', 0.0))
                sparse_score = float(metadata.get('sparse_score', 0.0))
                rerank_score = float(metadata.get('rerank_score', 0.0))
                
                # Create reference with enhanced info
                ref_id = self.reference_counter
                reference = {
                    "id": ref_id,
                    "document": doc_name,
                    "pages": pages,
                    "final_score": final_score,
                    "dense_score": dense_score,
                    "sparse_score": sparse_score,
                    "rerank_score": rerank_score,
                    "content": metadata.get('original_text', doc.page_content)[:200] + "...",
                    "full_content": metadata.get('original_text', doc.page_content),
                    "type": "text",
                    "semantic_density": float(metadata.get('semantic_density', 0.0))
                }
                references.append(reference)
                self.reference_counter += 1
                
                # Enhanced context with metadata
                section_info = metadata.get('section_info', {})
                section_text = ""
                if section_info:
                    sections = [f"{k}: {v}" for k, v in section_info.items()]
                    section_text = f" [Sections: {', '.join(sections)}]"
                
                context_text = f"[{ref_id}] {doc.page_content}{section_text}"
                context_parts.append(context_text)
                
                # Enhanced image retrieval - show images from chunks
                image_refs = metadata.get('image_refs', [])
                if image_refs and doc_name in self.documents:
                    pdf_doc = self.documents[doc_name]
                    for img_idx in image_refs:
                        if img_idx < len(pdf_doc.images):
                            img_data = pdf_doc.images[img_idx]
                            
                            # Calculate image relevance to query
                            img_relevance = self.processor.image_matcher.calculate_image_relevance(
                                processed_query, img_data
                            )
                            
                            # Lower threshold to show more images
                            if img_relevance > 0.1:  # Lowered threshold
                                all_images.append({
                                    "document": doc_name,
                                    "page": img_data.page_number,
                                    "type": img_data.image_type,
                                    "analysis": img_data.analysis,
                                    "confidence": float(img_data.confidence),
                                    "base64": img_data.base64_string,
                                    "reference_id": self.reference_counter,
                                    "relevance_score": float(img_relevance),
                                    "ocr_text": img_data.ocr_text[:150],
                                    "keywords": img_data.semantic_keywords,
                                    "full_ocr_text": img_data.ocr_text,
                                    "chunk_text": doc.page_content[:200]  # Add chunk text for context
                                })
                                
                                # Add image reference
                                img_ref = {
                                    "id": self.reference_counter,
                                    "document": doc_name,
                                    "pages": [img_data.page_number],
                                    "final_score": float(img_relevance),
                                    "content": f"{img_data.image_type}: {img_data.analysis[:150]}...",
                                    "full_content": f"{img_data.image_type}: {img_data.analysis}",
                                    "type": "image",
                                    "image_confidence": float(img_data.confidence),
                                    "base64": img_data.base64_string,
                                    "chunk_text": doc.page_content[:200]  # Add chunk text
                                }
                                references.append(img_ref)
                                self.reference_counter += 1
            
            # Select top 3 most relevant images
            top_images = sorted(all_images, key=lambda x: x['relevance_score'] * x['confidence'], reverse=True)[:3]
            
            # Enhanced context preparation
            context = self._prepare_enhanced_context(context_parts, top_images)
            
            # Create reference links
            ref_links = {ref["id"]: f"Ref {ref['id']}" for ref in references}
            
            # Enhanced prompt with better instructions
            prompt = self._create_enhanced_prompt(processed_query, context, ref_links)
            
            # Generate response
            response = self.gen_ai_provider.generate_content(prompt)
            
            if not response:
                response = "I apologize, but I couldn't generate a comprehensive response. Please try rephrasing your question with more specific terms."
            
            # Process response to make citations clickable
            response = self._make_citations_clickable(response, references)
            
            self.conversation_history.append((query, response))
            
            # --- Only keep references and images that are actually cited in the answer ---
            cited_ids = self._extract_cited_ids(response)
            filtered_references = [ref for ref in references if ref["id"] in cited_ids]
            filtered_images = [img for img in top_images if img.get("reference_id") in cited_ids]
            # ---------------------------------------------------------------------------
            
            # Reference processing complete
            
            # Convert all data to JSON serializable format
            return {
                "answer": response,
                "references": self._convert_to_serializable(filtered_references),
                "images": self._convert_to_serializable(filtered_images)
            }
            
        except Exception as e:
            logger.error(f"Error in ultra-accurate processing: {str(e)}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "references": [],
                "images": []
            }
    
    def _enhance_query(self, query: str) -> str:
        """Enhance query for better retrieval."""
        # Add domain-specific terms based on query content
        enhanced_terms = []
        
        query_lower = query.lower()
        
        # Add related terms for better matching
        if any(word in query_lower for word in ['chart', 'graph', 'data']):
            enhanced_terms.extend(['visualization', 'statistics', 'analysis'])
        
        if any(word in query_lower for word in ['table', 'list']):
            enhanced_terms.extend(['data', 'comparison', 'structure'])
        
        if any(word in query_lower for word in ['process', 'procedure']):
            enhanced_terms.extend(['steps', 'methodology', 'workflow'])
        
        # Combine original query with enhancements
        if enhanced_terms:
            enhanced_query = f"{query} {' '.join(enhanced_terms)}"
        else:
            enhanced_query = query
        
        return enhanced_query
    
    def _prepare_enhanced_context(self, context_parts: List[str], images: List[Dict]) -> str:
        """Prepare enhanced context with image information."""
        context = "\n\n".join(context_parts)
        
        if images:
            image_context_header = "\n\n=== RELEVANT VISUAL CONTENT ===\n"
            image_details = []
            for img in images:
                img_detail = f"""
Image Reference [{img['reference_id']}] - {img['type'].title()} from {img['document']}, Page {img['page']}:
- Analysis: {img['analysis']}
- OCR Content: {img['ocr_text']}
- Keywords: {', '.join(img['keywords'])}
- Relevance: {img['relevance_score']:.2f}
"""
                image_details.append(img_detail)
            
            context += image_context_header + "".join(image_details)
        
        return context
    
    def _create_enhanced_prompt(self, query: str, context: str, ref_links: Dict) -> str:
        """Create enhanced prompt for better responses with sentence-level citations."""
        
        prompt = f"""You are an expert research assistant providing comprehensive, accurate analysis. Your task is to answer the user's question using ONLY the provided context with maximum accuracy and detail.

CRITICAL CITATION REQUIREMENTS:
1. EVERY SINGLE SENTENCE that contains information from the provided context MUST end with a citation marker in the format [ID] where ID is the reference number.
2. If a sentence uses information from multiple sources, include all relevant citations: [1][2].
3. Place the citation immediately after the period of each sentence, not in the middle.
4. Your own transitional phrases or introductory statements don't need citations, but ANY factual claim must have one.
5. Example format: "The data shows a 15% increase in efficiency [3]. This trend is consistent with the analysis presented in the chart [4]."

ACCURACY REQUIREMENTS:
1. Use ONLY information explicitly present in the provided context.
2. For numerical data, charts, tables: be extremely precise and detailed.
3. For visual content: provide thorough descriptions and interpretations.
4. If information is insufficient, clearly state what cannot be determined.
5. Never make assumptions or add external knowledge.

RESPONSE STRUCTURE:
1. Direct answer to the question with sentence-level citations.
2. Supporting evidence with detailed explanations and citations for each sentence.
3. Visual content analysis (if applicable) with citations.
4. Summary of key findings with citations.

CONTEXT WITH REFERENCES:
{context}

AVAILABLE REFERENCE IDs: {list(ref_links.keys())}

USER QUESTION: {query}

INSTRUCTIONS FOR VISUAL CONTENT:
- For charts/graphs: Describe data points, trends, axes, values precisely [ID].
- For tables: Explain structure, key values, relationships systematically [ID].
- For diagrams: Detail components, connections, flow sequences [ID].
- Always cite the specific image reference number for each visual element discussed.

Remember: EVERY sentence with factual information MUST have a citation [ID] at the end.

Provide a comprehensive, well-researched response with precise sentence-level citations:"""

        return prompt
    
    def _make_citations_clickable(self, response: str, references: List[Dict]) -> str:
        """Convert citation markers to clickable buttons that trigger Python events."""
        citation_pattern = r'\[(\d+)\]'
        def replace_citation(match):
            citation_num = int(match.group(1))
            if 1 <= citation_num <= len(references):
                ref_id = references[citation_num - 1].get('id')
                return (
                    f'<span class="citation-btn" onclick="document.getElementById(\'citation_trigger\').value=\'{ref_id}\';'
                    f'document.getElementById(\'citation_trigger\').dispatchEvent(new Event(\'input\')));" '
                    f'style="background: #007bff; color: white; border: none; border-radius: 3px; padding: 2px 6px; margin: 0 2px; cursor: pointer; font-size: 0.9em; display: inline-block;">[{citation_num}]</span>'
                )
            else:
                return match.group(0)
        response_with_links = re.sub(citation_pattern, replace_citation, response)
        return response_with_links
    
    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
        self.reference_counter = 1
        return "Conversation reset successfully."

    def _extract_cited_ids(self, answer: str) -> set:
        """Extract all cited reference IDs from the answer text."""
        return set(int(match) for match in re.findall(r'\[(\d+)\]', answer))

# Update the interface to use the new system
class UltraAccurateInterface:
    """Interface for ultra-accurate RAG system."""

    def __init__(self):
        self.rag_system = None
        self.gen_ai_provider = None
        self.current_references = []
        self.current_images = []
        logger.info("Initialized UltraAccurateInterface")
        self.setup_interface()

    def setup_interface(self):
        """Set up the ultra-accurate interface."""
        
        custom_css = """
        /* Chat app styling */
        .main-container {
            max-width: 1200px !important;
            margin: 0 auto !important;
            padding: 0 !important;
        }
        
        .chat-container {
            display: flex;
            flex-direction: column;
            height: 100vh;
            max-width: 100%;
        }
        
        .logo-container {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin-bottom: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .logo-text {
            font-size: 24px;
            font-weight: bold;
            color: white;
            text-align: center;
        }
        
        .accuracy-badge {
            background: linear-gradient(45deg, #28a745, #20c997);
            color: white;
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 11px;
            margin-left: 8px;
        }
        
        .chatbot-container {
            flex: 1;
            min-height: 400px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        
        .input-container {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
        }
        
        .button-container {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }
        
        .reference-card {
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 12px;
            margin: 8px 0;
            background: linear-gradient(135deg, #f8f9ff 0%, #f1f3ff 100%);
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        
        .image-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
            background: linear-gradient(135deg, #fff8f0 0%, #fff4e6 100%);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .pdf-page-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
            background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .score-badge {
            display: inline-block;
            background: #007bff;
            color: white;
            padding: 2px 6px;
            border-radius: 10px;
            font-size: 10px;
            margin-left: 4px;
        }
        
        .citation-link {
            color: #007bff;
            cursor: pointer;
            text-decoration: underline;
            font-weight: 500;
        }
        
        .citation-btn:hover {
            background: #0056b3 !important;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main-container {
                padding: 10px !important;
            }
            
            .logo-text {
                font-size: 20px;
            }
            
            .button-container {
                flex-direction: column;
            }
        }
        """

        # No JavaScript needed - using Python-based modal handling

        with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Ultra-Accurate Research Assistant") as self.interface:
            
            # Main container with chat app layout
            with gr.Column(elem_classes=["main-container"]):
                # Enhanced header
                gr.HTML("""
                <div class="logo-container">
                    <div style="margin-right: 15px; font-size: 32px;">🎯</div>
                    <div class="logo-text">
                        Ultra-Accurate Research Assistant
                        <div class="accuracy-badge">Enhanced with Sentence-Level Citations & Source Inspector</div>
                    </div>
                </div>
                """)
                
                # AI Provider selection
                with gr.Row():
                    provider_choice = gr.Radio(
                        ["VertexAI", "Groq"], 
                        label="🤖 Choose AI Provider", 
                        value="VertexAI",
                        info="Select the backend. Groq requires a GROQ_API_KEY env var."
                    )

                # Chat container
                with gr.Column(elem_classes=["chat-container"]):
                    # Enhanced chatbot
                    chatbot = gr.Chatbot(
                        height=400,
                        show_label=False,
                        avatar_images=("👤", "🔬"),
                        bubble_full_width=False,
                        show_copy_button=True,
                        elem_classes=["chatbot-container"]
                    )
                    
                    # Hidden state to store references and page images
                    references_state = gr.State([])
                    page_images_state = gr.State({})
                    
                    # Input container
                    with gr.Column(elem_classes=["input-container"]):
                        msg_input = gr.Textbox(
                            placeholder="Ask detailed questions about your documents. Click on any [citation] in the response to inspect the source!",
                            show_label=False,
                            lines=2,
                            scale=5
                        )
                    
                    # Button container
                    with gr.Row(elem_classes=["button-container"]):
                        upload_btn = gr.Button("📤 Upload Documents", variant="primary", size="lg", scale=1)
                        ask_btn = gr.Button("🎯 Ask (Ultra-Accurate)", variant="secondary", size="lg", scale=1) 
                        reset_btn = gr.Button("🔄 Reset", variant="stop", size="lg", scale=1)
                    
                    upload_files = gr.File(
                        label="Select PDF Documents for Analysis",
                        file_types=[".pdf"],
                        file_count="multiple",
                        visible=False
                    )
                    
                    status_display = gr.Markdown("🤖 **Status:** Select an AI provider and upload documents to begin.")
                
                # References and visual content sections
                with gr.Tabs():
                    with gr.TabItem("📚 Text References"):
                        text_references = gr.HTML(value="<p>No references yet. Upload documents and ask questions to see detailed references with accuracy scores.</p>")
                    
                    with gr.TabItem("🖼️ Visual References"):
                        with gr.Row():
                            image1 = gr.Image(label="Most Relevant Image", visible=False, height=200)
                            image2 = gr.Image(label="Second Most Relevant", visible=False, height=200)
                            image3 = gr.Image(label="Third Most Relevant", visible=False, height=200)
                        image_info = gr.HTML(value="<p>No visual content found yet. Upload documents with charts, diagrams, or tables to see them here.</p>")
                    
                    with gr.TabItem("📄 PDF Pages"):
                        pdf_pages = gr.HTML(value="<p>No relevant PDF pages found yet.</p>")

                # Citation modal (using HTML overlay approach)
                citation_modal = gr.HTML(visible=False, value="")
                close_btn = gr.Button("Close Modal", variant="secondary", visible=False)

                initialized = gr.State(False)

            # --- Start of Event Handler Changes ---
            def handle_upload_click():
                return gr.update(visible=True)
            
            def handle_upload_files(files, provider, is_initialized):
                if not is_initialized:
                    try:
                        if provider == "VertexAI":
                            self.gen_ai_provider = VertexGenAI()
                            test_response = self.gen_ai_provider.generate_content("Test accuracy")
                            if not test_response:
                                raise ConnectionError("Failed to initialize VertexAI.")
                        elif provider == "Groq":
                            self.gen_ai_provider = GroqGenAI()
                            test_response = self.gen_ai_provider.generate_content("Test accuracy")
                            if not test_response:
                                raise ConnectionError("Failed to connect to Groq. Check API key.")
                        else:
                            return "❌ Invalid provider selected.", False, gr.update(visible=False)
                        
                        self.rag_system = UltraAccurateRAGSystem(gen_ai_provider=self.gen_ai_provider)
                        is_initialized = True
                    except Exception as e:
                        return f"❌ Initialization error with {provider}: {str(e)}", False, gr.update(visible=False)

                if not files:
                    return f"✅ System initialized with {provider}. Ready for document processing.", is_initialized, gr.update(visible=False)
                
                results = []
                with tqdm(total=len(files), desc="Processing Documents") as pbar:
                    for file in files:
                        try:
                            result = self.rag_system.upload_pdf(file.name)
                            results.append(result)
                        except Exception as e:
                            results.append(f"❌ Error processing {os.path.basename(file.name)}: {str(e)}")
                        pbar.update(1)
                
                status = "## 📄 Ultra-Accurate Processing Results\n" + "\n\n".join(results)
                return status, is_initialized, gr.update(visible=False)
            
            def handle_ask(message, history, is_initialized, current_refs, current_page_images):
                if not is_initialized or not self.rag_system:
                    error_msg = "❌ Please upload documents first to enable ultra-accurate processing"
                    history = history or []
                    return history + [[message, error_msg]], "", "", "", "", error_msg, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), [], {}
                
                if not message.strip():
                    history = history or []
                    return history, message, "", "", "", "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), current_refs, current_page_images
                
                try:
                    # Get ultra-accurate response
                    result = self.rag_system.ask(message)
                    answer = result["answer"]
                    self.current_references = result["references"]
                    self.current_images = result["images"]

                    # Handle the case where no relevant information is found
                    if answer.startswith("No relevant information found") or answer.startswith("Please upload documents first."):
                        history = history or []
                        updated_history = history + [[message, answer]]
                        return (
                            updated_history,
                            "",
                            "<p>No references found.</p>",
                            "<p>No images found.</p>",
                            "<p>No relevant PDF pages found.</p>",
                            "⚠️ No relevant information found.",
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            self.current_references,
                            {}
                        )

                    # Prepare page images for modal
                    page_images = {}
                    for doc_name, doc in self.rag_system.documents.items():
                        if doc.page_images:
                            page_images[doc_name] = doc.page_images
                    
                    # Ensure everything is JSON serializable
                    import numpy as np
                    import json
                    
                    def convert_to_serializable(obj):
                        if isinstance(obj, (np.float32, np.float64)):
                            return float(obj)
                        elif isinstance(obj, (np.int32, np.int64)):
                            return int(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif isinstance(obj, dict):
                            return {key: convert_to_serializable(value) for key, value in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_to_serializable(item) for item in obj]
                        else:
                            return obj
                    
                    # Convert references and page images to ensure JSON serialization
                    serializable_refs = convert_to_serializable(self.current_references)
                    
                    # Debug the references to see what's being passed
                    for i, ref in enumerate(serializable_refs):
                        # Make sure 'id' field exists and is an integer
                        if 'id' not in ref:
                            ref['id'] = i + 1
                        elif not isinstance(ref['id'], int):
                            try:
                                ref['id'] = int(ref['id'])
                            except:
                                ref['id'] = i + 1
                                
                        # Ensure other necessary fields exist
                        if 'pages' not in ref:
                            ref['pages'] = []
                        if 'document' not in ref:
                            ref['document'] = 'Unknown'
                        if 'final_score' not in ref:
                            ref['final_score'] = 0.0
                        if 'content' not in ref:
                            ref['content'] = ''
                        if 'full_content' not in ref:
                            ref['full_content'] = ref.get('content', '')
                        if 'type' not in ref:
                            ref['type'] = 'text'  # Default to text type
                    
                    # Debug logging
                                # Processed references and documents
                    
                    # Process page images for serialization
                    serializable_page_images = {}
                    for doc_name, doc_pages in page_images.items():
                        serializable_page_images[doc_name] = {}
                        for page_num, img_data in doc_pages.items():
                            # Store only strings for base64 images, convert page_num to string for JavaScript
                            if isinstance(img_data, str):
                                serializable_page_images[doc_name][str(page_num)] = img_data
                                logger.info(f"Serialized page image for {doc_name} page {page_num}")
                            else:
                                logger.warning(f"Non-string page image data for {doc_name} page {page_num}: {type(img_data)}")
                    
                    # Serialized page images
                    
                    # Update chat history without JavaScript injection
                    history = history or []
                    updated_history = history + [[message, answer]]
                    
                    # Format enhanced references
                    text_refs_html = self._format_enhanced_references()
                    
                    # Format enhanced image info
                    images_info_html = self._format_enhanced_images()
                    
                    # Prepare images for display
                    img1, img2, img3 = self._prepare_enhanced_images()
                    
                    # Prepare top PDF pages
                    top_pages, pdf_pages_html = self._prepare_top_pdf_pages()
                    
                    return (updated_history, "", text_refs_html, images_info_html, pdf_pages_html, "✅ Response generated with sentence-level citations.",
                            img1, img2, img3, self.current_references, serializable_page_images)
                    
                except Exception as e:
                    error_msg = f"❌ Ultra-accurate processing error: {str(e)}"
                    history = history or []
                    return (history + [[message, error_msg]], "", "", "", "", error_msg,
                            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), 
                            current_refs, {})
            
            def handle_citation_click(ref_id, current_refs, current_page_images):
                """Handle citation button clicks to show reference details."""
                try:
                    ref_id = int(ref_id)
                    
                    # Find the reference
                    ref = next((r for r in current_refs if r.get('id') == ref_id), None)
                    
                    if not ref:
                        error_html = f"""
                        <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); z-index: 1000; display: flex; align-items: center; justify-content: center;">
                            <div style="background: white; padding: 30px; border-radius: 10px; max-width: 600px; max-height: 80vh; overflow-y: auto;">
                                <h3>❌ Reference Not Found</h3>
                                <p>Reference [{ref_id}] was not found in the available references.</p>
                                <p><strong>Available references:</strong></p>
                                <ul>
                                    {''.join([f'<li>[{r.get("id", "?")}] {r.get("document", "Unknown")} ({r.get("type", "unknown")})</li>' for r in current_refs])}
                                </ul>
                                <button onclick="this.parentElement.parentElement.remove()" style="background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin-top: 15px;">Close</button>
                            </div>
                        </div>
                        """
                        return gr.update(visible=True, value=error_html), gr.update(visible=True)
                    
                    # Build modal content with overlay
                    modal_html = f"""
                    <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); z-index: 1000; display: flex; align-items: center; justify-content: center;">
                        <div style="background: white; padding: 30px; border-radius: 10px; max-width: 800px; max-height: 85vh; overflow-y: auto; position: relative;">
                            <button onclick="this.parentElement.parentElement.remove()" style="position: absolute; top: 10px; right: 15px; background: none; border: none; font-size: 24px; cursor: pointer; color: #666;">&times;</button>
                            
                            <div style="border-bottom: 2px solid #007bff; padding-bottom: 15px; margin-bottom: 20px;">
                                <h2>📚 Source Inspector - Reference [{ref.get('id')}]</h2>
                                <p><strong>Document:</strong> {ref.get('document', 'Unknown')} | <strong>Pages:</strong> {', '.join(map(str, ref.get('pages', []))) if ref.get('pages') else 'N/A'}</p>
                            </div>
                            
                            <div style="background-color: #e7f3ff; padding: 15px; border-radius: 5px; margin: 15px 0;">
                                <h3>📊 Quality & Relevance Scores</h3>
                                <p><strong>Final Score:</strong> {ref.get('final_score', 0):.3f}</p>
                                {f'<p><strong>Dense Retrieval Score:</strong> {ref.get("dense_score", 0):.3f}</p>' if ref.get('dense_score') else ''}
                                {f'<p><strong>Sparse Retrieval Score:</strong> {ref.get("sparse_score", 0):.3f}</p>' if ref.get('sparse_score') else ''}
                                {f'<p><strong>Reranking Score:</strong> {ref.get("rerank_score", 0):.3f}</p>' if ref.get('rerank_score') else ''}
                                {f'<p><strong>Semantic Density:</strong> {ref.get("semantic_density", 0):.1f}</p>' if ref.get('semantic_density') else ''}
                            </div>
                    """
                    
                    # Add source content
                    if ref.get('type') == 'text':
                        modal_html += f"""
                        <div style="margin: 15px 0;">
                            <h3>📄 Source Text</h3>
                            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; font-family: 'Courier New', monospace; white-space: pre-wrap; max-height: 300px; overflow-y: auto;">
                                {ref.get('full_content', ref.get('content', 'No content available'))}
                            </div>
                        </div>
                        """
                    elif ref.get('type') == 'image' and ref.get('base64'):
                        modal_html += f"""
                        <div style="margin: 15px 0;">
                            <h3>🖼️ Source Image</h3>
                            <img src="data:image/png;base64,{ref.get('base64')}" alt="Source image" style="max-width: 100%; border: 1px solid #ddd; border-radius: 5px;">
                            <p><strong>Analysis:</strong> {ref.get('full_content', ref.get('content', 'No analysis available'))}</p>
                        </div>
                        """
                    
                    # Add full page image if available
                    if ref.get('pages') and ref.get('pages'):
                        page_num = ref.get('pages')[0]
                        doc_name = ref.get('document', '')
                        
                        # Check if page image exists
                        page_image = None
                        if (doc_name in self.rag_system.documents and 
                            self.rag_system.documents[doc_name].page_images and 
                            page_num in self.rag_system.documents[doc_name].page_images):
                            page_image = self.rag_system.documents[doc_name].page_images[page_num]
                            logger.info(f"Found page image for {doc_name} page {page_num}")
                        
                        if page_image:
                            modal_html += f"""
                            <div style="margin: 15px 0;">
                                <h3>📄 Full PDF Page {page_num}</h3>
                                <img src="data:image/png;base64,{page_image}" alt="PDF page {page_num}" style="max-width: 100%; border: 1px solid #ddd; border-radius: 5px;">
                            </div>
                            """
                        else:
                            logger.warning(f"No page image found for {doc_name} page {page_num}")
                    
                    modal_html += """
                        </div>
                    </div>
                    """
                    
                    return gr.update(visible=True, value=modal_html), gr.update(visible=True)
                    
                except Exception as e:
                    logger.error(f"Error handling citation click: {e}")
                    error_html = f"""
                    <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); z-index: 1000; display: flex; align-items: center; justify-content: center;">
                        <div style="background: white; padding: 30px; border-radius: 10px; max-width: 600px;">
                            <h3>❌ Error</h3>
                            <p>Error displaying reference details: {str(e)}</p>
                            <button onclick="this.parentElement.parentElement.remove()" style="background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin-top: 15px;">Close</button>
                        </div>
                    </div>
                    """
                    return gr.update(visible=True, value=error_html), gr.update(visible=True)
            
            def handle_reset():
                if self.rag_system:
                    self.rag_system.reset_conversation()
                self.current_references = []
                self.current_images = []
                return ([], "🔄 Ultra-accurate conversation reset", 
                        "<p>No references yet.</p>", 
                        "<p>No images yet.</p>",
                        "<p>No relevant PDF pages found yet.</p>",
                        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False, value=""), gr.update(visible=False),
                        [], {})
            
            # Wire up events
            upload_btn.click(fn=handle_upload_click, outputs=upload_files)
            upload_files.upload(fn=handle_upload_files, inputs=[upload_files, provider_choice, initialized], outputs=[status_display, initialized, upload_files])
            ask_btn.click(
                fn=handle_ask, 
                inputs=[msg_input, chatbot, initialized, references_state, page_images_state], 
                outputs=[chatbot, msg_input, text_references, image_info, pdf_pages, status_display, image1, image2, image3, references_state, page_images_state]
            )
            msg_input.submit(
                fn=handle_ask, 
                inputs=[msg_input, chatbot, initialized, references_state, page_images_state], 
                outputs=[chatbot, msg_input, text_references, image_info, pdf_pages, status_display, image1, image2, image3, references_state, page_images_state]
            )
            reset_btn.click(
                fn=handle_reset, 
                outputs=[chatbot, status_display, text_references, image_info, pdf_pages, image1, image2, image3, citation_modal, close_btn, references_state, page_images_state]
            )
            
            # Citation modal events
            close_btn.click(
                fn=lambda: (gr.update(visible=False, value=""), gr.update(visible=False)),
                outputs=[citation_modal, close_btn]
            )
            
            # Add citation click handler (this will be triggered by HTML buttons)
            # We'll use a hidden textbox to capture citation clicks
            citation_trigger = gr.Textbox(visible=False, label="Citation Trigger", elem_id="citation_trigger")
            citation_trigger.change(
                fn=handle_citation_click,
                inputs=[citation_trigger, references_state, page_images_state],
                outputs=[citation_modal, close_btn]
            )
            # --- End of Event Handler Changes ---

    def _format_enhanced_references(self):
        """Format references with enhanced accuracy information."""
        if not self.current_references:
            return "<p>No references available.</p>"
        
        logger.info(f"Formatting {len(self.current_references)} references")
        
        html_parts = []
        # Show all references that are not explicitly marked as image type
        text_refs = [ref for ref in self.current_references if ref.get("type") != "image"]
        
        logger.info(f"Found {len(text_refs)} text references to display")
        
        for ref in text_refs:
            ref_id = ref.get("id", "?")
            doc = ref.get("document", "Unknown")
            pages = ref.get("pages", [])
            final_score = ref.get("final_score", 0.0)
            dense_score = ref.get("dense_score", 0.0)
            sparse_score = ref.get("sparse_score", 0.0) 
            rerank_score = ref.get("rerank_score", 0.0)
            semantic_density = ref.get("semantic_density", 0.0)
            content = ref.get("content", "")
            
            # Format page numbers
            if len(pages) > 1:
                page_str = f"pp. {min(pages)}-{max(pages)}"
            else:
                page_str = f"p. {pages[0]}" if pages else "p. ?"
            
            ref_html = f"""
            <div class="reference-card" id="ref{ref_id}">
                <strong>[{ref_id}]</strong> 📄 {doc}, {page_str}
                <span class="score-badge">Final: {final_score:.3f}</span>
                <span class="score-badge">Dense: {dense_score:.3f}</span>
                <span class="score-badge">Sparse: {sparse_score:.3f}</span>
                {f'<span class="score-badge">Rerank: {rerank_score:.3f}</span>' if rerank_score > 0 else ''}
                <br><small><strong>Semantic Density:</strong> {semantic_density:.1f} | <strong>Quality:</strong> {'High' if final_score > 0.7 else 'Medium' if final_score > 0.4 else 'Low'}</small>
                <br><em>"{content}"</em>
            </div>
            """
            html_parts.append(ref_html)
        
        if not html_parts:
            return "<p>No text references found.</p>"
        
        return "".join(html_parts)

    def _format_enhanced_images(self):
        """Format image information with enhanced details."""
        if not self.current_images:
            return "<p>No relevant visual content found for this query.</p>"
        
        html_parts = []
        for i, img in enumerate(self.current_images):
            doc = img["document"]
            page = img["page"]
            img_type = img["type"]
            analysis = img["analysis"]
            confidence = img["confidence"]
            relevance_score = img["relevance_score"]
            ref_id = img.get("reference_id", "")
            ocr_preview = img.get("ocr_text", "")[:100]
            keywords = ", ".join(img.get("keywords", []))
            
            # Calculate combined score
            combined_score = relevance_score * confidence
            
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            
            # Get chunk text for context
            chunk_text = img.get("chunk_text", "")[:150]
            
            img_html = f"""
            <div class="image-card">
                <h4>{medal} {img_type.title()} from {doc}, Page {page}</h4>
                <p>
                    <span class="score-badge">Relevance: {relevance_score:.3f}</span>
                    <span class="score-badge">Confidence: {confidence:.3f}</span>
                    <span class="score-badge">Combined: {combined_score:.3f}</span>
                </p>
                <p><strong>Analysis:</strong> {analysis}</p>
                {f'<p><strong>OCR Text:</strong> "{ocr_preview}..."</p>' if ocr_preview else ''}
                {f'<p><strong>Keywords:</strong> {keywords}</p>' if keywords else ''}
                {f'<p><strong>Chunk Context:</strong> "{chunk_text}..."</p>' if chunk_text else ''}
                {f'<p><span class="ref-badge">Ref: [{ref_id}]</span></p>' if ref_id else ''}
            </div>
            """
            html_parts.append(img_html)
        
        if not html_parts:
            return "<p>No visual content found yet. Upload documents with charts, diagrams, or tables to see them here.</p>"
            
        return "".join(html_parts)

    def _prepare_enhanced_images(self):
        """Prepare enhanced images for display."""
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
        
        # Enhanced labels with scores and chunk context
        if len(display_images) > 0:
            img_data = self.current_images[0]
            relevance_score = float(img_data['relevance_score'])
            confidence = float(img_data['confidence'])
            score = relevance_score * confidence
            chunk_text = img_data.get('chunk_text', '')[:50]
            label = f"🥇 {img_data['type'].title()} (Score: {score:.3f}) - {img_data['document']}, Page {img_data['page']}"
            if chunk_text:
                label += f" | Context: {chunk_text}..."
            
            img1 = gr.update(
                value=display_images[0], 
                visible=True,
                label=label
            )
        
        if len(display_images) > 1:
            img_data = self.current_images[1]
            relevance_score = float(img_data['relevance_score'])
            confidence = float(img_data['confidence'])
            score = relevance_score * confidence
            chunk_text = img_data.get('chunk_text', '')[:50]
            label = f"🥈 {img_data['type'].title()} (Score: {score:.3f}) - {img_data['document']}, Page {img_data['page']}"
            if chunk_text:
                label += f" | Context: {chunk_text}..."
            
            img2 = gr.update(
                value=display_images[1],
                visible=True,
                label=label
            )
        
        if len(display_images) > 2:
            img_data = self.current_images[2]
            relevance_score = float(img_data['relevance_score'])
            confidence = float(img_data['confidence'])
            score = relevance_score * confidence
            chunk_text = img_data.get('chunk_text', '')[:50]
            label = f"🥉 {img_data['type'].title()} (Score: {score:.3f}) - {img_data['document']}, Page {img_data['page']}"
            if chunk_text:
                label += f" | Context: {chunk_text}..."
            
            img3 = gr.update(
                value=display_images[2],
                visible=True,
                label=label
            )
        
        return img1, img2, img3
        
    def _prepare_top_pdf_pages(self):
        """Show ONLY PDF pages referenced by text or visual references used in the answer. If there are no text references, show no PDF page images."""
        if not self.rag_system or not self.current_references:
            return [], "<p>No relevant PDF pages found yet.</p>"
        try:
            # Only consider references that are actually present (i.e., cited)
            cited_ref_ids = set(ref.get('id') for ref in self.current_references)
            # Only show PDF pages if there is at least one text reference
            text_refs = [ref for ref in self.current_references if ref.get('type') == 'text']
            if not text_refs:
                return [], "<p>No relevant PDF pages found for this query.</p>"
            # Extract all page references from current references
            page_data = []
            seen = set()
            for ref in self.current_references:
                if ref.get('type') == 'text' and ref.get('id') in cited_ref_ids:
                    doc_name = ref.get('document', '')
                    pages = ref.get('pages', [])
                    score = float(ref.get('final_score', 0.0))
                    chunk_text = ref.get('full_content', '')[:200]  # Get chunk text
                    for page_num in pages:
                        page_key = str(page_num)
                        unique_key = f"{doc_name}:{page_key}"
                        if unique_key in seen:
                            continue
                        page_image_exists = (doc_name in self.rag_system.documents and 
                                           self.rag_system.documents[doc_name].page_images and 
                                           page_num in self.rag_system.documents[doc_name].page_images)
                        if page_image_exists:
                            base64_data = self.rag_system.documents[doc_name].page_images[page_num]
                        else:
                            base64_data = None
                        page_data.append({
                            'document': doc_name,
                            'page': page_num,
                            'score': score,
                            'base64': base64_data,
                            'ref_id': ref.get('id'),
                            'chunk_text': chunk_text,
                            'has_image': page_image_exists
                        })
                        seen.add(unique_key)
            for img in self.current_images:
                doc_name = img.get('document', '')
                page_num = img.get('page')
                page_key = str(page_num)
                unique_key = f"{doc_name}:{page_key}"
                if page_num and doc_name in self.rag_system.documents and unique_key not in seen:
                    if page_num in self.rag_system.documents[doc_name].page_images:
                        score = float(img.get('relevance_score', 0.0)) * float(img.get('confidence', 1.0))
                        ref_id = img.get('reference_id', f'img-{page_num}')
                        chunk_text = img.get('chunk_text', '')
                        if ref_id in cited_ref_ids:
                            page_data.append({
                                'document': doc_name,
                                'page': page_num,
                                'score': score,
                                'base64': self.rag_system.documents[doc_name].page_images[page_num],
                                'ref_id': ref_id,
                                'chunk_text': chunk_text,
                                'has_image': True
                            })
                            seen.add(unique_key)
            if not page_data:
                return [], "<p>No relevant PDF pages found for this query.</p>"
            page_data.sort(key=lambda x: x['score'], reverse=True)
            html = "<div class='pdf-pages-container'>"
            html += "<h3>PDF Pages Referenced in the Answer</h3>"
            for i, page in enumerate(page_data):
                medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "✨"
                doc_name = os.path.basename(page['document'])
                chunk_text = page.get('chunk_text', '')
                has_image = page.get('has_image', False)
                if has_image and page.get('base64'):
                    base64_img = page.get('base64', '')
                    if not base64_img.startswith('data:image'):
                        base64_img = f"data:image/png;base64,{base64_img}"
                    image_html = f'<img src="{base64_img}" style="max-width:100%; border:1px solid #ddd; border-radius:5px;" alt="PDF Page {page["page"]}" />'
                else:
                    image_html = f'<div style="background: #f8f9fa; border: 2px dashed #dee2e6; border-radius: 5px; padding: 40px; text-align: center; color: #6c757d;"><strong>Page {page["page"]}</strong><br>PDF page image not available</div>'
                html += f"""
                <div class='pdf-page-card'>
                    <h4>{medal} {doc_name} - Page {page['page']}</h4>
                    <p><span class='score-badge'>Relevance: {page['score']:.3f}</span> <span class='ref-badge'>Ref: [{page['ref_id']}]</span></p>
                    {f'<p><strong>Chunk Context:</strong> "{chunk_text}..."</p>' if chunk_text else ''}
                    {image_html}
                </div>
                """
            html += "</div>"
            return page_data, html
        except Exception as e:
            logger.error(f"Error preparing top PDF pages: {str(e)}")
            return [], f"<p>Error displaying PDF pages: {str(e)}</p>"

def launch_ultra_accurate_rag():
    """Launch the ultra-accurate RAG interface."""
    interface = UltraAccurateInterface()
    return interface.interface

def main():
    """Main function with enhanced system."""
    print("🎯 Starting Ultra-Accurate RAG System with Custom Reranker & Chat Interface")
    
    try:
        import google.colab
        print("📱 Running in Google Colab")
        from google.colab import auth
        auth.authenticate_user()
        print("✅ Authentication completed")
    except ImportError:
        print("💻 Running in local environment")
    
    print("\n🚀 **Ultra-Accurate Features:**")
    print("- 🤖 AI Provider Choice: VertexAI or Groq")
    print("- 🔍 Hybrid retrieval: Dense (HNSW) + Sparse (BM25)")
    print("- 🎯 Custom reranker with semantic similarity & keyword matching")
    print("- 🖼️ Semantic image-text matching with chunk context")
    print("- 📊 Enhanced chunking with context preservation")
    print("- 🔗 Multi-stage relevance filtering")
    print("- 📈 Accuracy scoring and quality metrics")
    print("- 📝 Sentence-level inline citations")
    print("- 🔍 Interactive Source Inspector modal")
    print("- 📄 Complete PDF page display with chunk context")
    print("- 💬 Chat-like interface with tabbed sections")
    
    interface = launch_ultra_accurate_rag()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        debug=False,
        show_error=True
    )

if __name__ == "__main__":
    main()
