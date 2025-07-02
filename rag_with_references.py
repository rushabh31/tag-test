# -*- coding: utf-8 -*-
"""Highly-Accurate-RAG-with-Hybrid-Retrieval-and-Reranking.ipynb

Enhanced RAG system with:
1. Hybrid retrieval (Dense HNSW + Sparse BM25)
2. Cross-encoder reranking for accuracy
3. Semantic image-text matching
4. Multi-stage relevance filtering
5. Query expansion and refinement
6. Better chunking and context preservation
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

    def extract_images_from_pdf(self, file_path: str) -> List[EnhancedImageData]:
        """Extract and analyze images with enhanced semantic processing."""
        images = []
        
        try:
            # Read PDF content first to get context
            pdf_text = self._extract_raw_text(file_path)
            pdf_images = convert_from_path(file_path, dpi=200)
            
            for page_num, page_image in enumerate(pdf_images, 1):
                # Get page text for context
                page_context = self._get_page_context(pdf_text, page_num)
                
                # Process images on this page
                page_images = self._extract_page_images(page_image, page_num, page_context)
                images.extend(page_images)
                    
        except Exception as e:
            logger.error(f"Error extracting images from PDF: {str(e)}")
            
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
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            
            # Enhanced edge detection
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 30, 80)
            
            # Morphological operations to connect nearby edges
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and sort contours by area
            min_area = 5000
            valid_contours = [(cv2.contourArea(c), c) for c in contours if cv2.contourArea(c) > min_area]
            valid_contours.sort(reverse=True)  # Largest first
            
            processed_count = 0
            for area, contour in valid_contours[:3]:  # Top 3 largest regions
                try:
                    image_data = self._process_image_region(
                        page_image, contour, page_num, page_context, processed_count
                    )
                    if image_data and image_data.confidence > 0.4:  # Only high-confidence images
                        images.append(image_data)
                        processed_count += 1
                except Exception as e:
                    logger.warning(f"Error processing image region: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error extracting page images: {e}")
            
        return images

    def _process_image_region(self, page_image: Image.Image, contour, 
                            page_num: int, page_context: str, 
                            region_idx: int) -> Optional[EnhancedImageData]:
        """Process a single image region."""
        try:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out very thin regions (likely text lines)
            aspect_ratio = w / h
            if aspect_ratio > 10 or aspect_ratio < 0.1:
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
            height, width = img_array.shape[:2]
            
            # Calculate image statistics
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
            
            # Detect lines (for tables/charts)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
            
            h_line_count = np.sum(horizontal_lines > 0)
            v_line_count = np.sum(vertical_lines > 0)
            
            # Analyze text characteristics
            text_lower = ocr_text.lower()
            context_lower = context.lower()
            
            # Enhanced type detection
            confidence = 0.3  # Base confidence
            image_type = "figure"
            analysis = ""
            
            # Table detection
            if (h_line_count > 100 and v_line_count > 100) or \
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
            analysis += f" [Size: {width}x{height}px, Quality: {'High' if min(width, height) > 150 else 'Medium'}]"
            
            return image_type, analysis, confidence
            
        except Exception as e:
            logger.error(f"Enhanced image analysis failed: {e}")
            return "figure", f"Image analysis error: {str(e)}", 0.2

    def extract_text_from_pdf(self, file_path: str) -> PDFDocument:
        """Extract text with enhanced processing."""
        doc = PDFDocument(filename=os.path.basename(file_path))
        
        try:
            # Extract images first
            doc.images = self.extract_images_from_pdf(file_path)
            
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
            
            logger.info(f"Created {len(doc.chunks)} enhanced chunks")
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
    """VertexAI embeddings optimized for accuracy."""
    
    def __init__(self, vertex_gen_ai: VertexGenAI, model_name: str = "text-embedding-004"):
        self.vertex_gen_ai = vertex_gen_ai
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
            
            logger.info(f"Embedding {len(processed_texts)} documents")
            embeddings_response = self.vertex_gen_ai.get_embeddings(processed_texts, self.model_name)
            
            embeddings = []
            for embedding in embeddings_response:
                if hasattr(embedding, 'values'):
                    vector = np.array(embedding.values, dtype=np.float32)
                    # L2 normalize for cosine similarity
                    vector = vector / (np.linalg.norm(vector) + 1e-12)
                    embeddings.append(vector.tolist())
                else:
                    embeddings.append(self._get_zero_vector())
            
            if embeddings and self._embedding_dimension is None:
                self._embedding_dimension = len(embeddings[0])
                logger.info(f"Set embedding dimension to {self._embedding_dimension}")
            
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

class HybridRetriever:
    """Hybrid retriever combining dense (HNSW) and sparse (BM25) retrieval with reranking."""
    
    def __init__(self, documents: List[Document] = None, vertex_gen_ai: VertexGenAI = None):
        logger.info("Initializing HybridRetriever with reranking...")
        
        self.vertex_gen_ai = vertex_gen_ai
        self.embeddings = VertexAIEmbeddings(vertex_gen_ai) if vertex_gen_ai else None
        
        # Storage
        self.documents = []
        self.document_embeddings = []
        
        # Dense retrieval (HNSW)
        self.hnsw_index = None
        self.dimension = None
        
        # Sparse retrieval (BM25)
        self.bm25 = None
        self.tokenized_docs = []
        
        # Reranking
        self.reranker = None
        self._init_reranker()
        
        if documents and len(documents) > 0:
            self.add_documents(documents)
    
    def _init_reranker(self):
        """Initialize cross-encoder for reranking."""
        try:
            # Use a lightweight cross-encoder
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("Initialized cross-encoder reranker")
        except Exception as e:
            logger.warning(f"Could not initialize reranker: {e}")
            self.reranker = None
    
    def add_documents(self, documents: List[Document]):
        """Add documents to both dense and sparse indices."""
        try:
            logger.info(f"Adding {len(documents)} documents to hybrid index")
            
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
                    logger.info(f"Initialized HNSW with dimension {self.dimension}")
                
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
            
            logger.info(f"Successfully added {len(documents)} documents to hybrid index")
            
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
            
            logger.info(f"Hybrid retrieval for query: {query[:50]}...")
            
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
            
            logger.info(f"Retrieved {len(final_results)} documents after hybrid processing")
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
            if max(scores) > min(scores):
                min_score, max_score = min(scores), max(scores)
                return [(doc, (score - min_score) / (max_score - min_score)) 
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
        """Rerank results using cross-encoder."""
        if not self.reranker or len(results) <= 1:
            return results
        
        try:
            # Prepare query-document pairs
            query_doc_pairs = []
            for doc, _ in results:
                # Use original text for reranking if available
                text = doc.metadata.get('original_text', doc.page_content)
                # Truncate long texts
                if len(text) > 2000:
                    text = text[:2000] + "..."
                query_doc_pairs.append([query, text])
            
            # Get reranking scores
            rerank_scores = self.reranker.predict(query_doc_pairs)
            
            # Combine with original scores
            reranked_results = []
            for i, (doc, original_score) in enumerate(results):
                rerank_score = float(rerank_scores[i])
                # Weighted combination: 60% rerank, 40% original
                final_score = 0.6 * rerank_score + 0.4 * original_score
                doc.metadata['rerank_score'] = rerank_score
                doc.metadata['final_score'] = final_score
                reranked_results.append((doc, final_score))
            
            # Sort by final score
            reranked_results.sort(key=lambda x: x[1], reverse=True)
            return reranked_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
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

class UltraAccurateRAGSystem:
    """Ultra-accurate RAG system with all enhancements."""
    
    def __init__(self, vertex_gen_ai: VertexGenAI = None):
        logger.info("Initializing UltraAccurateRAGSystem...")
        
        self.vertex_gen_ai = vertex_gen_ai
        self.processor = AdvancedPDFProcessor()
        self.documents = {}
        self.hybrid_retriever = None
        self.conversation_history = []
        self.reference_counter = 1
        
        if vertex_gen_ai:
            self.hybrid_retriever = HybridRetriever(vertex_gen_ai=vertex_gen_ai)
    
    def upload_pdf(self, file_path: str) -> str:
        """Process and index PDF with all enhancements."""
        try:
            logger.info(f"Processing PDF with enhanced accuracy: {file_path}")
            
            doc = self.processor.process_pdf(file_path)
            self.documents[doc.filename] = doc
            
            if self.hybrid_retriever:
                self.hybrid_retriever.add_documents(doc.langchain_documents)
            
            # Detailed stats
            high_conf_images = sum(1 for img in doc.images if img.confidence > 0.6)
            semantic_chunks = sum(1 for chunk in doc.chunks if chunk.semantic_density > 10)
            
            result = f"""✅ **Enhanced Processing Complete: {doc.filename}**

**Content Analysis:**
- 📄 Pages: {len(doc.pages)}
- 🧩 Chunks: {len(doc.chunks)} (with context preservation)
- 🖼️ Images: {len(doc.images)} ({high_conf_images} high-confidence)
- 🎯 Semantic chunks: {semantic_chunks}

**Quality Metrics:**
- Image analysis confidence: {np.mean([img.confidence for img in doc.images]):.2f}
- Chunk semantic density: {np.mean([chunk.semantic_density for chunk in doc.chunks]):.1f}
"""
            
            logger.info(result.replace('\n', ' '))
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
        
        try:
            logger.info(f"Processing query with enhanced accuracy: {query[:50]}...")
            
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
                
                # Enhanced scoring
                final_score = metadata.get('final_score', metadata.get('combined_score', 0.0))
                dense_score = metadata.get('dense_score', 0.0)
                sparse_score = metadata.get('sparse_score', 0.0)
                rerank_score = metadata.get('rerank_score', 0.0)
                
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
                    "type": "text",
                    "semantic_density": metadata.get('semantic_density', 0.0)
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
                
                # Enhanced image retrieval
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
                            
                            if img_relevance > 0.3:  # Only relevant images
                                all_images.append({
                                    "document": doc_name,
                                    "page": img_data.page_number,
                                    "type": img_data.image_type,
                                    "analysis": img_data.analysis,
                                    "confidence": img_data.confidence,
                                    "base64": img_data.base64_string,
                                    "reference_id": self.reference_counter,
                                    "relevance_score": img_relevance,
                                    "ocr_text": img_data.ocr_text[:150],
                                    "keywords": img_data.semantic_keywords
                                })
                                
                                # Add image reference
                                img_ref = {
                                    "id": self.reference_counter,
                                    "document": doc_name,
                                    "pages": [img_data.page_number],
                                    "final_score": img_relevance,
                                    "content": f"{img_data.image_type}: {img_data.analysis[:150]}...",
                                    "type": "image",
                                    "image_confidence": img_data.confidence
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
            response = self.vertex_gen_ai.generate_content(prompt)
            
            if not response:
                response = "I apologize, but I couldn't generate a comprehensive response. Please try rephrasing your question with more specific terms."
            
            # Add clickable reference links
            for ref_id, ref_text in ref_links.items():
                response = response.replace(f"[{ref_id}]", f"[[{ref_id}]](#ref{ref_id})")
            
            self.conversation_history.append((query, response))
            
            logger.info(f"Generated response with {len(references)} references and {len(top_images)} images")
            
            return {
                "answer": response,
                "references": references,
                "images": top_images
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
            image_context = "\n\n=== RELEVANT VISUAL CONTENT ===\n"
            for img in images:
                img_context = f"""
Image Reference [{img['reference_id']}] - {img['type'].title()} from {img['document']}, Page {img['page']}:
- Analysis: {img['analysis']}
- OCR Content: {img['ocr_text']}
- Keywords: {', '.join(img['keywords'])}
- Relevance: {img['relevance_score']:.2f}
"""
                image_context += img_context
            
            context += image_context
        
        return context
    
    def _create_enhanced_prompt(self, query: str, context: str, ref_links: Dict) -> str:
        """Create enhanced prompt for better responses."""
        
        prompt = f"""You are an expert research assistant providing comprehensive, accurate analysis. Your task is to answer the user's question using ONLY the provided context with maximum accuracy and detail.

CRITICAL ACCURACY REQUIREMENTS:
1. Use ONLY information explicitly present in the provided context
2. Add numbered citations [1], [2], etc. after EVERY factual statement
3. For numerical data, charts, tables: be extremely precise and detailed
4. For visual content: provide thorough descriptions and interpretations
5. If information is insufficient, clearly state what cannot be determined
6. Never make assumptions or add external knowledge

RESPONSE STRUCTURE:
1. Direct answer to the question with citations
2. Supporting evidence with detailed explanations
3. Visual content analysis (if applicable)
4. Summary of key findings

CONTEXT WITH REFERENCES:
{context}

AVAILABLE REFERENCE IDs: {list(ref_links.keys())}

USER QUESTION: {query}

INSTRUCTIONS FOR VISUAL CONTENT:
- For charts/graphs: Describe data points, trends, axes, values precisely
- For tables: Explain structure, key values, relationships systematically  
- For diagrams: Detail components, connections, flow sequences
- Always cite the specific image reference number

Provide a comprehensive, well-researched response with precise citations for every claim:"""

        return prompt
    
    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
        self.reference_counter = 1
        return "Conversation reset successfully."

# Update the interface to use the new system
class UltraAccurateInterface:
    """Interface for ultra-accurate RAG system."""

    def __init__(self):
        self.rag_system = None
        self.vertex_gen_ai = None
        self.current_references = []
        self.current_images = []
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
        .accuracy-badge {
            background: linear-gradient(45deg, #28a745, #20c997);
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
        """

        with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Ultra-Accurate Research Assistant") as self.interface:
            
            # Enhanced header
            gr.HTML("""
            <div class="logo-container">
                <div style="margin-right: 15px; font-size: 32px;">🎯</div>
                <div class="logo-text">
                    Ultra-Accurate Research Assistant
                    <div class="accuracy-badge">Enhanced with Hybrid Retrieval + Reranking</div>
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
                    placeholder="Ask detailed questions about your documents (e.g., 'What are the key findings in the charts?', 'Explain the process diagram on page 5')",
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
                label="Select PDF Documents for Analysis",
                file_types=[".pdf"],
                file_count="multiple",
                visible=False
            )
            
            status_display = gr.Markdown("🤖 **Status:** Ready to initialize ultra-accurate system")
            
            # Enhanced references section
            gr.Markdown("## 📚 Enhanced Text References (with Accuracy Scores)")
            text_references = gr.HTML(value="<p>No references yet. Upload documents and ask questions to see detailed references with accuracy scores.</p>")
            
            # Enhanced images section
            gr.Markdown("## 🖼️ Ultra-Accurate Visual References (Top 3 Most Relevant)")
            
            with gr.Row():
                image1 = gr.Image(label="Most Relevant Image", visible=False, height=250)
                image2 = gr.Image(label="Second Most Relevant", visible=False, height=250)
                image3 = gr.Image(label="Third Most Relevant", visible=False, height=250)
            
            image_info = gr.HTML(value="<p>No visual content found yet. Upload documents with charts, diagrams, or tables to see them here.</p>")

            initialized = gr.State(False)

            # Enhanced event handlers
            def handle_upload_click():
                return gr.update(visible=True)
            
            def handle_upload_files(files, is_initialized):
                if not is_initialized:
                    try:
                        self.vertex_gen_ai = VertexGenAI()
                        test_response = self.vertex_gen_ai.generate_content("Test accuracy")
                        if test_response:
                            self.rag_system = UltraAccurateRAGSystem(vertex_gen_ai=self.vertex_gen_ai)
                            is_initialized = True
                        else:
                            return "❌ Failed to initialize VertexAI for ultra-accurate processing", False, gr.update(visible=False)
                    except Exception as e:
                        return f"❌ Ultra-accurate initialization error: {str(e)}", False, gr.update(visible=False)
                
                if not files:
                    return "❌ No files selected for processing", is_initialized, gr.update(visible=False)
                
                results = []
                for file in files:
                    try:
                        result = self.rag_system.upload_pdf(file.name)
                        results.append(result)
                    except Exception as e:
                        results.append(f"❌ Error processing {os.path.basename(file.name)}: {str(e)}")
                
                status = "## 📄 Ultra-Accurate Processing Results\n" + "\n".join(results)
                return status, is_initialized, gr.update(visible=False)
            
            def handle_ask(message, history, is_initialized):
                if not is_initialized or not self.rag_system:
                    error_msg = "❌ Please upload documents first to enable ultra-accurate processing"
                    if history is None:
                        history = []
                    return history + [[message, error_msg]], "", "", "", "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                
                if not message.strip():
                    if history is None:
                        history = []
                    return history, message, "", "", "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                
                try:
                    # Get ultra-accurate response
                    result = self.rag_system.ask(message)
                    answer = result["answer"]
                    self.current_references = result["references"]
                    self.current_images = result["images"]
                    
                    # Update chat history
                    if history is None:
                        history = []
                    updated_history = history + [[message, answer]]
                    
                    # Format enhanced references
                    text_refs_html = self._format_enhanced_references()
                    
                    # Format enhanced image info
                    images_info_html = self._format_enhanced_images()
                    
                    # Prepare images for display
                    img1, img2, img3 = self._prepare_enhanced_images()
                    
                    return (updated_history, "", text_refs_html, images_info_html, "",
                           img1, img2, img3)
                    
                except Exception as e:
                    error_msg = f"❌ Ultra-accurate processing error: {str(e)}"
                    if history is None:
                        history = []
                    return (history + [[message, error_msg]], "", "", "", "",
                           gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
            
            def handle_reset():
                if self.rag_system:
                    self.rag_system.reset_conversation()
                self.current_references = []
                self.current_images = []
                return ([], "🔄 Ultra-accurate conversation reset", 
                       "<p>No references yet.</p>", 
                       "<p>No images yet.</p>",
                       gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
            
            # Wire up events
            upload_btn.click(fn=handle_upload_click, outputs=upload_files)
            upload_files.upload(fn=handle_upload_files, inputs=[upload_files, initialized], outputs=[status_display, initialized, upload_files])
            ask_btn.click(fn=handle_ask, inputs=[msg_input, chatbot, initialized], outputs=[chatbot, msg_input, text_references, image_info, status_display, image1, image2, image3])
            msg_input.submit(fn=handle_ask, inputs=[msg_input, chatbot, initialized], outputs=[chatbot, msg_input, text_references, image_info, status_display, image1, image2, image3])
            reset_btn.click(fn=handle_reset, outputs=[chatbot, status_display, text_references, image_info, image1, image2, image3])

    def _format_enhanced_references(self):
        """Format references with enhanced accuracy information."""
        if not self.current_references:
            return "<p>No references available.</p>"
        
        html_parts = []
        text_refs = [ref for ref in self.current_references if ref.get("type") != "image"]
        
        for ref in text_refs:
            ref_id = ref["id"]
            doc = ref["document"]
            pages = ref["pages"]
            final_score = ref.get("final_score", 0.0)
            dense_score = ref.get("dense_score", 0.0)
            sparse_score = ref.get("sparse_score", 0.0) 
            rerank_score = ref.get("rerank_score", 0.0)
            semantic_density = ref.get("semantic_density", 0.0)
            content = ref["content"]
            
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
            
            img_html = f"""
            <div class="image-card">
                <strong>Image {i+1} {f'[{ref_id}]' if ref_id else ''}</strong> 🖼️ {img_type.title()}
                <span class="score-badge">Relevance: {relevance_score:.3f}</span>
                <span class="score-badge">Confidence: {confidence:.3f}</span>
                <span class="score-badge">Combined: {combined_score:.3f}</span>
                <br><strong>Source:</strong> {doc}, Page {page}
                <br><strong>Analysis:</strong> {analysis}
                {f'<br><strong>OCR Preview:</strong> "{ocr_preview}..."' if ocr_preview else ''}
                {f'<br><strong>Keywords:</strong> {keywords}' if keywords else ''}
            </div>
            """
            html_parts.append(img_html)
        
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
        
        # Enhanced labels with scores
        if len(display_images) > 0:
            img_data = self.current_images[0]
            score = img_data['relevance_score'] * img_data['confidence']
            img1 = gr.update(
                value=display_images[0], 
                visible=True,
                label=f"🥇 {img_data['type'].title()} (Score: {score:.3f}) - {img_data['document']}"
            )
        
        if len(display_images) > 1:
            img_data = self.current_images[1]
            score = img_data['relevance_score'] * img_data['confidence']
            img2 = gr.update(
                value=display_images[1],
                visible=True,
                label=f"🥈 {img_data['type'].title()} (Score: {score:.3f}) - {img_data['document']}"
            )
        
        if len(display_images) > 2:
            img_data = self.current_images[2]
            score = img_data['relevance_score'] * img_data['confidence']
            img3 = gr.update(
                value=display_images[2],
                visible=True,
                label=f"🥉 {img_data['type'].title()} (Score: {score:.3f}) - {img_data['document']}"
            )
        
        return img1, img2, img3

def launch_ultra_accurate_rag():
    """Launch the ultra-accurate RAG interface."""
    interface = UltraAccurateInterface()
    return interface.interface

def main():
    """Main function with enhanced system."""
    print("🎯 Starting Ultra-Accurate RAG System")
    
    try:
        import google.colab
        print("📱 Running in Google Colab")
        from google.colab import auth
        auth.authenticate_user()
        print("✅ Authentication completed")
    except ImportError:
        print("💻 Running in local environment")
    
    print("\n🚀 **Ultra-Accurate Features:**")
    print("- 🔍 Hybrid retrieval: Dense (HNSW) + Sparse (BM25)")
    print("- 🎯 Cross-encoder reranking for maximum accuracy")
    print("- 🖼️ Semantic image-text matching")
    print("- 📊 Enhanced chunking with context preservation")
    print("- 🔗 Multi-stage relevance filtering")
    print("- 📈 Accuracy scoring and quality metrics")
    
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
