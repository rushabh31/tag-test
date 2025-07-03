# -*- coding: utf-8 -*-
"""Ultra-Accurate-Multimodal-RAG-with-Direct-Image-Analysis.ipynb

Ultra-accurate RAG system with:
1. Direct multimodal image analysis (no OCR dependency)
2. Enhanced visual understanding using multimodal LLM
3. Custom MMR reranking with multimodal scoring
4. VertexAI embeddings + multimodal capabilities
5. Full PDF page images with direct analysis
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

import hnswlib
from rank_bm25 import BM25Okapi

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
import gradio as gr

# Enhanced VertexAI with multimodal capabilities
import vertexai
from google.oauth2.credentials import Credentials
from helpers import get_coin_token
from vertexai.generative_models import GenerativeModel, ContentsType, Part, Content, Image as VertexImage

class MultimodalVertexGenAI:
    """Enhanced VertexAI with multimodal capabilities for direct image analysis."""
    
    def __init__(self):
        credentials = Credentials(get_coin_token())
        vertexai.init(
            project="pri-gen-ai",
            api_transport="rest",
            api_endpoint="https://xyz/vertex",
            credentials=credentials,
        )
        self.metadata = [("x-user", os.getenv("USERNAME"))]
        self.text_model = GenerativeModel("gemini-1.5-pro-002")
        self.multimodal_model = GenerativeModel("gemini-1.5-pro-002")  # Same model for multimodal

    def generate_content(self, prompt: str = "Provide interesting trivia"):
        """Generate content based on text prompt."""
        resp = self.text_model.generate_content(prompt, metadata=self.metadata)
        return resp.text if resp else None

    def analyze_image_directly(self, image: Image.Image, query: str = None) -> Dict[str, Any]:
        """Directly analyze image using multimodal LLM - much more accurate than OCR."""
        try:
            # Convert PIL Image to bytes
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Create image part
            image_part = Part.from_image(VertexImage.from_bytes(img_byte_arr))
            
            # Create comprehensive analysis prompt
            if query:
                analysis_prompt = f"""Analyze this image in the context of the query: "{query}"

Please provide a comprehensive analysis including:
1. Image Type: (chart, table, diagram, figure, etc.)
2. Detailed Content: Describe exactly what you see with specific details
3. Data Values: If there are numbers, percentages, or data, list them precisely
4. Text Content: Extract all readable text exactly as it appears
5. Visual Elements: Describe charts, graphs, tables, diagrams in detail
6. Relevance to Query: How does this image relate to the query?
7. Key Insights: What are the main takeaways from this visual content?
8. Confidence: Rate your confidence in this analysis (0.0-1.0)

Format your response as a structured analysis with clear sections."""
            else:
                analysis_prompt = """Analyze this image comprehensively and provide:

1. Image Type: (chart, table, diagram, figure, etc.)
2. Detailed Content: Describe exactly what you see with specific details
3. Data Values: If there are numbers, percentages, or data, list them precisely
4. Text Content: Extract all readable text exactly as it appears
5. Visual Elements: Describe any charts, graphs, tables, diagrams in detail
6. Key Information: What are the main pieces of information in this image?
7. Structure: How is the information organized or presented?
8. Confidence: Rate your confidence in this analysis (0.0-1.0)

Provide a detailed, structured analysis."""
            
            text_part = Part.from_text(analysis_prompt)
            
            # Create content with both text and image
            content = Content(
                parts=[text_part, image_part],
                role="user"
            )
            
            # Generate response
            response = self.multimodal_model.generate_content(content, metadata=self.metadata)
            
            if response and response.text:
                # Parse the structured response
                analysis_text = response.text
                
                # Extract key information using parsing
                parsed_info = self._parse_multimodal_response(analysis_text)
                
                return {
                    "full_analysis": analysis_text,
                    "image_type": parsed_info.get("type", "figure"),
                    "detailed_content": parsed_info.get("content", ""),
                    "extracted_text": parsed_info.get("text", ""),
                    "data_values": parsed_info.get("data", []),
                    "visual_elements": parsed_info.get("visual", ""),
                    "key_insights": parsed_info.get("insights", ""),
                    "confidence": parsed_info.get("confidence", 0.8),
                    "relevance_explanation": parsed_info.get("relevance", "")
                }
            else:
                return self._fallback_analysis()
                
        except Exception as e:
            logger.error(f"Error in multimodal image analysis: {e}")
            return self._fallback_analysis()
    
    def compare_image_to_query(self, image: Image.Image, query: str) -> Dict[str, float]:
        """Compare image content to query using multimodal understanding."""
        try:
            # Convert PIL Image to bytes
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            image_part = Part.from_image(VertexImage.from_bytes(img_byte_arr))
            
            relevance_prompt = f"""Given this query: "{query}"

Analyze the image and provide relevance scores (0.0-1.0) for:

1. Content Relevance: How well does the image content match the query topic?
2. Data Relevance: If the query asks for specific data, how well does the image provide it?
3. Visual Relevance: How appropriate is this image type for answering the query?
4. Semantic Relevance: How semantically related is the image to the query concepts?
5. Overall Relevance: Overall score for how useful this image is for the query

Respond in this exact format:
Content Relevance: X.X
Data Relevance: X.X  
Visual Relevance: X.X
Semantic Relevance: X.X
Overall Relevance: X.X
Explanation: [Brief explanation of relevance]"""
            
            text_part = Part.from_text(relevance_prompt)
            
            content = Content(
                parts=[text_part, image_part],
                role="user"
            )
            
            response = self.multimodal_model.generate_content(content, metadata=self.metadata)
            
            if response and response.text:
                return self._parse_relevance_scores(response.text)
            else:
                return {"overall_relevance": 0.5}
                
        except Exception as e:
            logger.error(f"Error in multimodal relevance comparison: {e}")
            return {"overall_relevance": 0.5}
    
    def _parse_multimodal_response(self, response_text: str) -> Dict[str, Any]:
        """Parse structured multimodal response."""
        parsed = {}
        
        try:
            # Extract image type
            type_match = re.search(r'Image Type:\s*([^\n]+)', response_text, re.IGNORECASE)
            if type_match:
                type_text = type_match.group(1).strip()
                if 'chart' in type_text.lower() or 'graph' in type_text.lower():
                    parsed["type"] = "chart"
                elif 'table' in type_text.lower():
                    parsed["type"] = "table"
                elif 'diagram' in type_text.lower() or 'flow' in type_text.lower():
                    parsed["type"] = "diagram"
                else:
                    parsed["type"] = "figure"
            
            # Extract detailed content
            content_match = re.search(r'Detailed Content:\s*([^\n]*(?:\n(?!(?:Data Values|Text Content|Visual Elements):)[^\n]*)*)', response_text, re.IGNORECASE)
            if content_match:
                parsed["content"] = content_match.group(1).strip()
            
            # Extract text content
            text_match = re.search(r'Text Content:\s*([^\n]*(?:\n(?!(?:Visual Elements|Key|Structure):)[^\n]*)*)', response_text, re.IGNORECASE)
            if text_match:
                parsed["text"] = text_match.group(1).strip()
            
            # Extract data values
            data_match = re.search(r'Data Values:\s*([^\n]*(?:\n(?!(?:Text Content|Visual Elements):)[^\n]*)*)', response_text, re.IGNORECASE)
            if data_match:
                data_text = data_match.group(1).strip()
                # Extract numbers and percentages
                numbers = re.findall(r'\d+\.?\d*%?', data_text)
                parsed["data"] = numbers
            
            # Extract visual elements
            visual_match = re.search(r'Visual Elements:\s*([^\n]*(?:\n(?!(?:Key|Relevance|Confidence):)[^\n]*)*)', response_text, re.IGNORECASE)
            if visual_match:
                parsed["visual"] = visual_match.group(1).strip()
            
            # Extract key insights
            insights_match = re.search(r'Key (?:Insights|Information):\s*([^\n]*(?:\n(?!(?:Confidence|Structure):)[^\n]*)*)', response_text, re.IGNORECASE)
            if insights_match:
                parsed["insights"] = insights_match.group(1).strip()
            
            # Extract confidence
            conf_match = re.search(r'Confidence:\s*([0-9.]+)', response_text, re.IGNORECASE)
            if conf_match:
                parsed["confidence"] = float(conf_match.group(1))
            
            # Extract relevance explanation
            rel_match = re.search(r'Relevance to Query:\s*([^\n]*(?:\n(?!(?:Key|Confidence):)[^\n]*)*)', response_text, re.IGNORECASE)
            if rel_match:
                parsed["relevance"] = rel_match.group(1).strip()
                
        except Exception as e:
            logger.error(f"Error parsing multimodal response: {e}")
        
        return parsed
    
    def _parse_relevance_scores(self, response_text: str) -> Dict[str, float]:
        """Parse relevance scores from multimodal response."""
        scores = {}
        
        try:
            patterns = {
                "content_relevance": r'Content Relevance:\s*([0-9.]+)',
                "data_relevance": r'Data Relevance:\s*([0-9.]+)',
                "visual_relevance": r'Visual Relevance:\s*([0-9.]+)',
                "semantic_relevance": r'Semantic Relevance:\s*([0-9.]+)',
                "overall_relevance": r'Overall Relevance:\s*([0-9.]+)'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    scores[key] = float(match.group(1))
                else:
                    scores[key] = 0.5  # Default
            
            # Extract explanation
            exp_match = re.search(r'Explanation:\s*([^\n]*)', response_text, re.IGNORECASE)
            if exp_match:
                scores["explanation"] = exp_match.group(1).strip()
                
        except Exception as e:
            logger.error(f"Error parsing relevance scores: {e}")
            scores = {"overall_relevance": 0.5}
        
        return scores
    
    def _fallback_analysis(self) -> Dict[str, Any]:
        """Fallback analysis when multimodal fails."""
        return {
            "full_analysis": "Multimodal analysis unavailable",
            "image_type": "figure",
            "detailed_content": "",
            "extracted_text": "",
            "data_values": [],
            "visual_elements": "",
            "key_insights": "",
            "confidence": 0.3,
            "relevance_explanation": ""
        }

    def get_embeddings(self, texts: list[str], model_name: str = "text-embedding-004"):
        """Get embeddings for texts using Vertex AI."""
        from vertexai.language_models import TextEmbeddingModel
        
        model = TextEmbeddingModel.from_pretrained(model_name, metadata=self.metadata)
        
        if not texts:
            return []
        
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = model.get_embeddings(texts, metadata=self.metadata)
        return embeddings

@dataclass
class MultimodalImageData:
    """Enhanced image data with direct multimodal analysis."""
    image: Image.Image
    page_number: int
    bbox: Optional[Tuple[int, int, int, int]] = None
    base64_string: str = ""
    
    # Multimodal analysis results
    multimodal_analysis: Dict[str, Any] = field(default_factory=dict)
    image_type: str = ""
    detailed_content: str = ""
    extracted_text: str = ""
    data_values: List[str] = field(default_factory=list)
    visual_elements: str = ""
    key_insights: str = ""
    confidence: float = 0.0
    
    # Legacy fields for compatibility
    ocr_text: str = ""
    analysis: str = ""
    semantic_keywords: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    context_text: str = ""
    visual_features: Dict[str, float] = field(default_factory=dict)

@dataclass
class PageImageData:
    """Full page image data with multimodal analysis."""
    page_number: int
    page_image: Image.Image
    base64_string: str
    multimodal_summary: str = ""  # Summary from multimodal analysis
    extracted_regions: List[Dict] = field(default_factory=list)
    confidence: float = 0.0

@dataclass
class UltraChunk:
    """Ultra-enhanced chunk with multimodal context."""
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
    page_coverage: float = 0.0
    multimodal_context: str = ""  # Context from multimodal image analysis
    
    def __post_init__(self):
        content = f"{self.text}{self.filename}{self.page_numbers}"
        self.chunk_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        
        sentences = len([s for s in self.text.split('.') if s.strip()])
        words = len(self.text.split())
        self.semantic_density = words / max(sentences, 1)

    def to_document(self) -> Document:
        # Include multimodal context
        enhanced_content = f"{self.context_before}\n\n{self.text}"
        if self.multimodal_context:
            enhanced_content += f"\n\n[Multimodal Visual Context]: {self.multimodal_context}"
        enhanced_content += f"\n\n{self.context_after}".strip()
        
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
                "relevance_scores": self.relevance_scores,
                "multimodal_context": self.multimodal_context
            }
        )

@dataclass
class MultimodalDocument:
    """Document with multimodal analysis capabilities."""
    filename: str
    content: str = ""
    pages: Dict[int, str] = field(default_factory=dict)
    char_to_page_map: List[int] = field(default_factory=list)
    chunks: List[UltraChunk] = field(default_factory=list)
    images: List[MultimodalImageData] = field(default_factory=list)
    page_images: Dict[int, PageImageData] = field(default_factory=dict)
    page_to_chunks: Dict[int, List[int]] = field(default_factory=dict)
    page_to_images: Dict[int, List[int]] = field(default_factory=dict)

    @property
    def langchain_documents(self) -> List[Document]:
        return [chunk.to_document() for chunk in self.chunks]

class MultimodalImageMatcher:
    """Advanced image-text matching using multimodal LLM."""
    
    def __init__(self, multimodal_ai: MultimodalVertexGenAI):
        self.multimodal_ai = multimodal_ai
        
    def analyze_image_with_context(self, image: Image.Image, context_text: str = "", query: str = "") -> MultimodalImageData:
        """Analyze image using multimodal LLM with context."""
        try:
            # Get comprehensive multimodal analysis
            analysis_result = self.multimodal_ai.analyze_image_directly(image, query)
            
            # Convert to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG", quality=90)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Create enhanced image data
            image_data = MultimodalImageData(
                image=image,
                page_number=0,  # Will be set by caller
                base64_string=img_base64,
                multimodal_analysis=analysis_result,
                image_type=analysis_result.get("image_type", "figure"),
                detailed_content=analysis_result.get("detailed_content", ""),
                extracted_text=analysis_result.get("extracted_text", ""),
                data_values=analysis_result.get("data_values", []),
                visual_elements=analysis_result.get("visual_elements", ""),
                key_insights=analysis_result.get("key_insights", ""),
                confidence=analysis_result.get("confidence", 0.8),
                context_text=context_text
            )
            
            # Set legacy fields for compatibility
            image_data.ocr_text = analysis_result.get("extracted_text", "")
            image_data.analysis = analysis_result.get("full_analysis", "")
            
            # Extract semantic keywords from multimodal analysis
            image_data.semantic_keywords = self._extract_keywords_from_analysis(analysis_result)
            
            return image_data
            
        except Exception as e:
            logger.error(f"Error in multimodal image analysis: {e}")
            return self._create_fallback_image_data(image)
    
    def calculate_multimodal_similarity(self, query: str, image_data: MultimodalImageData) -> float:
        """Calculate similarity using multimodal understanding."""
        try:
            # Get multimodal relevance scores
            relevance_scores = self.multimodal_ai.compare_image_to_query(image_data.image, query)
            
            # Extract scores
            overall_relevance = relevance_scores.get("overall_relevance", 0.5)
            content_relevance = relevance_scores.get("content_relevance", 0.5)
            data_relevance = relevance_scores.get("data_relevance", 0.5)
            visual_relevance = relevance_scores.get("visual_relevance", 0.5)
            semantic_relevance = relevance_scores.get("semantic_relevance", 0.5)
            
            # Weighted combination for final score
            final_score = (
                0.4 * overall_relevance +
                0.25 * content_relevance +
                0.15 * data_relevance +
                0.1 * visual_relevance +
                0.1 * semantic_relevance
            )
            
            # Apply confidence multiplier
            final_score *= image_data.confidence
            
            # Store relevance explanation
            explanation = relevance_scores.get("explanation", "")
            if hasattr(image_data, 'relevance_explanation'):
                image_data.relevance_explanation = explanation
            
            return min(final_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating multimodal similarity: {e}")
            return 0.5
    
    def _extract_keywords_from_analysis(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Extract keywords from multimodal analysis."""
        keywords = []
        
        try:
            # Extract from different analysis components
            components = [
                analysis_result.get("extracted_text", ""),
                analysis_result.get("detailed_content", ""),
                analysis_result.get("key_insights", ""),
                " ".join(analysis_result.get("data_values", []))
            ]
            
            for component in components:
                if component:
                    # Extract important words
                    words = re.findall(r'\b[A-Za-z]{3,}\b', component)
                    keywords.extend(words[:5])  # Limit per component
            
            # Remove duplicates and return
            return list(set(keywords))[:15]  # Limit total keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def _create_fallback_image_data(self, image: Image.Image) -> MultimodalImageData:
        """Create fallback image data when multimodal analysis fails."""
        buffered = BytesIO()
        image.save(buffered, format="PNG", quality=90)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return MultimodalImageData(
            image=image,
            page_number=0,
            base64_string=img_base64,
            image_type="figure",
            detailed_content="Multimodal analysis unavailable",
            confidence=0.3
        )

class MultimodalDocumentProcessor:
    """Document processor with multimodal image analysis."""

    def __init__(self, multimodal_ai: MultimodalVertexGenAI):
        self.multimodal_ai = multimodal_ai
        self.chunk_size = 500
        self.chunk_overlap = 100
        self.context_window = 300
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )
        
        self.image_matcher = MultimodalImageMatcher(multimodal_ai)
        
        self.section_patterns = [
            r'(?:Section|SECTION|Chapter|CHAPTER)\s+(\d+(?:\.\d+)*)[:\s]+(.*?)(?=\n|$)',
            r'^(\d+(?:\.\d+)*)\s+(.*?)$',
            r'^([A-Z][A-Z\s]{2,20})$',
            r'^(Abstract|Introduction|Methodology|Methods|Results|Discussion|Conclusion|References)$',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):?\s*$'
        ]

    def extract_full_page_images_multimodal(self, file_path: str) -> Dict[int, PageImageData]:
        """Extract full page images with multimodal analysis."""
        page_images = {}
        
        try:
            logger.info("Extracting full page images with multimodal analysis...")
            pdf_images = convert_from_path(file_path, dpi=200)
            
            for page_num, page_image in enumerate(pdf_images, 1):
                try:
                    # Convert to base64
                    buffered = BytesIO()
                    page_image.save(buffered, format="PNG", quality=85)
                    page_base64 = base64.b64encode(buffered.getvalue()).decode()
                    
                    # Get multimodal summary of the page
                    page_summary = self._get_page_summary_multimodal(page_image)
                    
                    page_data = PageImageData(
                        page_number=page_num,
                        page_image=page_image,
                        base64_string=page_base64,
                        multimodal_summary=page_summary,
                        confidence=0.9
                    )
                    
                    page_images[page_num] = page_data
                    
                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {e}")
                    continue
            
            logger.info(f"Extracted {len(page_images)} full page images with multimodal analysis")
            return page_images
            
        except Exception as e:
            logger.error(f"Error extracting full page images: {e}")
            return {}

    def _get_page_summary_multimodal(self, page_image: Image.Image) -> str:
        """Get a summary of page content using multimodal analysis."""
        try:
            analysis_result = self.multimodal_ai.analyze_image_directly(
                page_image, 
                "Provide a brief summary of this page's main content and key information"
            )
            
            summary_parts = []
            
            if analysis_result.get("image_type"):
                summary_parts.append(f"Type: {analysis_result['image_type']}")
            
            if analysis_result.get("key_insights"):
                summary_parts.append(f"Key Info: {analysis_result['key_insights'][:100]}")
            
            if analysis_result.get("data_values"):
                data_sample = analysis_result['data_values'][:3]
                summary_parts.append(f"Data: {', '.join(data_sample)}")
            
            return " | ".join(summary_parts) if summary_parts else "Page content analysis"
            
        except Exception as e:
            logger.error(f"Error getting page summary: {e}")
            return "Page summary unavailable"

    def extract_region_images_multimodal(self, file_path: str) -> List[MultimodalImageData]:
        """Extract image regions with multimodal analysis."""
        images = []
        
        try:
            pdf_text = self._extract_raw_text(file_path)
            pdf_images = convert_from_path(file_path, dpi=200)
            
            for page_num, page_image in enumerate(pdf_images, 1):
                page_context = self._get_page_context(pdf_text, page_num)
                page_images = self._extract_page_regions_multimodal(page_image, page_num, page_context)
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
        """Get text context around a page."""
        context_pages = []
        
        for p in range(max(1, page_num - 1), min(len(pdf_text) + 1, page_num + 2)):
            if p in pdf_text:
                context_pages.append(pdf_text[p][:600])
        
        return " ".join(context_pages)

    def _extract_page_regions_multimodal(self, page_image: Image.Image, page_num: int, 
                                       page_context: str) -> List[MultimodalImageData]:
        """Extract image regions with multimodal analysis."""
        images = []
        
        try:
            opencv_image = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            
            # Enhanced preprocessing
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges1 = cv2.Canny(blurred, 30, 80)
            edges2 = cv2.Canny(blurred, 50, 150)
            edges = cv2.bitwise_or(edges1, edges2)
            
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours
            min_area = 4000
            max_area = (page_image.width * page_image.height) * 0.8
            
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area <= area <= max_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    if 0.1 <= aspect_ratio <= 10 and y > 50:
                        valid_contours.append((area, contour))
            
            valid_contours.sort(reverse=True)
            
            processed_count = 0
            for area, contour in valid_contours[:3]:  # Top 3 regions
                try:
                    image_data = self._process_region_multimodal(
                        page_image, contour, page_num, page_context
                    )
                    
                    if image_data and image_data.confidence > 0.6:  # Higher threshold for multimodal
                        # Get embedding for multimodal content
                        content_for_embedding = f"{image_data.extracted_text} {image_data.detailed_content} {image_data.key_insights}"
                        if content_for_embedding.strip():
                            try:
                                emb_response = self.multimodal_ai.get_embeddings([content_for_embedding])
                                if emb_response and len(emb_response) > 0:
                                    image_data.embedding = emb_response[0].values
                            except Exception as e:
                                logger.warning(f"Could not get embedding for multimodal image: {e}")
                        
                        images.append(image_data)
                        processed_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error processing region with multimodal: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error extracting page regions: {e}")
            
        return images

    def _process_region_multimodal(self, page_image: Image.Image, contour, 
                                 page_num: int, page_context: str) -> Optional[MultimodalImageData]:
        """Process image region using multimodal analysis."""
        try:
            x, y, w, h = cv2.boundingRect(contour)
            
            aspect_ratio = w / h
            if aspect_ratio > 15 or aspect_ratio < 0.05:
                return None
            
            # Extract region
            roi = page_image.crop((x, y, x + w, y + h))
            
            # Use multimodal analysis instead of OCR
            image_data = self.image_matcher.analyze_image_with_context(roi, page_context)
            
            # Set page number and bbox
            image_data.page_number = page_num
            image_data.bbox = (x, y, x + w, y + h)
            
            return image_data
            
        except Exception as e:
            logger.error(f"Error in multimodal region processing: {e}")
            return None

    def extract_text_from_pdf(self, file_path: str) -> MultimodalDocument:
        """Extract text with multimodal image processing."""
        doc = MultimodalDocument(filename=os.path.basename(file_path))
        
        try:
            # Extract full page images with multimodal analysis
            doc.page_images = self.extract_full_page_images_multimodal(file_path)
            
            # Extract region images with multimodal analysis
            doc.images = self.extract_region_images_multimodal(file_path)
            
            # Build mappings
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
                    
                    # Add multimodal content from images
                    if page_num in doc.page_to_images:
                        multimodal_texts = []
                        for img_idx in doc.page_to_images[page_num]:
                            img = doc.images[img_idx]
                            if img.confidence > 0.7:  # High confidence multimodal content
                                multimodal_content = f"[{img.image_type.title()} - Multimodal]: {img.detailed_content}"
                                if img.extracted_text:
                                    multimodal_content += f" Text: {img.extracted_text}"
                                if img.key_insights:
                                    multimodal_content += f" Insights: {img.key_insights}"
                                multimodal_texts.append(multimodal_content)
                        
                        if multimodal_texts:
                            page_text += f"\n\n=== Multimodal Visual Analysis - Page {page_num} ===\n" + "\n".join(multimodal_texts)
                    
                    if page_text.strip():
                        doc.pages[page_num] = page_text
                        full_text += page_text + "\n\n"
                        char_to_page.extend([page_num] * len(page_text + "\n\n"))
            
            doc.content = full_text
            doc.char_to_page_map = char_to_page
            
            return doc
            
        except Exception as e:
            logger.error(f"Error in multimodal text extraction: {e}")
            raise

    def _clean_pdf_text(self, text: str) -> str:
        """Clean extracted PDF text."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s*\n\s*', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'([a-z])-\s*\n\s*([a-z])', r'\1\2', text)
        text = re.sub(r'([a-z])- ?([a-z])', r'\1\2', text)
        text = re.sub(r'\b(\w)\s+(\w)\b', r'\1\2', text)
        text = re.sub(r'(\w)\s*\.\s*(\w)', r'\1.\2', text)
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
        text = re.sub(r'(\d)\s*,\s*(\d)', r'\1,\2', text)
        return text.strip()

    def chunk_document_multimodal(self, doc: MultimodalDocument) -> MultimodalDocument:
        """Chunk document with multimodal context integration."""
        if not doc.content:
            return doc
        
        try:
            raw_chunks = self.text_splitter.create_documents([doc.content])
            
            for i, chunk in enumerate(raw_chunks):
                chunk_text = chunk.page_content
                
                start_pos = doc.content.find(chunk_text)
                if start_pos == -1:
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
                
                # Find pages
                chunk_pages = set()
                for pos in range(max(0, start_pos), min(end_pos + 1, len(doc.char_to_page_map))):
                    if pos < len(doc.char_to_page_map):
                        chunk_pages.add(doc.char_to_page_map[pos])
                
                if not chunk_pages:
                    chunk_pages = {1}
                
                # Calculate page coverage
                page_coverage = len(chunk_text) / max([len(doc.pages.get(p, "")) for p in chunk_pages], 1)
                
                # Multimodal image matching
                relevant_images = self._find_relevant_images_multimodal(
                    chunk_text, list(chunk_pages), doc.images, doc.page_to_images
                )
                
                # Build multimodal context from relevant images
                multimodal_context = self._build_multimodal_context(relevant_images, doc.images)
                
                # Enhanced section extraction
                section_info = self._extract_section_info_enhanced(chunk_text)
                
                # Create enhanced chunk
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
                    page_coverage=page_coverage,
                    multimodal_context=multimodal_context
                )
                
                # Get embedding including multimodal context
                embedding_text = f"{chunk_text} {multimodal_context}"
                try:
                    emb_response = self.multimodal_ai.get_embeddings([embedding_text])
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
            
            logger.info(f"Created {len(doc.chunks)} multimodal-enhanced chunks")
            return doc
            
        except Exception as e:
            logger.error(f"Error in multimodal chunking: {e}")
            return doc

    def _find_relevant_images_multimodal(self, chunk_text: str, chunk_pages: List[int], 
                                       images: List[MultimodalImageData], 
                                       page_to_images: Dict[int, List[int]]) -> List[int]:
        """Find relevant images using multimodal similarity."""
        relevant_images = []
        
        try:
            # Get candidate images
            candidate_images = []
            for page in chunk_pages:
                if page in page_to_images:
                    candidate_images.extend(page_to_images[page])
                for adj_page in [page - 1, page + 1]:
                    if adj_page in page_to_images:
                        candidate_images.extend(page_to_images[adj_page])
            
            candidate_images = list(set(candidate_images))
            
            # Score using multimodal similarity
            image_scores = []
            for img_idx in candidate_images:
                if img_idx < len(images):
                    img = images[img_idx]
                    
                    # Use multimodal similarity calculation
                    relevance_score = self.image_matcher.calculate_multimodal_similarity(chunk_text, img)
                    
                    # Enhanced factors
                    page_proximity = 1.0 if img.page_number in chunk_pages else 0.7
                    confidence_factor = img.confidence
                    
                    final_score = relevance_score * page_proximity * confidence_factor
                    
                    # Higher threshold for multimodal accuracy
                    if final_score > 0.5:
                        image_scores.append((img_idx, final_score))
            
            # Sort and select top images
            image_scores.sort(key=lambda x: x[1], reverse=True)
            relevant_images = [img_idx for img_idx, score in image_scores[:3]]
            
            logger.debug(f"Found {len(relevant_images)} multimodal-relevant images for chunk")
            
        except Exception as e:
            logger.error(f"Error in multimodal image matching: {e}")
        
        return relevant_images

    def _build_multimodal_context(self, image_refs: List[int], images: List[MultimodalImageData]) -> str:
        """Build multimodal context from relevant images."""
        context_parts = []
        
        try:
            for img_idx in image_refs:
                if img_idx < len(images):
                    img = images[img_idx]
                    
                    context_part = f"Visual {img.image_type}: {img.detailed_content}"
                    if img.key_insights:
                        context_part += f" Key insights: {img.key_insights}"
                    if img.data_values:
                        context_part += f" Data: {', '.join(img.data_values[:5])}"
                    
                    context_parts.append(context_part)
            
            return " | ".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error building multimodal context: {e}")
            return ""

    def _extract_section_info_enhanced(self, text: str) -> Dict[str, str]:
        """Enhanced section extraction."""
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
        
        except Exception as e:
            logger.error(f"Error in enhanced section extraction: {e}")
        
        return section_info

    def process_pdf(self, file_path: str) -> MultimodalDocument:
        """Process PDF with multimodal capabilities."""
        doc = self.extract_text_from_pdf(file_path)
        return self.chunk_document_multimodal(doc)

# Continue with the rest of the system using the same pattern...
# (I'll continue with the remaining classes in the next part due to length)

class VertexAIEmbeddings(Embeddings):
    """VertexAI embeddings for multimodal system."""
    
    def __init__(self, multimodal_ai: MultimodalVertexGenAI, model_name: str = "text-embedding-004"):
        self.multimodal_ai = multimodal_ai
        self.model_name = model_name
        self._embedding_dimension = None
        super().__init__()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with preprocessing."""
        if not texts:
            return []
        
        try:
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            logger.info(f"Embedding {len(processed_texts)} documents with VertexAI")
            embeddings_response = self.multimodal_ai.get_embeddings(processed_texts, self.model_name)
            
            embeddings = []
            for embedding in embeddings_response:
                if hasattr(embedding, 'values'):
                    vector = np.array(embedding.values, dtype=np.float32)
                    vector = vector / (np.linalg.norm(vector) + 1e-12)
                    embeddings.append(vector.tolist())
                else:
                    embeddings.append(self._get_zero_vector())
            
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
            embeddings_response = self.multimodal_ai.get_embeddings([processed_text], self.model_name)
            
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
        """Preprocess text for embeddings."""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        if len(text) > 7500:
            truncated = text[:7500]
            last_period = truncated.rfind('.')
            if last_period > 6000:
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

# Enhanced MMR Reranker with multimodal support
class MultimodalMMRReranker:
    """MMR reranker enhanced with multimodal understanding."""
    
    def __init__(self, multimodal_ai: MultimodalVertexGenAI):
        self.multimodal_ai = multimodal_ai
        self.lambda_diversity = 0.6
        self.relevance_weight = 0.7
        self.diversity_weight = 0.3
        
    def multimodal_mmr_rerank(self, query: str, documents: List[Document], k: int = 5) -> List[Tuple[Document, float]]:
        """Enhanced MMR reranking with multimodal context consideration."""
        try:
            if not documents or k <= 0:
                return []
            
            logger.info(f"Multimodal MMR reranking {len(documents)} documents")
            
            # Calculate enhanced relevance scores including multimodal context
            relevance_scores = []
            for doc in documents:
                relevance = self._calculate_enhanced_relevance(query, doc)
                relevance_scores.append(relevance)
            
            # MMR selection with multimodal awareness
            selected_docs = []
            selected_indices = set()
            remaining_indices = list(range(len(documents)))
            
            for selection_round in range(min(k, len(documents))):
                if not remaining_indices:
                    break
                
                best_score = -float('inf')
                best_idx = -1
                
                for idx in remaining_indices:
                    current_doc = documents[idx]
                    current_relevance = relevance_scores[idx]
                    
                    # Calculate multimodal-aware diversity
                    max_similarity = 0.0
                    if selected_indices:
                        similarities = []
                        for selected_idx in selected_indices:
                            selected_doc = documents[selected_idx]
                            similarity = self._calculate_multimodal_similarity(current_doc, selected_doc)
                            similarities.append(similarity)
                        
                        max_similarity = max(similarities) if similarities else 0.0
                    
                    # Enhanced MMR score
                    mmr_score = (
                        self.relevance_weight * current_relevance - 
                        self.diversity_weight * max_similarity
                    )
                    
                    # Multimodal content bonus
                    if current_doc.metadata.get('multimodal_context'):
                        mmr_score *= 1.1
                    
                    if selection_round == 0:
                        mmr_score = current_relevance
                    
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = idx
                
                if best_idx != -1:
                    selected_doc = documents[best_idx]
                    final_score = relevance_scores[best_idx]
                    
                    selected_doc.metadata['multimodal_mmr_score'] = best_score
                    selected_doc.metadata['multimodal_relevance'] = final_score
                    
                    selected_docs.append((selected_doc, final_score))
                    selected_indices.add(best_idx)
                    remaining_indices.remove(best_idx)
            
            selected_docs.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Multimodal MMR selected {len(selected_docs)} documents")
            return selected_docs
            
        except Exception as e:
            logger.error(f"Error in multimodal MMR reranking: {e}")
            return [(doc, 0.5) for doc in documents[:k]]
    
    def _calculate_enhanced_relevance(self, query: str, document: Document) -> float:
        """Calculate enhanced relevance including multimodal context."""
        try:
            # Base text relevance
            doc_text = document.metadata.get('original_text', document.page_content)
            
            # Get text embedding similarity
            embeddings = self.multimodal_ai.get_embeddings([query, doc_text])
            
            text_similarity = 0.5
            if len(embeddings) == 2:
                query_emb = np.array(embeddings[0].values)
                doc_emb = np.array(embeddings[1].values)
                
                query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-12)
                doc_emb = doc_emb / (np.linalg.norm(doc_emb) + 1e-12)
                
                text_similarity = np.dot(query_emb, doc_emb)
            
            # Multimodal context bonus
            multimodal_bonus = 0.0
            multimodal_context = document.metadata.get('multimodal_context', '')
            if multimodal_context:
                # Check if multimodal content is relevant to query
                query_words = set(query.lower().split())
                context_words = set(multimodal_context.lower().split())
                
                if query_words & context_words:
                    multimodal_bonus = 0.2  # Significant bonus for relevant multimodal content
            
            # Other relevance factors
            additional_factors = self._calculate_additional_factors(query, document)
            
            final_relevance = text_similarity + multimodal_bonus + additional_factors
            return min(final_relevance, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating enhanced relevance: {e}")
            return 0.5
    
    def _calculate_multimodal_similarity(self, doc1: Document, doc2: Document) -> float:
        """Calculate similarity between documents considering multimodal content."""
        try:
            # Text similarity
            text1 = doc1.metadata.get('original_text', doc1.page_content)
            text2 = doc2.metadata.get('original_text', doc2.page_content)
            
            embeddings = self.multimodal_ai.get_embeddings([text1, text2])
            
            text_sim = 0.5
            if len(embeddings) == 2:
                emb1 = np.array(embeddings[0].values)
                emb2 = np.array(embeddings[1].values)
                
                emb1 = emb1 / (np.linalg.norm(emb1) + 1e-12)
                emb2 = emb2 / (np.linalg.norm(emb2) + 1e-12)
                
                text_sim = np.dot(emb1, emb2)
            
            # Multimodal content similarity
            multimodal_sim = 0.0
            context1 = doc1.metadata.get('multimodal_context', '')
            context2 = doc2.metadata.get('multimodal_context', '')
            
            if context1 and context2:
                words1 = set(context1.lower().split())
                words2 = set(context2.lower().split())
                
                if words1 and words2:
                    intersection = words1 & words2
                    union = words1 | words2
                    multimodal_sim = len(intersection) / len(union) if union else 0.0
            
            # Weighted combination
            final_similarity = 0.7 * text_sim + 0.3 * multimodal_sim
            return final_similarity
            
        except Exception as e:
            logger.error(f"Error calculating multimodal similarity: {e}")
            return 0.5
    
    def _calculate_additional_factors(self, query: str, document: Document) -> float:
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
                        score += 0.15
                        break
            
            # Semantic density bonus
            semantic_density = metadata.get('semantic_density', 0)
            if semantic_density > 15:
                score += 0.1
            
            # Image reference bonus for visual queries
            image_refs = metadata.get('image_refs', [])
            if image_refs and any(keyword in query.lower() for keyword in ['chart', 'graph', 'table', 'figure', 'image', 'visual']):
                score += 0.15
            
            # Page coverage bonus
            page_coverage = metadata.get('page_coverage', 0)
            if page_coverage > 0.3:
                score += 0.1
            
            return min(score, 0.5)  # Cap additional factors
            
        except Exception as e:
            logger.error(f"Error calculating additional factors: {e}")
            return 0.0

# Continue with the main system classes...

def main():
    """Main function with multimodal system."""
    print("🎯 Starting Ultra-Accurate Multimodal RAG System")
    
    try:
        import google.colab
        print("📱 Running in Google Colab")
        from google.colab import auth
        auth.authenticate_user()
        print("✅ Authentication completed")
    except ImportError:
        print("💻 Running in local environment")
    
    print("\n🚀 **Multimodal Features:**")
    print("- 🎯 Direct multimodal image analysis (no OCR dependency)")
    print("- 🔍 Enhanced visual understanding using multimodal LLM")
    print("- 📊 Precise chart, table, diagram analysis")
    print("- 📄 Full PDF page images with multimodal summaries")
    print("- 🔗 Multimodal-aware MMR reranking")
    print("- 📈 Advanced relevance scoring with visual context")
    print("- ⚡ Maximum accuracy through direct image understanding")
    
    print("\n📋 **Usage Instructions:**")
    print("1. Upload PDF documents with visual content")
    print("2. Ask questions about charts, tables, diagrams, or figures")
    print("3. Get precise answers with actual image content understanding")
    print("4. View full page images when their content is referenced")
    
    # Launch interface would go here...

if __name__ == "__main__":
    main()
