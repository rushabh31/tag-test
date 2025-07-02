# -*- coding: utf-8 -*-
"""Enhanced-RAG-with-HNSW-and-Research-References.ipynb

This notebook enhances the RAG system with:
1. HNSW (Hierarchical Navigable Small World) index for fast ANN search
2. Cosine similarity for embeddings
3. Research paper-style references with numbered citations
4. Enhanced image analysis and explanation
5. Simplified UI with default configurations
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import pypdf
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import cv2

# HNSW for fast similarity search
import hnswlib

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
class ImageData:
    """Store extracted image data with enhanced analysis."""
    image: Image.Image
    page_number: int
    bbox: Optional[Tuple[int, int, int, int]] = None
    ocr_text: str = ""
    base64_string: str = ""
    image_type: str = ""  # chart, diagram, figure, table, etc.
    analysis: str = ""    # AI-generated analysis of the image
    confidence: float = 0.0

@dataclass
class Reference:
    """Store reference information for research-style citations."""
    ref_id: str
    document: str
    page: int
    section: str = ""
    content_snippet: str = ""
    image_ref: bool = False
    image_description: str = ""

@dataclass
class PageChunk:
    """Enhanced chunk with reference tracking."""
    text: str
    page_numbers: List[int]
    start_char_idx: int
    end_char_idx: int
    filename: str
    section_info: Dict[str, str] = field(default_factory=dict)
    image_refs: List[int] = field(default_factory=list)
    chunk_hash: str = ""

    def __post_init__(self):
        # Generate unique hash for this chunk
        content = f"{self.text}{self.filename}{self.page_numbers}"
        self.chunk_hash = hashlib.md5(content.encode()).hexdigest()[:8]

    def to_document(self) -> Document:
        """Convert to LangChain Document."""
        return Document(
            page_content=self.text,
            metadata={
                "page_numbers": self.page_numbers,
                "source": self.filename,
                "start_idx": self.start_char_idx,
                "end_idx": self.end_char_idx,
                "section_info": self.section_info,
                "image_refs": self.image_refs,
                "chunk_hash": self.chunk_hash
            }
        )

@dataclass
class PDFDocument:
    """Enhanced PDF document with reference system."""
    filename: str
    content: str = ""
    pages: Dict[int, str] = field(default_factory=dict)
    char_to_page_map: List[int] = field(default_factory=list)
    chunks: List[PageChunk] = field(default_factory=list)
    images: List[ImageData] = field(default_factory=list)
    image_to_page_map: Dict[int, int] = field(default_factory=dict)
    references: Dict[str, Reference] = field(default_factory=dict)

    @property
    def langchain_documents(self) -> List[Document]:
        """Convert chunks to LangChain Documents."""
        return [chunk.to_document() for chunk in self.chunks]

class EnhancedPDFProcessor:
    """Advanced PDF processor with image analysis."""

    def __init__(self):
        # Default optimized settings
        self.chunk_size = 800
        self.chunk_overlap = 200
        self.enable_ocr = True

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=False
        )

        # Enhanced patterns for better section detection
        self.section_patterns = [
            r'(?:Section|SECTION|Chapter|CHAPTER)\s+(\d+(?:\.\d+)*)[:\s]+(.*?)(?=\n|$)',
            r'(\d+(?:\.\d+)*)\s+(.*?)(?=\n|$)',
            r'(?:\n|\A)([A-Z][A-Z\s]{2,})(?:\n|:)',
            r'(?:\n|\A)(Abstract|Introduction|Methodology|Results|Discussion|Conclusion)(?:\n|:)'
        ]

    def analyze_image(self, image: Image.Image, ocr_text: str) -> Tuple[str, str, float]:
        """Analyze image to determine type and generate description."""
        try:
            # Convert to numpy array for analysis
            img_array = np.array(image)
            
            # Basic image analysis
            height, width = img_array.shape[:2]
            aspect_ratio = width / height
            
            # Analyze content based on OCR and visual features
            image_type = "figure"
            analysis = ""
            confidence = 0.5
            
            # Determine image type based on OCR content
            if ocr_text:
                text_lower = ocr_text.lower()
                if any(word in text_lower for word in ['table', 'column', 'row']):
                    image_type = "table"
                    confidence = 0.8
                elif any(word in text_lower for word in ['chart', 'graph', 'plot', '%']):
                    image_type = "chart"
                    confidence = 0.9
                elif any(word in text_lower for word in ['diagram', 'flow', 'process', 'step']):
                    image_type = "diagram"
                    confidence = 0.8
                elif any(word in text_lower for word in ['figure', 'fig', 'image']):
                    image_type = "figure"
                    confidence = 0.7
            
            # Generate analysis based on type
            if image_type == "chart":
                analysis = f"This appears to be a {image_type} with data visualization elements. "
                if ocr_text:
                    analysis += f"The chart contains the following text elements: {ocr_text[:200]}..."
            elif image_type == "table":
                analysis = f"This is a {image_type} containing structured data. "
                if ocr_text:
                    analysis += f"Table content includes: {ocr_text[:200]}..."
            elif image_type == "diagram":
                analysis = f"This is a {image_type} showing relationships or processes. "
                if ocr_text:
                    analysis += f"Diagram elements include: {ocr_text[:200]}..."
            else:
                analysis = f"This is a {image_type}. "
                if ocr_text:
                    analysis += f"Image contains text: {ocr_text[:200]}..."
            
            analysis += f" Image dimensions: {width}x{height} pixels, aspect ratio: {aspect_ratio:.2f}."
            
            return image_type, analysis, confidence
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return "figure", f"Image analysis failed: {str(e)}", 0.1

    def extract_images_from_pdf(self, file_path: str) -> List[ImageData]:
        """Extract and analyze images from PDF."""
        images = []
        
        try:
            pdf_images = convert_from_path(file_path, dpi=300)
            
            for page_num, page_image in enumerate(pdf_images, 1):
                opencv_image = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
                
                # Enhanced edge detection
                edges = cv2.Canny(gray, 30, 100)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                min_area = 8000
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > min_area:
                        x, y, w, h = cv2.boundingRect(contour)
                        roi = page_image.crop((x, y, x + w, y + h))
                        
                        # OCR
                        ocr_text = ""
                        if self.enable_ocr:
                            try:
                                ocr_text = pytesseract.image_to_string(roi, config='--psm 6')
                            except Exception as e:
                                logger.warning(f"OCR failed for image on page {page_num}: {str(e)}")
                        
                        # Analyze image
                        image_type, analysis, confidence = self.analyze_image(roi, ocr_text)
                        
                        # Convert to base64
                        buffered = BytesIO()
                        roi.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()
                        
                        image_data = ImageData(
                            image=roi,
                            page_number=page_num,
                            bbox=(x, y, x + w, y + h),
                            ocr_text=ocr_text,
                            base64_string=img_base64,
                            image_type=image_type,
                            analysis=analysis,
                            confidence=confidence
                        )
                        images.append(image_data)
                
                # Full page backup if no regions found
                if not any(img.page_number == page_num for img in images):
                    ocr_text = ""
                    if self.enable_ocr:
                        try:
                            ocr_text = pytesseract.image_to_string(page_image, config='--psm 3')
                        except Exception as e:
                            logger.warning(f"OCR failed for page {page_num}: {str(e)}")
                    
                    image_type, analysis, confidence = self.analyze_image(page_image, ocr_text)
                    
                    buffered = BytesIO()
                    page_image.save(buffered, format="PNG")
                    full_page_base64 = base64.b64encode(buffered.getvalue()).decode()
                    
                    full_page_data = ImageData(
                        image=page_image,
                        page_number=page_num,
                        ocr_text=ocr_text,
                        base64_string=full_page_base64,
                        image_type=image_type,
                        analysis=analysis,
                        confidence=confidence
                    )
                    images.append(full_page_data)
                    
        except Exception as e:
            logger.error(f"Error extracting images from PDF: {str(e)}")
            
        return images

    def extract_text_from_pdf(self, file_path: str) -> PDFDocument:
        """Extract text from PDF with reference generation."""
        doc = PDFDocument(filename=os.path.basename(file_path))
        full_text = ""
        char_to_page = []

        try:
            doc.images = self.extract_images_from_pdf(file_path)
            
            for idx, img in enumerate(doc.images):
                doc.image_to_page_map[idx] = img.page_number
            
            with open(file_path, "rb") as file:
                pdf = pypdf.PdfReader(file)

                if len(pdf.pages) == 0:
                    logger.warning(f"PDF {file_path} has no pages.")
                    return doc

                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    page_text = page.extract_text() or ""
                    page_text = self._clean_pdf_text(page_text)
                    
                    # Add OCR text from images
                    ocr_texts = [img.ocr_text for img in doc.images if img.page_number == page_num]
                    if ocr_texts:
                        page_text += f"\n\n[OCR Content from Page {page_num}]:\n" + "\n".join(ocr_texts)

                    if page_text.strip():
                        doc.pages[page_num] = page_text
                        full_text += page_text
                        char_to_page.extend([page_num] * len(page_text))

            doc.content = full_text
            doc.char_to_page_map = char_to_page
            
            # Generate references
            self._generate_references(doc)
            
            return doc

        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            raise

    def _generate_references(self, doc: PDFDocument):
        """Generate reference entries for the document."""
        ref_counter = 1
        
        # Create references for each page
        for page_num, page_content in doc.pages.items():
            # Extract section if available
            section = self._extract_main_section(page_content)
            
            ref_id = f"ref{ref_counter:03d}"
            reference = Reference(
                ref_id=ref_id,
                document=doc.filename,
                page=page_num,
                section=section,
                content_snippet=page_content[:200] + "..." if len(page_content) > 200 else page_content
            )
            doc.references[ref_id] = reference
            ref_counter += 1
        
        # Create references for images
        for idx, img in enumerate(doc.images):
            if img.confidence > 0.5:  # Only high-confidence images
                ref_id = f"img{idx+1:03d}"
                reference = Reference(
                    ref_id=ref_id,
                    document=doc.filename,
                    page=img.page_number,
                    section="",
                    content_snippet=img.ocr_text[:100] + "..." if img.ocr_text else "",
                    image_ref=True,
                    image_description=img.analysis
                )
                doc.references[ref_id] = reference

    def _extract_main_section(self, text: str) -> str:
        """Extract the main section heading from text."""
        for pattern in self.section_patterns:
            match = re.search(pattern, text)
            if match:
                if len(match.groups()) >= 2:
                    return f"{match.group(1)} {match.group(2)}"
                else:
                    return match.group(1)
        return ""

    def _clean_pdf_text(self, text: str) -> str:
        """Clean extracted PDF text."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s*\n\s*', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'([a-z])- ?([a-z])', r'\1\2', text)
        return text

    def chunk_document(self, doc: PDFDocument) -> PDFDocument:
        """Chunk document with enhanced reference tracking."""
        if not doc.content:
            return doc

        try:
            raw_chunks = self.text_splitter.create_documents([doc.content])
        except Exception as e:
            logger.error(f"Error creating chunks: {str(e)}")
            return doc

        for i, chunk in enumerate(raw_chunks):
            chunk_text = chunk.page_content
            start_pos = doc.content.find(chunk_text)
            
            if start_pos == -1:
                start_pos = 0
                end_pos = min(len(chunk_text), len(doc.content)) - 1
            else:
                end_pos = start_pos + len(chunk_text) - 1

            # Find pages
            chunk_pages = set()
            for pos in range(start_pos, min(end_pos + 1, len(doc.char_to_page_map))):
                if pos < len(doc.char_to_page_map):
                    chunk_pages.add(doc.char_to_page_map[pos])

            if not chunk_pages:
                chunk_pages = {1}

            # Find related images with better matching
            image_refs = []
            for idx, img in enumerate(doc.images):
                if img.page_number in chunk_pages:
                    # Check OCR text overlap
                    if img.ocr_text:
                        ocr_words = set(img.ocr_text.lower().split())
                        chunk_words = set(chunk_text.lower().split())
                        overlap = len(ocr_words.intersection(chunk_words))
                        if overlap > 2 or len(ocr_words) < 5:
                            image_refs.append(idx)
                    # Check for figure references
                    elif any(keyword in chunk_text.lower() for keyword in 
                            ["figure", "fig.", "chart", "graph", "table", "diagram"]):
                        image_refs.append(idx)

            section_info = self._extract_section_info(chunk_text)

            page_chunk = PageChunk(
                text=chunk_text,
                page_numbers=sorted(list(chunk_pages)),
                start_char_idx=start_pos,
                end_char_idx=end_pos,
                filename=doc.filename,
                section_info=section_info,
                image_refs=image_refs
            )

            doc.chunks.append(page_chunk)

        return doc

    def _extract_section_info(self, text: str) -> Dict[str, str]:
        """Extract section information from text."""
        section_info = {}
        for pattern in self.section_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if len(match.groups()) >= 2:
                    section_number = match.group(1)
                    section_title = match.group(2).strip()
                    section_info[section_number] = section_title
                elif len(match.groups()) == 1:
                    section_title = match.group(1).strip()
                    section_info[f"heading_{len(section_info)}"] = section_title
        return section_info

    def process_pdf(self, file_path: str) -> PDFDocument:
        """Process PDF file."""
        doc = self.extract_text_from_pdf(file_path)
        return self.chunk_document(doc)

class VertexAIEmbeddings(Embeddings):
    """VertexAI embeddings with cosine similarity optimization."""
    
    def __init__(self, vertex_gen_ai: VertexGenAI, model_name: str = "text-embedding-004"):
        self.vertex_gen_ai = vertex_gen_ai
        self.model_name = model_name
        self._embedding_dimension = None
        super().__init__()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with L2 normalization for cosine similarity."""
        if not texts:
            return []
            
        try:
            logger.info(f"Embedding {len(texts)} documents with VertexAI")
            embeddings_response = self.vertex_gen_ai.get_embeddings(texts, self.model_name)
            
            embeddings = []
            for embedding in embeddings_response:
                if hasattr(embedding, 'values'):
                    vector = np.array(embedding.values, dtype=np.float32)
                    # L2 normalize for cosine similarity
                    vector = vector / np.linalg.norm(vector)
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
        """Embed query with L2 normalization."""
        if not text:
            return self._get_zero_vector()
            
        try:
            logger.info(f"Embedding query with VertexAI: {text[:50]}...")
            embeddings_response = self.vertex_gen_ai.get_embeddings([text], self.model_name)
            
            if embeddings_response and len(embeddings_response) > 0:
                embedding = embeddings_response[0]
                if hasattr(embedding, 'values'):
                    vector = np.array(embedding.values, dtype=np.float32)
                    # L2 normalize for cosine similarity
                    vector = vector / np.linalg.norm(vector)
                    
                    if self._embedding_dimension is None:
                        self._embedding_dimension = len(vector)
                    return vector.tolist()
                    
            return self._get_zero_vector()
            
        except Exception as e:
            logger.error(f"Error getting query embedding: {str(e)}")
            return self._get_zero_vector()
    
    def _get_zero_vector(self) -> List[float]:
        """Get normalized zero vector."""
        dim = self._embedding_dimension if self._embedding_dimension else 768
        return [0.0] * dim

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)

class HNSWRetriever:
    """HNSW-based retriever with cosine similarity."""

    def __init__(self, documents: List[Document] = None, vertex_gen_ai: VertexGenAI = None):
        logger.info("Initializing HNSWRetriever...")
        
        self.vertex_gen_ai = vertex_gen_ai
        self.embeddings = VertexAIEmbeddings(vertex_gen_ai) if vertex_gen_ai else None
        self.documents = []
        self.document_embeddings = []
        self.hnsw_index = None
        self.dimension = None
        
        if documents and len(documents) > 0:
            self.add_documents(documents)

    def add_documents(self, documents: List[Document]):
        """Add documents to HNSW index."""
        try:
            logger.info(f"Adding {len(documents)} documents to HNSW index")
            
            # Get embeddings
            texts = [doc.page_content for doc in documents]
            embeddings = self.embeddings.embed_documents(texts)
            
            if not embeddings:
                logger.error("No embeddings generated")
                return
            
            # Initialize HNSW index if needed
            if self.hnsw_index is None:
                self.dimension = len(embeddings[0])
                self.hnsw_index = hnswlib.Index(space='cosine', dim=self.dimension)
                self.hnsw_index.init_index(max_elements=10000, ef_construction=200, M=16)
                logger.info(f"Initialized HNSW index with dimension {self.dimension}")
            
            # Add to index
            start_idx = len(self.documents)
            ids = list(range(start_idx, start_idx + len(documents)))
            
            embeddings_array = np.array(embeddings, dtype=np.float32)
            self.hnsw_index.add_items(embeddings_array, ids)
            
            # Store documents and embeddings
            self.documents.extend(documents)
            self.document_embeddings.extend(embeddings)
            
            # Set ef parameter for search
            self.hnsw_index.set_ef(50)
            
            logger.info(f"Successfully added {len(documents)} documents to HNSW index")
            
        except Exception as e:
            logger.error(f"Error adding documents to HNSW index: {str(e)}")
            raise

    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """Get relevant documents using HNSW cosine similarity search."""
        try:
            if not self.hnsw_index or not self.documents:
                logger.warning("No documents in HNSW index")
                return []
            
            logger.info(f"Searching HNSW index for query: {query[:50]}...")
            
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            if not query_embedding:
                logger.error("Failed to get query embedding")
                return []
            
            # Search HNSW index
            query_array = np.array([query_embedding], dtype=np.float32)
            labels, distances = self.hnsw_index.knn_query(query_array, k=min(k, len(self.documents)))
            
            # Get documents
            relevant_docs = []
            for label, distance in zip(labels[0], distances[0]):
                if label < len(self.documents):
                    doc = self.documents[label]
                    # Add similarity score to metadata
                    doc.metadata['similarity_score'] = 1 - distance  # Convert distance to similarity
                    relevant_docs.append(doc)
            
            logger.info(f"Retrieved {len(relevant_docs)} documents from HNSW index")
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Error searching HNSW index: {str(e)}")
            return []

class EnhancedRAGSystem:
    """Enhanced RAG system with HNSW and research-style references."""

    def __init__(self, vertex_gen_ai: VertexGenAI = None):
        logger.info("Initializing EnhancedRAGSystem...")
        
        self.vertex_gen_ai = vertex_gen_ai
        self.processor = EnhancedPDFProcessor()
        self.documents = {}
        self.hnsw_retriever = None
        self.conversation_history = []
        self.reference_counter = 1
        
        if vertex_gen_ai:
            self.hnsw_retriever = HNSWRetriever(vertex_gen_ai=vertex_gen_ai)

    def upload_pdf(self, file_path: str) -> str:
        """Process and index PDF with HNSW."""
        try:
            logger.info(f"Processing PDF: {file_path}")
            
            doc = self.processor.process_pdf(file_path)
            self.documents[doc.filename] = doc
            
            # Add to HNSW index
            if self.hnsw_retriever:
                self.hnsw_retriever.add_documents(doc.langchain_documents)
            
            image_count = len(doc.images)
            result = f"✅ Processed {doc.filename}: {len(doc.pages)} pages, {len(doc.chunks)} chunks, {image_count} images"
            logger.info(result)
            return result
            
        except Exception as e:
            error_msg = f"❌ Error processing PDF: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def ask(self, query: str) -> Dict[str, Any]:
        """Process query with research-style response."""
        if not self.hnsw_retriever or not self.documents:
            return {
                "answer": "Please upload documents first.",
                "references": [],
                "images": []
            }

        try:
            logger.info(f"Processing query: {query[:50]}...")
            
            # Get relevant documents
            relevant_docs = self.hnsw_retriever.get_relevant_documents(query, k=8)
            
            if not relevant_docs:
                return {
                    "answer": "No relevant information found for your query.",
                    "references": [],
                    "images": []
                }
            
            # Prepare context with references
            context_parts = []
            references = []
            images = []
            ref_map = {}
            
            for i, doc in enumerate(relevant_docs):
                metadata = doc.metadata
                doc_name = metadata.get('source', 'Unknown')
                pages = metadata.get('page_numbers', [])
                similarity = metadata.get('similarity_score', 0.0)
                
                # Create reference
                ref_id = f"[{self.reference_counter}]"
                reference = {
                    "id": self.reference_counter,
                    "document": doc_name,
                    "pages": pages,
                    "similarity": similarity,
                    "content": doc.page_content[:200] + "..."
                }
                references.append(reference)
                ref_map[i] = ref_id
                self.reference_counter += 1
                
                # Add context with reference
                context_parts.append(f"{ref_id} {doc.page_content}")
                
                # Get images
                image_refs = metadata.get('image_refs', [])
                if image_refs and doc_name in self.documents:
                    pdf_doc = self.documents[doc_name]
                    for img_idx in image_refs:
                        if img_idx < len(pdf_doc.images):
                            img_data = pdf_doc.images[img_idx]
                            images.append({
                                "document": doc_name,
                                "page": img_data.page_number,
                                "type": img_data.image_type,
                                "analysis": img_data.analysis,
                                "confidence": img_data.confidence,
                                "base64": img_data.base64_string,
                               "reference_id": self.reference_counter
                           })
                           
                           # Add image reference
                           img_ref = {
                               "id": self.reference_counter,
                               "document": doc_name,
                               "pages": [img_data.page_number],
                               "similarity": 1.0,
                               "content": f"Image ({img_data.image_type}): {img_data.analysis}",
                               "is_image": True
                           }
                           references.append(img_ref)
                           self.reference_counter += 1
           
           context = "\n\n".join(context_parts)
           
           # Create research-style prompt
           prompt = f"""You are an expert academic researcher writing a comprehensive analysis. Provide a detailed, elaborate response to the user's question using ONLY the provided context. 

CRITICAL REQUIREMENTS:
1. Write in an academic, research paper style with detailed explanations
2. Add numbered reference citations [1], [2], etc. after EVERY factual statement
3. For image-derived information, be extremely detailed in explaining charts, diagrams, tables, and figures
4. Provide thorough analysis and interpretation of visual data
5. Use multiple sentences to elaborate on each point
6. Connect related concepts and provide comprehensive coverage
7. When referencing images, describe what the visual shows and its significance

Context with References:
{context}

User Question: {query}

Instructions for Image Content:
- If information comes from charts/graphs: Describe trends, patterns, values, axes, legends
- If from tables: Explain the data structure, key values, relationships
- If from diagrams: Detail the process, flow, components, connections
- If from figures: Describe visual elements, their meaning, and implications

Write a comprehensive, well-referenced academic response with detailed explanations:"""

           # Generate response
           response = self.vertex_gen_ai.generate_content(prompt)
           
           if not response:
               response = "I couldn't generate a response. Please try rephrasing your question."
           
           # Add conversation to history
           self.conversation_history.append((query, response))
           
           return {
               "answer": response,
               "references": references,
               "images": images
           }
           
       except Exception as e:
           logger.error(f"Error in ask method: {str(e)}")
           return {
               "answer": f"Error processing query: {str(e)}",
               "references": [],
               "images": []
           }

   def reset_conversation(self):
       """Reset conversation history."""
       self.conversation_history = []
       self.reference_counter = 1
       return "Conversation reset successfully."

   def get_document_stats(self) -> Dict[str, Any]:
       """Get enhanced document statistics."""
       if not self.documents:
           return {}
       
       total_stats = {
           "total_documents": len(self.documents),
           "total_pages": 0,
           "total_chunks": 0,
           "total_images": 0,
           "images_by_type": {},
           "documents": {}
       }
       
       for name, doc in self.documents.items():
           # Count image types
           image_types = {}
           for img in doc.images:
               img_type = img.image_type
               image_types[img_type] = image_types.get(img_type, 0) + 1
               total_stats["images_by_type"][img_type] = total_stats["images_by_type"].get(img_type, 0) + 1
           
           doc_stats = {
               "pages": len(doc.pages),
               "chunks": len(doc.chunks),
               "characters": len(doc.content),
               "images": len(doc.images),
               "image_types": image_types,
               "references": len(doc.references)
           }
           
           total_stats["documents"][name] = doc_stats
           total_stats["total_pages"] += doc_stats["pages"]
           total_stats["total_chunks"] += doc_stats["chunks"]
           total_stats["total_images"] += doc_stats["images"]
       
       return total_stats

class SimplifiedRAGInterface:
   """Simplified Gradio interface with default configurations."""

   def __init__(self):
       self.rag_system = None
       self.vertex_gen_ai = None
       logger.info("Initialized SimplifiedRAGInterface")
       self.setup_interface()

   def setup_interface(self):
       """Set up simplified Gradio interface."""
       with gr.Blocks(
           theme=gr.themes.Soft(),
           title="Research Assistant with Advanced RAG",
           css="""
           .reference-box {
               background-color: #f8f9fa;
               border-left: 4px solid #007bff;
               padding: 10px;
               margin: 5px 0;
               border-radius: 5px;
           }
           .image-analysis {
               background-color: #fff3cd;
               border: 1px solid #ffeaa7;
               padding: 10px;
               border-radius: 5px;
               margin: 5px 0;
           }
           """
       ) as self.interface:
           
           gr.Markdown("""
           # 📚 Research Assistant with Advanced RAG
           
           **Features:**
           - 🔍 HNSW-based similarity search with cosine distance
           - 📖 Research paper-style numbered references  
           - 🖼️ Advanced image analysis and OCR
           - 📊 Detailed explanations of charts, diagrams, and tables
           
           **Instructions:**
           1. Click "Initialize System" to set up the AI
           2. Upload your PDF documents
           3. Ask detailed questions about your documents
           """)

           with gr.Row():
               with gr.Column(scale=1):
                   # Simplified initialization
                   gr.Markdown("## 🚀 System Setup")
                   
                   with gr.Group():
                       init_btn = gr.Button("🤖 Initialize System", variant="primary", size="lg")
                       system_status = gr.Markdown("❌ **Status:** Not initialized")
                   
                   # Document upload
                   gr.Markdown("## 📄 Document Upload")
                   
                   pdf_files = gr.File(
                       label="Upload PDF Documents",
                       file_types=[".pdf"],
                       file_count="multiple",
                       height=100
                   )
                   
                   upload_btn = gr.Button("📤 Process Documents", variant="secondary")
                   upload_status = gr.Markdown("No documents uploaded")
                   
                   # Document statistics
                   with gr.Accordion("📊 Document Statistics", open=False):
                       doc_stats_display = gr.JSON(label="Statistics")
                   
                   # System controls
                   gr.Markdown("## ⚙️ Controls")
                   reset_btn = gr.Button("🔄 Reset Conversation")

               with gr.Column(scale=2):
                   # Main chat interface
                   gr.Markdown("## 💬 Research Chat")
                   
                   chatbot = gr.Chatbot(
                       height=500,
                       show_label=False,
                       avatar_images=("👤", "🤖"),
                       bubble_full_width=False
                   )
                   
                   with gr.Row():
                       msg_input = gr.Textbox(
                           placeholder="Ask a detailed question about your documents...",
                           label="Your Question",
                           lines=2,
                           scale=4
                       )
                       ask_btn = gr.Button("🔍 Ask", variant="primary", scale=1)
                   
                   # Enhanced reference display
                   with gr.Accordion("📚 References & Sources", open=True):
                       references_display = gr.HTML(label="References")
                   
                   # Image gallery with analysis
                   with gr.Accordion("🖼️ Visual Content Analysis", open=True):
                       image_gallery = gr.Gallery(
                           label="Referenced Images",
                           show_label=False,
                           elem_id="gallery",
                           columns=2,
                           rows=1,
                           height=300
                       )
                       image_analysis = gr.HTML(label="Image Analysis")

           # Event handlers
           init_btn.click(
               fn=self.initialize_system,
               outputs=[system_status]
           )
           
           upload_btn.click(
               fn=self.upload_documents,
               inputs=[pdf_files],
               outputs=[upload_status, doc_stats_display]
           )
           
           def chat_handler(message, history):
               if history is None:
                   history = []
               return self.process_query(message, history)
           
           ask_btn.click(
               fn=chat_handler,
               inputs=[msg_input, chatbot],
               outputs=[chatbot, msg_input, references_display, image_gallery, image_analysis]
           )
           
           msg_input.submit(
               fn=chat_handler,
               inputs=[msg_input, chatbot],
               outputs=[chatbot, msg_input, references_display, image_gallery, image_analysis]
           )
           
           reset_btn.click(
               fn=self.reset_system,
               outputs=[chatbot, references_display, image_gallery, image_analysis, upload_status]
           )

   def initialize_system(self):
       """Initialize the system with default optimized settings."""
       try:
           logger.info("Initializing system with default settings...")
           
           # Initialize VertexAI
           self.vertex_gen_ai = VertexGenAI()
           
           # Test connection
           test_response = self.vertex_gen_ai.generate_content("Test connection")
           if not test_response:
               return "❌ **Status:** VertexAI connection failed"
           
           # Initialize RAG system
           self.rag_system = EnhancedRAGSystem(vertex_gen_ai=self.vertex_gen_ai)
           
           return """✅ **Status:** System initialized successfully!
           
**Configuration:**
- 🔍 HNSW index with cosine similarity
- 📖 Research-style numbered references
- 🖼️ Advanced image analysis enabled
- 📊 OCR and visual content extraction enabled
           
Ready to process documents!"""
           
       except Exception as e:
           error_msg = f"❌ **Status:** Initialization failed - {str(e)}"
           logger.error(error_msg)
           return error_msg

   def upload_documents(self, files):
       """Upload and process documents."""
       if not self.rag_system:
           return "❌ Please initialize the system first", {}
       
       if not files:
           return "❌ No files selected", {}
       
       results = []
       total_processed = 0
       
       for file in files:
           try:
               result = self.rag_system.upload_pdf(file.name)
               results.append(result)
               if "✅" in result:
                   total_processed += 1
           except Exception as e:
               results.append(f"❌ Error processing {os.path.basename(file.name)}: {str(e)}")
       
       # Get statistics
       stats = self.rag_system.get_document_stats()
       
       status = f"""## 📄 Document Processing Complete

**Results:** {total_processed}/{len(files)} files processed successfully

""" + "\n".join(results)
       
       return status, stats

   def process_query(self, message, history):
       """Process user query and return formatted response."""
       if not message or not message.strip():
           return history, "", "", [], ""
       
       if not self.rag_system:
           error_response = "❌ Please initialize the system and upload documents first."
           return history + [[message, error_response]], "", "", [], ""
       
       try:
           # Get response from RAG system
           result = self.rag_system.ask(message)
           
           answer = result["answer"]
           references = result["references"]
           images = result["images"]
           
           # Format references as HTML
           references_html = self._format_references_html(references)
           
           # Prepare images for gallery
           gallery_images = []
           image_analysis_html = ""
           
           if images:
               image_analysis_parts = []
               for img in images:
                   try:
                       # Decode base64 image
                       img_bytes = base64.b64decode(img["base64"])
                       img_pil = Image.open(BytesIO(img_bytes))
                       gallery_images.append(img_pil)
                       
                       # Add analysis
                       analysis_part = f"""
                       <div class="image-analysis">
                           <h4>📊 {img['type'].title()} from {img['document']} (Page {img['page']})</h4>
                           <p><strong>Analysis:</strong> {img['analysis']}</p>
                           <p><strong>Confidence:</strong> {img['confidence']:.2f}</p>
                       </div>
                       """
                       image_analysis_parts.append(analysis_part)
                       
                   except Exception as e:
                       logger.error(f"Error processing image: {str(e)}")
               
               image_analysis_html = "".join(image_analysis_parts)
           
           # Update chat history
           updated_history = history + [[message, answer]]
           
           return updated_history, "", references_html, gallery_images, image_analysis_html
           
       except Exception as e:
           error_response = f"❌ Error processing query: {str(e)}"
           logger.error(error_response)
           return history + [[message, error_response]], "", "", [], ""

   def _format_references_html(self, references):
       """Format references in research paper style."""
       if not references:
           return "<p>No references found.</p>"
       
       html_parts = ["<h3>📚 References</h3>"]
       
       for ref in references:
           ref_id = ref["id"]
           doc = ref["document"]
           pages = ref["pages"]
           similarity = ref.get("similarity", 0.0)
           content = ref["content"]
           is_image = ref.get("is_image", False)
           
           # Format page numbers
           if len(pages) > 1:
               page_str = f"pp. {min(pages)}-{max(pages)}"
           else:
               page_str = f"p. {pages[0]}" if pages else "p. ?"
           
           # Create reference entry
           ref_type = "🖼️ Image" if is_image else "📄 Text"
           
           ref_html = f"""
           <div class="reference-box">
               <strong>[{ref_id}]</strong> {ref_type} - {doc}, {page_str}
               <br><small><strong>Similarity:</strong> {similarity:.3f}</small>
               <br><small><strong>Content:</strong> {content}</small>
           </div>
           """
           html_parts.append(ref_html)
       
       return "".join(html_parts)

   def reset_system(self):
       """Reset the conversation and clear displays."""
       if self.rag_system:
           self.rag_system.reset_conversation()
       
       return [], "", [], "", "🔄 **Status:** Conversation reset. Documents remain loaded."

# Enhanced helper functions
def setup_optimized_environment():
   """Set up optimized environment for the RAG system."""
   try:
       # Set environment variables for optimal performance
       os.environ["TOKENIZERS_PARALLELISM"] = "false"
       os.environ["OMP_NUM_THREADS"] = "4"
       
       # Configure logging
       logging.getLogger("transformers").setLevel(logging.WARNING)
       logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
       
       logger.info("Environment optimized for RAG system")
       return True
       
   except Exception as e:
       logger.error(f"Error setting up environment: {str(e)}")
       return False

def launch_enhanced_rag():
   """Launch the enhanced RAG interface."""
   setup_optimized_environment()
   interface = SimplifiedRAGInterface()
   return interface.interface

def main():
   """Main function with auto-configuration."""
   print("🚀 Starting Enhanced RAG System with HNSW and Research References")
   
   # Check environment
   try:
       import google.colab
       print("📱 Running in Google Colab")
       
       # Set up authentication
       from google.colab import auth
       auth.authenticate_user()
       print("✅ Google authentication completed")
       
       # Create directories
       os.makedirs("/content/uploads", exist_ok=True)
       print("📁 Created directories")
       
   except ImportError:
       print("💻 Running in local environment")
   
   print("\n🔧 **System Features:**")
   print("- HNSW index with cosine similarity for fast retrieval")
   print("- Research paper-style numbered references")
   print("- Advanced image analysis with OCR")
   print("- Detailed explanations of visual content")
   print("- Simplified UI with optimized defaults")
   
   print("\n📋 **Usage Instructions:**")
   print("1. Click 'Initialize System' (uses optimized default settings)")
   print("2. Upload your PDF documents")
   print("3. Ask detailed research questions")
   print("4. Get comprehensive answers with numbered references")
   
   # Launch interface
   interface = launch_enhanced_rag()
   interface.launch(
       server_name="0.0.0.0",
       server_port=7860,
       share=True,
       debug=False,
       show_error=True
   )

if __name__ == "__main__":
   main()
