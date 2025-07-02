# -*- coding: utf-8 -*-
"""Simple-RAG-with-Visual-References.ipynb

Enhanced RAG system with:
1. Simple, clean UI design
2. Actual image display (top 3)
3. Reference links in chat responses
4. Visual references section
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
    image_type: str = ""
    analysis: str = ""
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
        content = f"{self.text}{self.filename}{self.page_numbers}"
        self.chunk_hash = hashlib.md5(content.encode()).hexdigest()[:8]

    def to_document(self) -> Document:
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
        return [chunk.to_document() for chunk in self.chunks]

class EnhancedPDFProcessor:
    """Advanced PDF processor with image analysis."""

    def __init__(self):
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

        self.section_patterns = [
            r'(?:Section|SECTION|Chapter|CHAPTER)\s+(\d+(?:\.\d+)*)[:\s]+(.*?)(?=\n|$)',
            r'(\d+(?:\.\d+)*)\s+(.*?)(?=\n|$)',
            r'(?:\n|\A)([A-Z][A-Z\s]{2,})(?:\n|:)',
            r'(?:\n|\A)(Abstract|Introduction|Methodology|Results|Discussion|Conclusion)(?:\n|:)'
        ]

    def analyze_image(self, image: Image.Image, ocr_text: str) -> Tuple[str, str, float]:
        """Analyze image to determine type and generate description."""
        try:
            img_array = np.array(image)
            height, width = img_array.shape[:2]
            aspect_ratio = width / height
            
            image_type = "figure"
            analysis = ""
            confidence = 0.5
            
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
            
            if image_type == "chart":
                analysis = f"Chart/Graph showing data visualization. Contains: {ocr_text[:150]}..." if ocr_text else "Data visualization chart"
            elif image_type == "table":
                analysis = f"Table with structured data. Content: {ocr_text[:150]}..." if ocr_text else "Structured data table"
            elif image_type == "diagram":
                analysis = f"Diagram illustrating process/relationship. Shows: {ocr_text[:150]}..." if ocr_text else "Process or relationship diagram"
            else:
                analysis = f"Figure/Image. Content: {ocr_text[:150]}..." if ocr_text else "Visual figure or image"
            
            analysis += f" (Dimensions: {width}x{height}, Confidence: {confidence:.1f})"
            
            return image_type, analysis, confidence
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return "figure", f"Image analysis failed: {str(e)}", 0.1

    def extract_images_from_pdf(self, file_path: str) -> List[ImageData]:
        """Extract and analyze images from PDF."""
        images = []
        
        try:
            pdf_images = convert_from_path(file_path, dpi=200)  # Reduced DPI for faster processing
            
            for page_num, page_image in enumerate(pdf_images, 1):
                opencv_image = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
                
                edges = cv2.Canny(gray, 30, 100)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                min_area = 8000
                page_images_found = 0
                
                for contour in contours:
                    if page_images_found >= 2:  # Limit images per page
                        break
                        
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
                        
                        # Only keep high-confidence images
                        if confidence > 0.6:
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
                            page_images_found += 1
                    
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
                    ocr_texts = [img.ocr_text for img in doc.images if img.page_number == page_num and img.ocr_text]
                    if ocr_texts:
                        page_text += f"\n\n[Image Content from Page {page_num}]:\n" + "\n".join(ocr_texts)

                    if page_text.strip():
                        doc.pages[page_num] = page_text
                        full_text += page_text
                        char_to_page.extend([page_num] * len(page_text))

            doc.content = full_text
            doc.char_to_page_map = char_to_page
            
            return doc

        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            raise

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

            # Find related images
            image_refs = []
            for idx, img in enumerate(doc.images):
                if img.page_number in chunk_pages:
                    if img.ocr_text:
                        ocr_words = set(img.ocr_text.lower().split())
                        chunk_words = set(chunk_text.lower().split())
                        overlap = len(ocr_words.intersection(chunk_words))
                        if overlap > 2 or len(ocr_words) < 5:
                            image_refs.append(idx)
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
            embeddings_response = self.vertex_gen_ai.get_embeddings([text], self.model_name)
            
            if embeddings_response and len(embeddings_response) > 0:
                embedding = embeddings_response[0]
                if hasattr(embedding, 'values'):
                    vector = np.array(embedding.values, dtype=np.float32)
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
            
            texts = [doc.page_content for doc in documents]
            embeddings = self.embeddings.embed_documents(texts)
            
            if not embeddings:
                logger.error("No embeddings generated")
                return
            
            if self.hnsw_index is None:
                self.dimension = len(embeddings[0])
                self.hnsw_index = hnswlib.Index(space='cosine', dim=self.dimension)
                self.hnsw_index.init_index(max_elements=10000, ef_construction=200, M=16)
                logger.info(f"Initialized HNSW index with dimension {self.dimension}")
            
            start_idx = len(self.documents)
            ids = list(range(start_idx, start_idx + len(documents)))
            
            embeddings_array = np.array(embeddings, dtype=np.float32)
            self.hnsw_index.add_items(embeddings_array, ids)
            
            self.documents.extend(documents)
            self.document_embeddings.extend(embeddings)
            
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
            
            query_embedding = self.embeddings.embed_query(query)
            if not query_embedding:
                logger.error("Failed to get query embedding")
                return []
            
            query_array = np.array([query_embedding], dtype=np.float32)
            labels, distances = self.hnsw_index.knn_query(query_array, k=min(k, len(self.documents)))
            
            relevant_docs = []
            for label, distance in zip(labels[0], distances[0]):
                if label < len(self.documents):
                    doc = self.documents[label]
                    doc.metadata['similarity_score'] = 1 - distance
                    relevant_docs.append(doc)
            
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
        """Process query with research-style response and reference links."""
        if not self.hnsw_retriever or not self.documents:
            return {
                "answer": "Please upload documents first.",
                "references": [],
                "images": []
            }

        try:
            logger.info(f"Processing query: {query[:50]}...")
            
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
            all_images = []
            ref_map = {}
            
            for i, doc in enumerate(relevant_docs):
                metadata = doc.metadata
                doc_name = metadata.get('source', 'Unknown')
                pages = metadata.get('page_numbers', [])
                similarity = metadata.get('similarity_score', 0.0)
                
                # Create reference
                ref_id = self.reference_counter
                reference = {
                    "id": ref_id,
                    "document": doc_name,
                    "pages": pages,
                    "similarity": similarity,
                    "content": doc.page_content[:150] + "...",
                    "type": "text"
                }
                references.append(reference)
                ref_map[i] = ref_id
                self.reference_counter += 1
                
                # Add context with reference
                context_parts.append(f"[{ref_id}] {doc.page_content}")
                
                # Get images
                image_refs = metadata.get('image_refs', [])
                if image_refs and doc_name in self.documents:
                    pdf_doc = self.documents[doc_name]
                    for img_idx in image_refs:
                        if img_idx < len(pdf_doc.images):
                            img_data = pdf_doc.images[img_idx]
                            all_images.append({
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
                                "content": f"{img_data.image_type}: {img_data.analysis[:100]}...",
                                "type": "image"
                            }
                            references.append(img_ref)
                            self.reference_counter += 1
            
            # Select top 3 most relevant images
            top_images = sorted(all_images, key=lambda x: x['confidence'], reverse=True)[:3]
            
            # Create reference links for the response
            ref_links = {}
            for ref in references:
                ref_links[ref["id"]] = f"Ref {ref['id']}"
            
            context = "\n\n".join(context_parts)
            
            # Create research-style prompt with reference instructions
            prompt = f"""You are an expert academic researcher. Provide a detailed, comprehensive response using ONLY the provided context.

CRITICAL REQUIREMENTS:
1. Add numbered reference citations [1], [2], etc. after EVERY factual statement
2. Be extremely detailed and elaborate in explanations
3. For image-derived information, describe charts, diagrams, tables in detail
4. Use academic writing style with thorough analysis
5. Connect related concepts and provide comprehensive coverage

Context with References:
{context}

Reference Guidelines:
- Use [1], [2], [3] etc. for citations
- Available reference IDs: {list(ref_links.keys())}
- Cite after every factual claim
- For visual content, explain what the image shows and its significance

User Question: {query}

Provide a comprehensive academic response with detailed explanations and proper citations:"""

            response = self.vertex_gen_ai.generate_content(prompt)
            
            if not response:
                response = "I couldn't generate a response. Please try rephrasing your question."
            
            # Add reference links to the response
            for ref_id, ref_text in ref_links.items():
                # Replace citations with clickable links
                response = response.replace(f"[{ref_id}]", f"[[{ref_id}]](#ref{ref_id})")
            
            self.conversation_history.append((query, response))
            
            return {
                "answer": response,
                "references": references,
                "images": top_images
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

class SimpleRAGInterface:
    """Very simple UI with logo, chatbot, and three buttons."""

    def __init__(self):
        self.rag_system = None
        self.vertex_gen_ai = None
        self.current_references = []
        self.current_images = []
        logger.info("Initialized SimpleRAGInterface")
        self.setup_interface()

    def setup_interface(self):
        """Set up very simple Gradio interface."""
        
        # Custom CSS for clean, simple design
        custom_css = """
        .logo-container {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            margin-bottom: 20px;
            border-radius: 10px;
        }
        .logo-text {
            font-size: 28px;
            font-weight: bold;
            color: white;
            text-align: center;
        }
        .button-container {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin: 20px 0;
        }
        .reference-card {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            background-color: #f9f9f9;
        }
        .image-card {
           border: 1px solid #ddd;
           border-radius: 8px;
           padding: 15px;
           margin: 10px 0;
           background-color: #fff;
           box-shadow: 0 2px 4px rgba(0,0,0,0.1);
       }
       .ref-text {
           background-color: #e8f4fd;
           border-left: 4px solid #2196F3;
           padding: 10px;
           margin: 5px 0;
       }
       .ref-image {
           background-color: #f3e5f5;
           border-left: 4px solid #9c27b0;
           padding: 10px;
           margin: 5px 0;
       }
       """

       with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Research Assistant") as self.interface:
           
           # Header with logo and app name
           with gr.Row():
               with gr.Column():
                   gr.HTML("""
                   <div class="logo-container">
                       <div style="margin-right: 15px; font-size: 32px;">🔬</div>
                       <div class="logo-text">Research Assistant with Visual RAG</div>
                   </div>
                   """)
           
           # Main chatbot interface
           chatbot = gr.Chatbot(
               height=500,
               show_label=False,
               avatar_images=("👤", "🤖"),
               bubble_full_width=False,
               show_copy_button=True
           )
           
           # Input area
           with gr.Row():
               msg_input = gr.Textbox(
                   placeholder="Ask a question about your documents...",
                   show_label=False,
                   lines=2,
                   scale=5
               )
           
           # Three main buttons
           with gr.Row():
               upload_btn = gr.Button("📤 Upload Documents", variant="primary", size="lg", scale=1)
               ask_btn = gr.Button("🔍 Ask Question", variant="secondary", size="lg", scale=1)
               reset_btn = gr.Button("🔄 Reset", variant="stop", size="lg", scale=1)
           
           # File upload (initially hidden)
           upload_files = gr.File(
               label="Select PDF Documents",
               file_types=[".pdf"],
               file_count="multiple",
               visible=False
           )
           
           # Status display
           status_display = gr.Markdown("🤖 **Status:** Ready to initialize system")
           
           # References Section
           gr.Markdown("## 📚 Text References")
           text_references = gr.HTML(value="<p>No references yet. Upload documents and ask questions to see references here.</p>")
           
           # Images Section  
           gr.Markdown("## 🖼️ Visual References (Top 3)")
           
           with gr.Row():
               image1 = gr.Image(label="Image 1", visible=False, height=200)
               image2 = gr.Image(label="Image 2", visible=False, height=200)
               image3 = gr.Image(label="Image 3", visible=False, height=200)
           
           image_info = gr.HTML(value="<p>No images yet. Upload documents with visual content to see them here.</p>")

           # Hidden state to track initialization
           initialized = gr.State(False)

           # Event handlers
           def handle_upload_click():
               return gr.update(visible=True)
           
           def handle_upload_files(files, is_initialized):
               if not is_initialized:
                   # Initialize system first
                   try:
                       self.vertex_gen_ai = VertexGenAI()
                       test_response = self.vertex_gen_ai.generate_content("Test")
                       if test_response:
                           self.rag_system = EnhancedRAGSystem(vertex_gen_ai=self.vertex_gen_ai)
                           is_initialized = True
                       else:
                           return "❌ Failed to initialize VertexAI", False, gr.update(visible=False)
                   except Exception as e:
                       return f"❌ Initialization error: {str(e)}", False, gr.update(visible=False)
               
               if not files:
                   return "❌ No files selected", is_initialized, gr.update(visible=False)
               
               results = []
               for file in files:
                   try:
                       result = self.rag_system.upload_pdf(file.name)
                       results.append(result)
                   except Exception as e:
                       results.append(f"❌ Error: {str(e)}")
               
               status = "## 📄 Upload Results\n" + "\n".join(results)
               return status, is_initialized, gr.update(visible=False)
           
           def handle_ask(message, history, is_initialized):
               if not is_initialized or not self.rag_system:
                   error_msg = "❌ Please upload documents first"
                   if history is None:
                       history = []
                   return history + [[message, error_msg]], "", "", "", "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
               
               if not message.strip():
                   if history is None:
                       history = []
                   return history, message, "", "", "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
               
               try:
                   # Get response
                   result = self.rag_system.ask(message)
                   answer = result["answer"]
                   self.current_references = result["references"]
                   self.current_images = result["images"]
                   
                   # Update chat history
                   if history is None:
                       history = []
                   updated_history = history + [[message, answer]]
                   
                   # Format text references
                   text_refs_html = self._format_text_references()
                   
                   # Format image info
                   images_info_html = self._format_image_info()
                   
                   # Prepare images for display
                   img1, img2, img3 = self._prepare_images_for_display()
                   
                   return (updated_history, "", text_refs_html, images_info_html, "",
                          img1, img2, img3)
                   
               except Exception as e:
                   error_msg = f"❌ Error: {str(e)}"
                   if history is None:
                       history = []
                   return (history + [[message, error_msg]], "", "", "", "",
                          gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
           
           def handle_reset():
               if self.rag_system:
                   self.rag_system.reset_conversation()
               self.current_references = []
               self.current_images = []
               return ([], "🔄 Conversation reset", 
                      "<p>No references yet.</p>", 
                      "<p>No images yet.</p>",
                      gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
           
           # Wire up events
           upload_btn.click(
               fn=handle_upload_click,
               outputs=upload_files
           )
           
           upload_files.upload(
               fn=handle_upload_files,
               inputs=[upload_files, initialized],
               outputs=[status_display, initialized, upload_files]
           )
           
           ask_btn.click(
               fn=handle_ask,
               inputs=[msg_input, chatbot, initialized],
               outputs=[chatbot, msg_input, text_references, image_info, status_display, 
                       image1, image2, image3]
           )
           
           msg_input.submit(
               fn=handle_ask,
               inputs=[msg_input, chatbot, initialized],
               outputs=[chatbot, msg_input, text_references, image_info, status_display,
                       image1, image2, image3]
           )
           
           reset_btn.click(
               fn=handle_reset,
               outputs=[chatbot, status_display, text_references, image_info,
                       image1, image2, image3]
           )

   def _format_text_references(self):
       """Format text references as HTML."""
       if not self.current_references:
           return "<p>No references available.</p>"
       
       html_parts = []
       text_refs = [ref for ref in self.current_references if ref.get("type") != "image"]
       
       for ref in text_refs:
           ref_id = ref["id"]
           doc = ref["document"]
           pages = ref["pages"]
           similarity = ref.get("similarity", 0.0)
           content = ref["content"]
           
           # Format page numbers
           if len(pages) > 1:
               page_str = f"pp. {min(pages)}-{max(pages)}"
           else:
               page_str = f"p. {pages[0]}" if pages else "p. ?"
           
           ref_html = f"""
           <div class="reference-card" id="ref{ref_id}">
               <div class="ref-text">
                   <strong>[{ref_id}]</strong> 📄 {doc}, {page_str}
                   <br><small><strong>Relevance:</strong> {similarity:.3f}</small>
                   <br><em>"{content}"</em>
               </div>
           </div>
           """
           html_parts.append(ref_html)
       
       if not html_parts:
           return "<p>No text references found.</p>"
       
       return "".join(html_parts)

   def _format_image_info(self):
       """Format image information as HTML."""
       if not self.current_images:
           return "<p>No images found in the context.</p>"
       
       html_parts = []
       for i, img in enumerate(self.current_images):
           doc = img["document"]
           page = img["page"]
           img_type = img["type"]
           analysis = img["analysis"]
           confidence = img["confidence"]
           ref_id = img.get("reference_id", "")
           
           img_html = f"""
           <div class="image-card">
               <div class="ref-image">
                   <strong>Image {i+1} {f'[{ref_id}]' if ref_id else ''}</strong> 🖼️ {img_type.title()}
                   <br><strong>Source:</strong> {doc}, Page {page}
                   <br><strong>Confidence:</strong> {confidence:.2f}
                   <br><strong>Analysis:</strong> {analysis}
               </div>
           </div>
           """
           html_parts.append(img_html)
       
       return "".join(html_parts)

   def _prepare_images_for_display(self):
       """Prepare images for Gradio display components."""
       # Initialize all as hidden
       img1 = gr.update(visible=False)
       img2 = gr.update(visible=False) 
       img3 = gr.update(visible=False)
       
       if not self.current_images:
           return img1, img2, img3
       
       # Convert base64 images to PIL Images and show them
       display_images = []
       for img_data in self.current_images[:3]:  # Top 3 only
           try:
               img_bytes = base64.b64decode(img_data["base64"])
               pil_image = Image.open(BytesIO(img_bytes))
               display_images.append(pil_image)
           except Exception as e:
               logger.error(f"Error converting image: {str(e)}")
               continue
       
       # Update the image components
       if len(display_images) > 0:
           img1 = gr.update(value=display_images[0], visible=True, 
                          label=f"📊 {self.current_images[0]['type'].title()} - {self.current_images[0]['document']}")
       
       if len(display_images) > 1:
           img2 = gr.update(value=display_images[1], visible=True,
                          label=f"📊 {self.current_images[1]['type'].title()} - {self.current_images[1]['document']}")
       
       if len(display_images) > 2:
           img3 = gr.update(value=display_images[2], visible=True,
                          label=f"📊 {self.current_images[2]['type'].title()} - {self.current_images[2]['document']}")
       
       return img1, img2, img3

def launch_simple_rag():
   """Launch the simple RAG interface."""
   interface = SimpleRAGInterface()
   return interface.interface

def main():
   """Main function to launch the application."""
   print("🚀 Starting Simple RAG System with Visual References")
   
   # Check environment
   try:
       import google.colab
       print("📱 Running in Google Colab")
       
       from google.colab import auth
       auth.authenticate_user()
       print("✅ Authentication completed")
       
   except ImportError:
       print("💻 Running in local environment")
   
   print("\n🎯 **Simple Interface Features:**")
   print("- 🔬 Logo and clean design")
   print("- 💬 Central chatbot interface") 
   print("- 📤 Upload, 🔍 Ask, 🔄 Reset buttons")
   print("- 📚 Text references with clickable links")
   print("- 🖼️ Top 3 actual images displayed")
   print("- 🔗 Reference links in chat responses")
   
   # Launch interface
   interface = launch_simple_rag()
   interface.launch(
       server_name="0.0.0.0",
       server_port=7860,
       share=True,
       debug=False,
       show_error=True
   )

if __name__ == "__main__":
   main()
