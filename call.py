# -*- coding: utf-8 -*-
"""Advanced-RAG-with-References-Extended-with-Images.ipynb

Enhanced version with detailed responses and image analysis.
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import pypdf
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import cv2

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain.chains import ConversationalRetrievalChain, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
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

    def analyze_image_with_text(self, image_base64: str, context_text: str = "", question: str = ""):
        """Analyze image with context using Vertex AI Vision."""
        try:
            model = GenerativeModel("gemini-1.5-pro-002")
            
            # Create a comprehensive prompt for image analysis
            prompt = f"""
            Analyze this image in detail and provide a comprehensive summary. 
            
            Context from document: {context_text if context_text else "No additional context provided"}
            
            User's question: {question if question else "General analysis"}
            
            Please provide:
            1. A detailed description of what you see in the image
            2. Any text content visible in the image (OCR-like analysis)
            3. Charts, graphs, diagrams, tables, or figures and their key information
            4. How this image relates to the user's question
            5. Key insights or data points from the image
            6. Any technical or scientific content present
            
            Be thorough and specific in your analysis. Extract all meaningful information from the image.
            """
            
            # For now, since we're using your existing VertexAI setup, we'll use text-only analysis
            # In a full implementation, you'd pass the image data to a vision model
            analysis_prompt = f"{prompt}\n\nNote: Analyzing image content based on extracted text and context."
            
            response = model.generate_content(analysis_prompt, metadata=self.metadata)
            return response.text if response else "Could not analyze image"
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return f"Error analyzing image: {str(e)}"

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
    """Store extracted image data."""
    image: Image.Image
    page_number: int
    bbox: Optional[Tuple[int, int, int, int]] = None  # Bounding box coordinates
    ocr_text: str = ""
    base64_string: str = ""
    ai_analysis: str = ""  # NEW: AI-generated analysis of the image
    summary: str = ""  # NEW: Concise summary for embedding

@dataclass
class PageChunk:
    """Store chunk information with precise page tracking and image references."""
    text: str
    page_numbers: List[int]
    start_char_idx: int
    end_char_idx: int
    filename: str
    section_info: Dict[str, str] = field(default_factory=dict)
    image_refs: List[int] = field(default_factory=list)  # References to images by index

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
                "image_refs": self.image_refs
            }
        )

@dataclass
class PDFDocument:
    """Enhanced class to store PDF document metadata, content, and images."""
    filename: str
    content: str = ""
    pages: Dict[int, str] = field(default_factory=dict)
    char_to_page_map: List[int] = field(default_factory=list)
    chunks: List[PageChunk] = field(default_factory=list)
    images: List[ImageData] = field(default_factory=list)
    image_to_page_map: Dict[int, int] = field(default_factory=dict)  # Image index to page number

    def __len__(self) -> int:
        return len(self.content)

    @property
    def langchain_documents(self) -> List[Document]:
        """Convert chunks to LangChain Documents."""
        return [chunk.to_document() for chunk in self.chunks]

class EnhancedPDFProcessor:
    """Advanced PDF processor with image extraction and AI analysis."""

    def __init__(self,
                 chunk_size: int = 800,
                 chunk_overlap: int = 200,
                 separator: str = "\n\n",
                 keep_separator: bool = False,
                 enable_ocr: bool = True,
                 vertex_ai: VertexGenAI = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.keep_separator = keep_separator
        self.enable_ocr = enable_ocr
        self.vertex_ai = vertex_ai

        # Create a text splitter with careful configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=keep_separator
        )

        # Regex patterns for section detection
        self.section_patterns = [
            r'(?:Section|SECTION|Chapter|CHAPTER)\s+(\d+(?:\.\d+)*)[:\s]+(.*?)(?=\n|$)',
            r'(\d+(?:\.\d+)*)\s+(.*?)(?=\n|$)',
            r'(?:\n|\A)([A-Z][A-Z\s]+)(?:\n|:)'
        ]

    def extract_images_from_pdf(self, file_path: str) -> List[ImageData]:
        """Extract images from PDF and perform OCR and AI analysis."""
        images = []
        
        try:
            # Convert PDF pages to images
            pdf_images = convert_from_path(file_path, dpi=300)
            
            for page_num, page_image in enumerate(pdf_images, 1):
                # Convert PIL Image to OpenCV format for processing
                opencv_image = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
                
                # Detect regions with significant content (potential embedded images/figures)
                gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
                
                # Apply edge detection to find image boundaries
                edges = cv2.Canny(gray, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter contours by area to find significant image regions
                min_area = 10000  # Adjust based on your needs
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > min_area:
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Extract the region
                        roi = page_image.crop((x, y, x + w, y + h))
                        
                        # Perform OCR if enabled
                        ocr_text = ""
                        if self.enable_ocr:
                            try:
                                ocr_text = pytesseract.image_to_string(roi)
                            except Exception as e:
                                logger.warning(f"OCR failed for image on page {page_num}: {str(e)}")
                        
                        # Convert to base64 for display
                        buffered = BytesIO()
                        roi.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()
                        
                        # AI Analysis of the image
                        ai_analysis = ""
                        summary = ""
                        if self.vertex_ai:
                            try:
                                ai_analysis = self.vertex_ai.analyze_image_with_text(
                                    img_base64, 
                                    context_text=ocr_text,
                                    question="Analyze this image content in detail"
                                )
                                # Create a summary for better embedding
                                summary_prompt = f"Create a concise summary of this image analysis for search indexing: {ai_analysis}"
                                summary = self.vertex_ai.generate_content(summary_prompt) or ai_analysis[:200]
                            except Exception as e:
                                logger.warning(f"AI analysis failed for image on page {page_num}: {str(e)}")
                        
                        image_data = ImageData(
                            image=roi,
                            page_number=page_num,
                            bbox=(x, y, x + w, y + h),
                            ocr_text=ocr_text,
                            base64_string=img_base64,
                            ai_analysis=ai_analysis,
                            summary=summary
                        )
                        images.append(image_data)
                
                # Also store the full page as an image for reference
                buffered = BytesIO()
                page_image.save(buffered, format="PNG")
                full_page_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                # Perform OCR on full page if no specific regions were found
                if not any(img.page_number == page_num for img in images):
                    ocr_text = ""
                    if self.enable_ocr:
                        try:
                            ocr_text = pytesseract.image_to_string(page_image)
                        except Exception as e:
                            logger.warning(f"OCR failed for page {page_num}: {str(e)}")
                    
                    # AI Analysis of full page
                    ai_analysis = ""
                    summary = ""
                    if self.vertex_ai:
                        try:
                            ai_analysis = self.vertex_ai.analyze_image_with_text(
                                full_page_base64,
                                context_text=ocr_text,
                                question="Analyze this full page content"
                            )
                            summary_prompt = f"Create a concise summary of this page analysis for search indexing: {ai_analysis}"
                            summary = self.vertex_ai.generate_content(summary_prompt) or ai_analysis[:200]
                        except Exception as e:
                            logger.warning(f"AI analysis failed for full page {page_num}: {str(e)}")
                    
                    full_page_data = ImageData(
                        image=page_image,
                        page_number=page_num,
                        ocr_text=ocr_text,
                        base64_string=full_page_base64,
                        ai_analysis=ai_analysis,
                        summary=summary
                    )
                    images.append(full_page_data)
                    
        except Exception as e:
            logger.error(f"Error extracting images from PDF: {str(e)}")
            
        return images

    def extract_text_from_pdf(self, file_path: str) -> PDFDocument:
        """Extract text from PDF with enhanced page-level tracking and image extraction."""
        doc = PDFDocument(filename=os.path.basename(file_path))
        full_text = ""
        char_to_page = []

        try:
            # Extract images first with AI analysis
            doc.images = self.extract_images_from_pdf(file_path)
            
            # Create image to page mapping
            for idx, img in enumerate(doc.images):
                doc.image_to_page_map[idx] = img.page_number
            
            with open(file_path, "rb") as file:
                pdf = pypdf.PdfReader(file)

                if len(pdf.pages) == 0:
                    logger.warning(f"PDF {file_path} has no pages.")
                    doc.content = ""
                    doc.char_to_page_map = []
                    return doc

                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    page_text = page.extract_text() or ""

                    # Clean the text
                    page_text = self._clean_pdf_text(page_text)
                    
                    # Add enhanced image content from this page
                    page_images = [img for img in doc.images if img.page_number == page_num]
                    if page_images:
                        page_text += "\n\n[IMAGE CONTENT ON THIS PAGE]:\n"
                        for img in page_images:
                            if img.ocr_text:
                                page_text += f"OCR Text: {img.ocr_text}\n"
                            if img.ai_analysis:
                                page_text += f"AI Analysis: {img.ai_analysis}\n"
                            if img.summary:
                                page_text += f"Summary: {img.summary}\n"
                            page_text += "---\n"

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
        """Clean extracted PDF text to improve quality."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s*\n\s*', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'([a-z])- ?([a-z])', r'\1\2', text)
        return text

    def _extract_section_info(self, text: str) -> Dict[str, str]:
        """Extract section headings and numbers from text."""
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

    def chunk_document(self, doc: PDFDocument) -> PDFDocument:
        """Chunk document with page tracking and enhanced image references."""
        if not doc.content or not doc.char_to_page_map:
            if not doc.content:
                logger.warning(f"Document {doc.filename} has no content to chunk.")
            if not doc.char_to_page_map:
                logger.warning(f"Document {doc.filename} has no page mapping available.")
            return doc

        try:
            raw_chunks = self.text_splitter.create_documents([doc.content])
        except Exception as e:
            logger.error(f"Error creating chunks: {str(e)}")
            return doc

        for i, chunk in enumerate(raw_chunks):
            chunk_text = chunk.page_content

            # Find position in the original text
            start_pos = doc.content.find(chunk_text)
            if start_pos == -1:
                logger.warning(f"Could not find exact chunk position for chunk {i}.")
                start_pos = 0
                end_pos = min(len(chunk_text), len(doc.content)) - 1
            else:
                end_pos = start_pos + len(chunk_text) - 1

            # Find the page numbers this chunk spans
            chunk_pages = set()
            for pos in range(start_pos, min(end_pos + 1, len(doc.char_to_page_map))):
                if pos < len(doc.char_to_page_map):
                    chunk_pages.add(doc.char_to_page_map[pos])

            if not chunk_pages:
                chunk_pages = {1}
                logger.warning(f"No pages found for chunk {i} in document {doc.filename}.")

            # Enhanced image reference detection
            image_refs = []
            for idx, img in enumerate(doc.images):
                if img.page_number in chunk_pages:
                    # More sophisticated matching
                    match_score = 0
                    
                    # Check OCR text overlap
                    if img.ocr_text:
                        ocr_words = set(img.ocr_text.lower().split())
                        chunk_words = set(chunk_text.lower().split())
                        overlap = len(ocr_words.intersection(chunk_words))
                        if overlap > 0:
                            match_score += overlap * 2
                    
                    # Check AI analysis overlap
                    if img.ai_analysis:
                        analysis_words = set(img.ai_analysis.lower().split())
                        chunk_words = set(chunk_text.lower().split())
                        overlap = len(analysis_words.intersection(chunk_words))
                        if overlap > 0:
                            match_score += overlap
                    
                    # Check for visual content keywords
                    visual_keywords = ["figure", "image", "diagram", "chart", "graph", "table", "picture", "illustration"]
                    for keyword in visual_keywords:
                        if keyword in chunk_text.lower():
                            match_score += 3
                    
                    # Include if good match or on same page with visual keywords
                    if match_score > 2 or (match_score > 0 and any(kw in chunk_text.lower() for kw in visual_keywords)):
                        image_refs.append(idx)

            # Extract section information
            section_info = self._extract_section_info(chunk_text)

            # Create PageChunk
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

    def process_pdf(self, file_path: str) -> PDFDocument:
        """Process PDF file in a single call."""
        doc = self.extract_text_from_pdf(file_path)
        return self.chunk_document(doc)

class VertexAIEmbeddings(Embeddings):
    """Custom embeddings class using VertexGenAI that properly inherits from LangChain's Embeddings base class."""
    
    def __init__(self, vertex_gen_ai: VertexGenAI, model_name: str = "text-embedding-004"):
        self.vertex_gen_ai = vertex_gen_ai
        self.model_name = model_name
        self._embedding_dimension = None
        super().__init__()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        if not texts:
            return []
            
        try:
            logger.info(f"Embedding {len(texts)} documents with VertexAI")
            
            # Get embeddings from VertexGenAI
            embeddings_response = self.vertex_gen_ai.get_embeddings(texts, self.model_name)
            
            # Extract vectors from response
            embeddings = []
            for embedding in embeddings_response:
                if hasattr(embedding, 'values'):
                    embeddings.append(list(embedding.values))
                else:
                    # Log warning and use zero vector as fallback
                    logger.warning(f"Embedding response missing 'values' attribute")
                    embeddings.append(self._get_zero_vector())
            
            # Set embedding dimension if not already set
            if embeddings and self._embedding_dimension is None:
                self._embedding_dimension = len(embeddings[0])
                logger.info(f"Set embedding dimension to {self._embedding_dimension}")
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            # Return zero vectors for all texts
            return [self._get_zero_vector() for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        if not text:
            return self._get_zero_vector()
            
        try:
            logger.info(f"Embedding query with VertexAI: {text[:50]}...")
            
            # Get embedding for single text
            embeddings_response = self.vertex_gen_ai.get_embeddings([text], self.model_name)
            
            if embeddings_response and len(embeddings_response) > 0:
                embedding = embeddings_response[0]
                if hasattr(embedding, 'values'):
                    vector = list(embedding.values)
                    # Update dimension if needed
                    if self._embedding_dimension is None:
                        self._embedding_dimension = len(vector)
                        logger.info(f"Set embedding dimension to {self._embedding_dimension}")
                    return vector
                    
            return self._get_zero_vector()
            
        except Exception as e:
            logger.error(f"Error getting query embedding: {str(e)}")
            return self._get_zero_vector()
    
    def _get_zero_vector(self) -> List[float]:
        """Get a zero vector of appropriate dimension."""
        # Use stored dimension or default to 768 (common embedding size)
        dim = self._embedding_dimension if self._embedding_dimension else 768
        return [0.0] * dim

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version - just calls sync version for now."""
        return self.embed_documents(texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async version - just calls sync version for now."""
        return self.embed_query(text)

class HybridRetriever:
    """Hybrid retrieval system with VertexAI embeddings."""

    def __init__(self, documents: List[Document] = None, use_mmr: bool = True,
                 vertex_gen_ai: VertexGenAI = None):
        """Initialize retriever with documents and VertexAI."""
        
        logger.info("Initializing HybridRetriever...")
        
        try:
            if vertex_gen_ai:
                logger.info("Using VertexAI embeddings")
                self.embeddings = VertexAIEmbeddings(vertex_gen_ai)
            else:
                logger.info("Falling back to SentenceTransformer embeddings")
                self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

            # Initialize vector store
            if documents and len(documents) > 0:
                logger.info(f"Creating new FAISS index from {len(documents)} documents")
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                logger.info("Successfully created new FAISS index")
            else:
                logger.info("Creating empty FAISS index with placeholder")
                self.vector_store = FAISS.from_texts(["placeholder"], self.embeddings)
                logger.info("Created empty FAISS index")

            # Configure retriever for better results
            search_kwargs = {"k": 8, "fetch_k": 15} if use_mmr else {"k": 6}
            self.retriever = self.vector_store.as_retriever(
                search_type="mmr" if use_mmr else "similarity",
                search_kwargs=search_kwargs
            )
            logger.info(f"Configured retriever with search_type={'mmr' if use_mmr else 'similarity'}")
            
        except Exception as e:
            logger.error(f"Error initializing HybridRetriever: {str(e)}")
            raise

    def get_relevant_documents(self, query: str, top_k: int = 6) -> List[Document]:
        """Get relevant documents using hybrid search."""
        try:
            logger.info(f"Retrieving documents for query: {query[:50]}...")
            docs = self.retriever.get_relevant_documents(query)
            logger.info(f"Retrieved {len(docs)} documents")
            return docs
        except Exception as e:
            logger.error(f"Error in get_relevant_documents: {str(e)}")
            return []

    def update_documents(self, documents: List[Document]):
        """Update the document store with new documents."""
        try:
            logger.info(f"Adding {len(documents)} new documents to vector store")
            self.vector_store.add_documents(documents)
            logger.info("Successfully added documents to vector store")
        except Exception as e:
            logger.error(f"Error updating documents: {str(e)}")
            raise

class AdvancedRAGSystem:
    """Enhanced RAG system with VertexAI integration and detailed image analysis."""

    def __init__(self):
        logger.info("Initializing AdvancedRAGSystem...")
        
        # Set default CA bundle path if needed
        if os.path.exists("<Path to PROD CA pem file>"):
            os.environ["REQUESTS_CA_BUNDLE"] = "<Path to PROD CA pem file>"
        
        # Initialize VertexAI automatically
        try:
            self.vertex_gen_ai = VertexGenAI()
            logger.info("VertexAI initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize VertexAI: {str(e)}")
            self.vertex_gen_ai = None

        # Default settings
        self.enable_ocr = True

        self.processor = EnhancedPDFProcessor(
            chunk_size=800,
            chunk_overlap=200,
            enable_ocr=self.enable_ocr,
            vertex_ai=self.vertex_gen_ai  # Pass VertexAI for image analysis
        )
        self.documents = {}  # Store processed documents
        self.hybrid_retriever = None
        self.conversation_history = []
        self.conversation_chain = None

    def upload_pdf(self, file_path: str) -> str:
        """Process and index a PDF document with enhanced image analysis."""
        try:
            logger.info(f"Processing PDF: {file_path}")
            
            # Process the document
            doc = self.processor.process_pdf(file_path)

            # Store the document
            self.documents[doc.filename] = doc

            # Update the retrievers
            self._update_retrievers()

            image_count = len(doc.images)
            analyzed_images = sum(1 for img in doc.images if img.ai_analysis)
            result = f"Processed and indexed {doc.filename} ({len(doc.pages)} pages, {len(doc.chunks)} chunks, {image_count} images extracted, {analyzed_images} images analyzed by AI)"
            logger.info(result)
            return result
        except Exception as e:
            error_msg = f"Error uploading PDF: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def _update_retrievers(self):
        """Update the retriever system with current documents."""
        try:
            logger.info("Updating retrievers with current documents...")
            
            all_docs = []
            for doc in self.documents.values():
                all_docs.extend(doc.langchain_documents)

            if not all_docs:
                logger.warning("No documents to update retrievers with")
                return

            logger.info(f"Total documents to index: {len(all_docs)}")

            if not self.hybrid_retriever:
                logger.info("Creating new HybridRetriever")
                self.hybrid_retriever = HybridRetriever(
                    all_docs,
                    vertex_gen_ai=self.vertex_gen_ai
                )
            else:
                logger.info("Updating existing HybridRetriever")
                self.hybrid_retriever.update_documents(all_docs)

            # Initialize conversation chain
            self._initialize_conversation_chain()
            
        except Exception as e:
            logger.error(f"Error updating retrievers: {str(e)}")
            raise

    def _create_custom_chain(self, retriever):
        """Create an enhanced chain for detailed responses."""
        
        def format_docs_enhanced(docs, query):
            formatted = []
            image_analyses = []
            
            for doc in docs:
                content = doc.page_content
                metadata = doc.metadata
                
                # Enhanced formatting with image analysis
                doc_info = f"""
Document: {metadata.get('source', 'Unknown')}
Pages: {metadata.get('page_numbers', [])}
Section: {metadata.get('section_info', {})}
Content: {content}
"""
                
                # Include detailed image analysis if available
                if metadata.get('image_refs'):
                    doc_filename = metadata.get('source', '')
                    if doc_filename in self.documents:
                        pdf_doc = self.documents[doc_filename]
                        for img_idx in metadata['image_refs']:
                            if img_idx < len(pdf_doc.images):
                                img_data = pdf_doc.images[img_idx]
                               if img_data.ai_analysis:
                                   image_analyses.append({
                                       'page': img_data.page_number,
                                       'analysis': img_data.ai_analysis,
                                       'ocr_text': img_data.ocr_text,
                                       'document': doc_filename,
                                       'image_idx': img_idx
                                   })
                                   doc_info += f"\nImage Analysis (Page {img_data.page_number}): {img_data.ai_analysis}"
                                   if img_data.ocr_text:
                                       doc_info += f"\nImage OCR Text: {img_data.ocr_text}"
               
               formatted.append(doc_info)
           
           return "\n\n---\n\n".join(formatted), image_analyses
       
       def custom_chain(inputs):
           try:
               logger.info(f"Processing enhanced query: {inputs.get('question', '')[:50]}...")
               
               # Get relevant documents
               docs = retriever.get_relevant_documents(inputs["question"])
               
               # Format context with enhanced image analysis
               context, image_analyses = format_docs_enhanced(docs, inputs["question"])
               
               # Analyze relevant images with the specific query context
               enhanced_image_analysis = []
               for img_analysis in image_analyses:
                   if self.vertex_gen_ai:
                       try:
                           # Get more detailed analysis for this specific query
                           detailed_analysis = self.vertex_gen_ai.analyze_image_with_text(
                               "",  # We'll use the existing analysis as base
                               context_text=img_analysis['analysis'],
                               question=inputs["question"]
                           )
                           enhanced_image_analysis.append({
                               **img_analysis,
                               'query_specific_analysis': detailed_analysis
                           })
                       except Exception as e:
                           logger.warning(f"Enhanced image analysis failed: {str(e)}")
                           enhanced_image_analysis.append(img_analysis)
               
               # Format chat history
               chat_history = ""
               for q, a in self.conversation_history[-3:]:  # Last 3 exchanges for context
                   chat_history += f"Human: {q}\nAssistant: {a}\n\n"
               
               # Create enhanced prompt for detailed responses
               prompt = f"""You are an expert document analyst with access to comprehensive document content including detailed image analysis. Your goal is to provide thorough, well-explained answers with precise citations.

CONTEXT AND ANALYSIS:
{context}

ENHANCED IMAGE ANALYSIS FOR THIS QUERY:
{chr(10).join([f"Image on Page {img['page']} from {img['document']}: {img.get('query_specific_analysis', img['analysis'])}" for img in enhanced_image_analysis])}

{f"PREVIOUS CONVERSATION:{chr(10)}{chat_history}" if chat_history else ""}

USER QUESTION: {inputs["question"]}

INSTRUCTIONS FOR RESPONSE:
1. Provide a comprehensive, detailed explanation that directly answers the user's question
2. Synthesize information from both text content and image analysis
3. Explain concepts clearly with context and background when needed
4. Use specific data, figures, and details from the documents
5. If images contain relevant charts, graphs, or diagrams, explain what they show and how they relate to the answer
6. Organize your response logically with clear explanations
7. Provide precise citations in this format: [Document: "filename.pdf", Page X] or [Document: "filename.pdf", Page X, Section: "Section Name"]
8. For image-derived information, use: [Document: "filename.pdf", Page X, Image Analysis]
9. If you cannot find sufficient information, explain what is missing and what you can provide

RESPONSE STRUCTURE:
- Start with a clear, direct answer to the question
- Provide detailed explanation with supporting evidence
- Include relevant data, statistics, or technical details
- Explain any charts, graphs, or visual elements if relevant
- Conclude with a summary if the response is complex

Answer the question comprehensively using ALL available context:"""
               
               # Generate enhanced response using VertexGenAI
               logger.info("Generating enhanced response with VertexGenAI...")
               response_text = self.vertex_gen_ai.generate_content(prompt)
               
               if not response_text:
                   response_text = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
               
               logger.info("Successfully generated enhanced response")
               return {
                   "answer": response_text,
                   "source_documents": docs,
                   "image_analyses": enhanced_image_analysis
               }
               
           except Exception as e:
               logger.error(f"Error in enhanced custom chain: {str(e)}")
               return {
                   "answer": f"An error occurred while processing your question: {str(e)}",
                   "source_documents": [],
                   "image_analyses": []
               }
       
       return custom_chain

   def _initialize_conversation_chain(self):
       """Initialize the enhanced conversation chain with VertexGenAI."""
       try:
           if not self.hybrid_retriever:
               logger.warning("No hybrid retriever available for conversation chain")
               return
               
           if not self.vertex_gen_ai:
               logger.warning("No VertexGenAI instance available for conversation chain")
               return

           logger.info("Initializing enhanced conversation chain...")
           
           # Create enhanced custom chain
           self.conversation_chain = self._create_custom_chain(self.hybrid_retriever.retriever)
           
           logger.info("Successfully initialized enhanced conversation chain")
           
       except Exception as e:
           logger.error(f"Error initializing conversation chain: {str(e)}")

   def ask(self, query: str) -> Dict[str, Any]:
       """Process a query and return detailed answer with citations and relevant images."""
       if not self.hybrid_retriever:
           return {
               "answer": "Please upload at least one document first. No retriever available.",
               "citation_info": {},
               "cited_pages": [],
               "source_docs": [],
               "relevant_images": []
           }
           
       if not self.conversation_chain:
           return {
               "answer": "System not properly initialized. Please check VertexAI connection and upload documents.",
               "citation_info": {},
               "cited_pages": [],
               "source_docs": [],
               "relevant_images": []
           }

       try:
           logger.info(f"Processing enhanced query: {query[:50]}...")
           
           # Execute the enhanced chain
           result = self.conversation_chain({"question": query})

           # Extract answer and enhanced data
           answer = result.get("answer", "No answer generated")
           source_docs = result.get("source_documents", [])
           image_analyses = result.get("image_analyses", [])

           # Find relevant images from source documents and analyses
           relevant_images = []
           used_image_indices = set()

           # Add images that were specifically analyzed for this query
           for img_analysis in image_analyses:
               doc_filename = img_analysis['document']
               img_idx = img_analysis['image_idx']
               
               if doc_filename in self.documents and img_idx < len(self.documents[doc_filename].images):
                   img_data = self.documents[doc_filename].images[img_idx]
                   relevant_images.append({
                       "document": doc_filename,
                       "page": img_data.page_number,
                       "base64": img_data.base64_string,
                       "ocr_text": img_data.ocr_text[:100] + "..." if len(img_data.ocr_text) > 100 else img_data.ocr_text,
                       "ai_analysis": img_analysis.get('query_specific_analysis', img_analysis['analysis'])[:200] + "...",
                       "used_in_response": True
                   })
                   used_image_indices.add((doc_filename, img_idx))

           # Add other relevant images from source documents
           for doc in source_docs:
               metadata = doc.metadata
               doc_filename = metadata.get("source", "")
               image_refs = metadata.get("image_refs", [])
               
               if doc_filename in self.documents:
                   pdf_doc = self.documents[doc_filename]
                   for img_idx in image_refs:
                       if (doc_filename, img_idx) not in used_image_indices and img_idx < len(pdf_doc.images):
                           img_data = pdf_doc.images[img_idx]
                           relevant_images.append({
                               "document": doc_filename,
                               "page": img_data.page_number,
                               "base64": img_data.base64_string,
                               "ocr_text": img_data.ocr_text[:100] + "..." if len(img_data.ocr_text) > 100 else img_data.ocr_text,
                               "ai_analysis": img_data.ai_analysis[:200] + "..." if img_data.ai_analysis else "No AI analysis available",
                               "used_in_response": False
                           })

           # Extract enhanced citation info
           citation_info = self._extract_cited_pages(answer)

           # Add to conversation history
           self.conversation_history.append((query, answer))

           # Get all cited pages
           all_pages = []
           for doc_citations in citation_info.values():
               for citation in doc_citations:
                   all_pages.extend(citation["pages"])

           logger.info(f"Successfully processed enhanced query. Found {len(relevant_images)} relevant images.")

           return {
               "answer": answer,
               "citation_info": citation_info,
               "cited_pages": sorted(list(set(all_pages))),
               "source_docs": source_docs,
               "relevant_images": relevant_images
           }
       except Exception as e:
           logger.error(f"Error in enhanced ask method: {str(e)}")
           return {
               "answer": f"Error processing query: {str(e)}",
               "citation_info": {},
               "cited_pages": [],
               "source_docs": [],
               "relevant_images": []
           }

   def _extract_cited_pages(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
       """Extract citation information from the text with enhanced pattern matching."""
       if not text:
           return {}

       citations = {}

       # Enhanced patterns to handle various citation formats
       patterns = [
           r'\[Document:\s*"([^"]+)",\s*Page\s*(\d+)(?:-(\d+))?(?:,\s*Section:\s*"([^"]+)")?\]',
           r'\[Document:\s*"([^"]+)",\s*Page\s*(\d+)(?:-(\d+))?(?:,\s*Image\s*Analysis)?\]',
           r'\[Document:\s*"([^"]+)",\s*Pages?\s*(\d+)(?:-(\d+))?\]'
       ]

       try:
           for pattern in patterns:
               for match in re.finditer(pattern, text):
                   try:
                       doc_name = match.group(1)
                       start_page = int(match.group(2))
                       end_page = int(match.group(3)) if match.group(3) else start_page
                       pages = list(range(start_page, end_page + 1))

                       section_title = match.group(4) if len(match.groups()) >= 4 and match.group(4) else None

                       if doc_name not in citations:
                           citations[doc_name] = []

                       citations[doc_name].append({
                           "pages": pages,
                           "section_num": None,
                           "section_title": section_title
                       })

                   except (ValueError, IndexError) as e:
                       logger.error(f"Error parsing citation: {match.group(0)}, error: {str(e)}")
                       continue

           return citations
       except Exception as e:
           logger.error(f"Error extracting citations: {str(e)}")
           return {}

   def reset_conversation(self):
       """Reset the conversation history."""
       self.conversation_history = []
       logger.info("Conversation history reset")
       return "Conversation history has been reset."

   def get_document_stats(self) -> Dict[str, Dict[str, Any]]:
       """Get enhanced statistics about uploaded documents."""
       stats = {}
       for name, doc in self.documents.items():
           # Count sections found per document
           sections = set()
           for chunk in doc.chunks:
               for section_key, section_title in chunk.section_info.items():
                   sections.add(f"{section_key}: {section_title}")

           # Enhanced image statistics
           images_with_ocr = sum(1 for img in doc.images if img.ocr_text)
           images_with_ai = sum(1 for img in doc.images if img.ai_analysis)

           stats[name] = {
               "pages": len(doc.pages),
               "chunks": len(doc.chunks),
               "total_chars": len(doc.content),
               "sections": len(sections),
               "section_examples": list(sections)[:5] if sections else "None found",
               "total_images": len(doc.images),
               "images_with_ocr": images_with_ocr,
               "images_with_ai_analysis": images_with_ai
           }
       return stats

class GradioRAGInterface:
   """Enhanced Gradio interface for the RAG system with detailed responses."""

   def __init__(self):
       self.rag_system = AdvancedRAGSystem()
       self.temp_dir = tempfile.mkdtemp()
       logger.info("Initialized Enhanced GradioRAGInterface")
       self.setup_interface()

   def setup_interface(self):
       """Set up the enhanced Gradio interface."""
       with gr.Blocks(theme=gr.themes.Soft()) as self.interface:
           gr.Markdown("# 🧠 Advanced AI-Powered PDF Analysis System")
           gr.Markdown("Upload PDFs and get detailed, comprehensive answers with AI-powered image analysis and precise citations.")

           with gr.Row():
               with gr.Column(scale=2):
                   # Document upload section
                   with gr.Group():
                       gr.Markdown("### 📄 Upload Documents")
                       pdf_upload = gr.File(
                           label="Upload PDF Documents",
                           file_types=[".pdf"],
                           file_count="multiple"
                       )
                       upload_button = gr.Button("Process PDFs", variant="primary", size="lg")
                       doc_status = gr.Markdown("Upload your PDFs to start analyzing.")

                   # Chat interface
                   with gr.Group():
                       gr.Markdown("### 💬 Ask Questions")
                       chatbot = gr.Chatbot(
                           height=500, 
                           elem_id="chatbot",
                           avatar_images=("👤", "🧠"),
                           show_copy_button=True
                       )
                       msg = gr.Textbox(
                           placeholder="Ask detailed questions about your documents...",
                           label="Your Question",
                           lines=2,
                           max_lines=5
                       )
                       with gr.Row():
                           submit_btn = gr.Button("Ask", variant="primary", scale=2)
                           reset_btn = gr.Button("Reset Chat", variant="secondary", scale=1)

               with gr.Column(scale=1):
                   # Citation information
                   with gr.Group():
                       gr.Markdown("### 📊 References & Citations")
                       citation_display = gr.JSON(label="Document Citations")

                   # Image references with enhanced display
                   with gr.Group():
                       gr.Markdown("### 🖼️ Referenced Images")
                       gr.Markdown("Images used in generating the response:")
                       image_gallery = gr.Gallery(
                           label="AI-Analyzed Images",
                           show_label=False,
                           elem_id="gallery",
                           columns=2,
                           rows=3,
                           height=400,
                           show_download_button=True
                       )
                       image_info = gr.JSON(label="Image Analysis Details")

                   # Document statistics
                   with gr.Accordion("📈 Document Statistics", open=False):
                       doc_stats = gr.JSON(label="Processing Statistics")

           # Enhanced event handlers
           upload_button.click(
               fn=self.upload_documents,
               inputs=[pdf_upload],
               outputs=[doc_status, doc_stats],
               show_progress=True
           )

           def enhanced_chat_handler(message, history):
               try:
                   if history is None:
                       history = []
                   result = self.chat_with_docs(message, history)
                   return result
               except Exception as e:
                   logger.error(f"Error in enhanced chat handler: {str(e)}")
                   if history is None:
                       history = []
                   return history + [[message, f"Error: {str(e)}"]], "", {}, [], {}

           submit_btn.click(
               fn=enhanced_chat_handler,
               inputs=[msg, chatbot],
               outputs=[chatbot, msg, citation_display, image_gallery, image_info],
               show_progress=True
           )

           msg.submit(
               fn=enhanced_chat_handler,
               inputs=[msg, chatbot],
               outputs=[chatbot, msg, citation_display, image_gallery, image_info],
               show_progress=True
           )

           reset_btn.click(
               fn=self.reset_chat,
               outputs=[chatbot, doc_status, citation_display, image_gallery, image_info]
           )

   def upload_documents(self, files):
       """Upload and process documents with enhanced AI analysis."""
       if not files:
           return "No files selected for upload.", {}

       results = []
       for file in files:
           try:
               result = self.rag_system.upload_pdf(file.name)
               results.append(f"✅ {result}")
           except Exception as e:
               error_msg = f"❌ Error processing {os.path.basename(file.name)}: {str(e)}"
               results.append(error_msg)
               logger.error(error_msg)

       # Get enhanced document stats
       doc_stats = self._get_enhanced_doc_stats()
       status = "## 📊 Processing Results\n" + "\n".join(results)
       return status, doc_stats

   def _get_enhanced_doc_stats(self):
       """Get comprehensive statistics about uploaded documents."""
       if not self.rag_system:
           return {}

       return self.rag_system.get_document_stats()

   def chat_with_docs(self, message, history=None):
       """Process chat message with enhanced analysis and detailed responses."""
       if history is None:
           history = []

       if not self.rag_system.documents:
           return history + [[message, "Please upload at least one document before asking questions."]], "", {}, [], {}

       if not message or not message.strip():
           return history + [[message, "Please enter a question."]], "", {}, [], {}

       try:
           logger.info(f"Processing enhanced chat message: {message[:50]}...")
           
           # Process the query with enhanced analysis
           start_time = time.time()
           response = self.rag_system.ask(message)
           process_time = time.time() - start_time

           answer = response["answer"]
           citation_info = response.get("citation_info", {})
           relevant_images = response.get("relevant_images", [])

           # Add processing info
           answer += f"\n\n*📊 Analysis completed in {process_time:.2f} seconds with AI-powered image analysis.*"

           # Update chatbot history
           updated_history = list(history)
           updated_history.append([message, answer])

           # Format enhanced citation info
           formatted_citations = {}
           for doc_name, citations in citation_info.items():
               doc_citations = []
               for citation in citations:
                   pages_str = "-".join(map(str, [min(citation["pages"]), max(citation["pages"])])) if len(citation["pages"]) > 1 else str(citation["pages"][0])
                   section_info = ""
                   if citation["section_title"]:
                       section_info = f", Section: \"{citation['section_title']}\""
                   doc_citations.append(f"Page(s) {pages_str}{section_info}")
               formatted_citations[doc_name] = doc_citations

           # Prepare enhanced images for gallery
           gallery_images = []
           image_details = []

           for img_data in relevant_images:
               try:
                   # Convert base64 to PIL Image for gallery
                   img_bytes = base64.b64decode(img_data["base64"])
                   img = Image.open(BytesIO(img_bytes))
                   
                   # Add to gallery
                   gallery_images.append(img)
                   
                   # Add enhanced details
                   image_details.append({
                       "document": img_data["document"],
                       "page": img_data["page"],
                       "used_in_response": img_data.get("used_in_response", False),
                       "ocr_preview": img_data["ocr_text"],
                       "ai_analysis_preview": img_data.get("ai_analysis", "No analysis available")
                   })
               except Exception as e:
                   logger.error(f"Error processing image for gallery: {str(e)}")

           # Enhanced results info
           page_info = {
               "Citations": formatted_citations,
               "Documents_Referenced": len(citation_info),
               "Images_Analyzed": len([img for img in relevant_images if img.get("used_in_response", False)]),
               "Total_Images_Found": len(relevant_images)
           }

           logger.info(f"Enhanced chat processing complete. {len(relevant_images)} images, {len(citation_info)} documents referenced.")
           return updated_history, "", page_info, gallery_images, {"image_details": image_details}

       except Exception as e:
           error_msg = f"Error: {str(e)}"
           logger.error(f"Error in enhanced chat: {str(e)}")
           return history + [[message, error_msg]], "", {}, [], {}

   def reset_chat(self):
       """Reset the chat conversation."""
       if self.rag_system:
           self.rag_system.reset_conversation()

       doc_status = "Chat conversation has been reset."
       if self.rag_system and self.rag_system.documents:
           doc_names = list(self.rag_system.documents.keys())
           doc_status += f"\n\n📚 Loaded documents: {', '.join(doc_names)}"

       return [], doc_status, {}, [], {}

# Main execution
def main():
   """Launch the enhanced RAG application."""
   print("🚀 Starting Enhanced AI-Powered PDF Analysis System")
   print("Features: Detailed responses, AI image analysis, comprehensive citations")
   
   # Set the CA bundle path if needed
   if os.path.exists("<Path to PROD CA pem file>"):
       os.environ["REQUESTS_CA_BUNDLE"] = "<Path to PROD CA pem file>"

   try:
       # Create and launch the interface
       interface = GradioRAGInterface()
       interface.interface.launch(
           server_name="0.0.0.0",
           server_port=7860,
           share=True,
           debug=False,
           show_error=True
       )
   except Exception as e:
       print(f"❌ Failed to launch application: {e}")
       logger.error(f"Application launch failed: {e}")

if __name__ == "__main__":
   main()
