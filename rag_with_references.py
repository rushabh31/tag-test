# -*- coding: utf-8 -*-
"""Advanced-RAG-with-References-Extended-with-Images.ipynb

This notebook builds on the original Advanced RAG system, adding:
1. Support for loading/saving FAISS vector databases
2. Integration with Google Vertex AI as an alternative to Groq
3. Image extraction and OCR capabilities for PDFs
4. Display of referenced images in responses
"""

# Install required packages
!pip install langchain langchain-groq sentence-transformers faiss-cpu pypdf gradio tiktoken -q
!pip install -q pypdf langchain langchain_groq faiss-cpu langchain_community sentence-transformers chromadb tiktoken langchain_core
!pip install -q google-cloud-aiplatform langchain-google-vertexai
!pip install -q pdf2image pytesseract pillow opencv-python-headless
!apt-get install -y poppler-utils tesseract-ocr

import os
import re
import tempfile
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Set, Union
from dataclasses import dataclass, field
import json
from tqdm.notebook import tqdm
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
from langchain_groq import ChatGroq
from langchain_google_vertexai import ChatVertexAI
from langchain.chains import ConversationalRetrievalChain, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import gradio as gr

# Import your VertexGenAI class
from connection import VertexGenAI

@dataclass
class ImageData:
    """Store extracted image data."""
    image: Image.Image
    page_number: int
    bbox: Optional[Tuple[int, int, int, int]] = None  # Bounding box coordinates
    ocr_text: str = ""
    base64_string: str = ""

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
    """Advanced PDF processor with image extraction and OCR capabilities."""

    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 separator: str = "\n\n",
                 keep_separator: bool = False,
                 enable_ocr: bool = True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.keep_separator = keep_separator
        self.enable_ocr = enable_ocr

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
        """Extract images from PDF and perform OCR if enabled."""
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
                        
                        image_data = ImageData(
                            image=roi,
                            page_number=page_num,
                            bbox=(x, y, x + w, y + h),
                            ocr_text=ocr_text,
                            base64_string=img_base64
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
                    
                    full_page_data = ImageData(
                        image=page_image,
                        page_number=page_num,
                        ocr_text=ocr_text,
                        base64_string=full_page_base64
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
            # Extract images first
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
                    
                    # Add OCR text from images on this page
                    ocr_texts = [img.ocr_text for img in doc.images if img.page_number == page_num]
                    if ocr_texts:
                        page_text += "\n\n[OCR Extracted Text]:\n" + "\n".join(ocr_texts)

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
        """Chunk document with page tracking and image references."""
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

            # Find related images
            image_refs = []
            for idx, img in enumerate(doc.images):
                if img.page_number in chunk_pages:
                    # Check if the image's OCR text is mentioned in the chunk
                    if img.ocr_text and any(text_part in chunk_text for text_part in img.ocr_text.split()[:5]):
                        image_refs.append(idx)
                    # Or if chunk mentions "figure", "image", "diagram" near this page
                    elif any(keyword in chunk_text.lower() for keyword in ["figure", "image", "diagram", "chart", "graph"]):
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

class VertexAIEmbeddings:
    """Custom embeddings class using VertexGenAI."""
    
    def __init__(self, vertex_gen_ai: VertexGenAI, model_name: str = "text-embedding-004"):
        self.vertex_gen_ai = vertex_gen_ai
        self.model_name = model_name
        self._embedding_dimension = None
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        if not texts:
            return []
            
        try:
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
            # Get embedding for single text
            embeddings_response = self.vertex_gen_ai.get_embeddings([text], self.model_name)
            
            if embeddings_response and len(embeddings_response) > 0:
                embedding = embeddings_response[0]
                if hasattr(embedding, 'values'):
                    vector = list(embedding.values)
                    # Update dimension if needed
                    if self._embedding_dimension is None:
                        self._embedding_dimension = len(vector)
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
                 faiss_index_path: str = None, vertex_gen_ai: VertexGenAI = None):
        """Initialize retriever with documents or existing FAISS index."""
        
        if vertex_gen_ai:
            self.embeddings = VertexAIEmbeddings(vertex_gen_ai)
        else:
            # Fallback to sentence transformers if VertexGenAI not provided
            self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        if faiss_index_path and os.path.exists(faiss_index_path):
            self.vector_store = FAISS.load_local(faiss_index_path, self.embeddings)
            logger.info(f"Loaded existing FAISS index from {faiss_index_path}")
        elif documents:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            logger.info("Created new FAISS index from documents")
        else:
            self.vector_store = FAISS.from_texts(["placeholder"], self.embeddings)
            logger.info("Created empty FAISS index")

        self.retriever = self.vector_store.as_retriever(
            search_type="mmr" if use_mmr else "similarity",
            search_kwargs={"k": 6, "fetch_k": 10} if use_mmr else {"k": 5}
        )

    def get_relevant_documents(self, query: str, top_k: int = 5) -> List[Document]:
        """Get relevant documents using hybrid search."""
        return self.retriever.get_relevant_documents(query)

    def update_documents(self, documents: List[Document]):
        """Update the document store with new documents."""
        self.vector_store.add_documents(documents)

    def save_index(self, path: str):
        """Save the FAISS index to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.vector_store.save_local(path)
        return f"Saved FAISS index to {path}"

class AdvancedRAGSystem:
    """Enhanced RAG system with VertexAI integration and image support."""

    def __init__(self,
                 vertex_gen_ai: VertexGenAI = None,
                 faiss_index_path: str = None,
                 enable_ocr: bool = True):

        self.vertex_gen_ai = vertex_gen_ai
        self.enable_ocr = enable_ocr

        self.processor = EnhancedPDFProcessor(
            chunk_size=800,
            chunk_overlap=200,
            enable_ocr=enable_ocr
        )
        self.documents = {}  # Store processed documents
        self.retrievers = {}  # Store retrievers by document

        # Initialize hybrid retriever
        self.hybrid_retriever = None
        self.faiss_index_path = faiss_index_path

        # If we have a FAISS path, initialize the retriever
        if faiss_index_path and os.path.exists(faiss_index_path):
            self.hybrid_retriever = HybridRetriever(
                faiss_index_path=faiss_index_path,
                vertex_gen_ai=vertex_gen_ai
            )

        self.conversation_history = []
        self.conversation_chain = None

        # Initialize conversation chain if we have VertexGenAI
        if vertex_gen_ai:
            self._initialize_conversation_chain()

    def upload_pdf(self, file_path: str) -> str:
        """Process and index a PDF document with image extraction."""
        try:
            # Process the document
            doc = self.processor.process_pdf(file_path)

            # Store the document
            self.documents[doc.filename] = doc

            # Update the retrievers
            self._update_retrievers()

            image_count = len(doc.images)
            return f"Processed and indexed {doc.filename} ({len(doc.pages)} pages, {len(doc.chunks)} chunks, {image_count} images extracted)"
        except Exception as e:
            logger.error(f"Error uploading PDF: {str(e)}")
            raise

    def _update_retrievers(self):
        """Update the retriever system with current documents."""
        all_docs = []
        for doc in self.documents.values():
            all_docs.extend(doc.langchain_documents)

        if not all_docs:
            return

        if not self.hybrid_retriever:
            self.hybrid_retriever = HybridRetriever(
                all_docs,
                vertex_gen_ai=self.vertex_gen_ai
            )
        else:
            self.hybrid_retriever.update_documents(all_docs)

        self._initialize_conversation_chain()

    def _create_custom_chain(self, retriever):
        """Create a custom chain that handles our specific prompt format."""
        
        def format_docs(docs):
            formatted = []
            for doc in docs:
                content = doc.page_content
                metadata = doc.metadata
                
                # Include image references if available
                image_note = ""
                if metadata.get('image_refs'):
                    image_note = f"\nImage References: {metadata['image_refs']}"
                    
                formatted_doc = f"""
Document: {metadata.get('source', 'Unknown')}
Pages: {metadata.get('page_numbers', [])}
Section: {metadata.get('section_info', {})}
Content: {content}{image_note}
"""
                formatted.append(formatted_doc)
            return "\n\n---\n\n".join(formatted)
        
        def custom_chain(inputs):
            try:
                # Get relevant documents
                docs = retriever.get_relevant_documents(inputs["question"])
                
                # Format context
                context = format_docs(docs)
                
                # Format chat history
                chat_history = ""
                for q, a in self.conversation_history[-5:]:  # Last 5 exchanges
                    chat_history += f"Human: {q}\nAssistant: {a}\n\n"
                
                # Create prompt
                prompt = f"""You are a precise document assistant with perfect knowledge of the provided document information.
Your goal is to give accurate, thorough answers with exact document and page citations.

IMPORTANT RULES:
1. Use ONLY the information from the retrieved context below to answer the question
2. Do NOT use any prior knowledge or information not present in the context
3. If the answer cannot be found in the context, clearly state: "I cannot find information about this in the provided documents."
4. Provide exact citations for every piece of information

Retrieved Context:
{context}

{f"Previous Conversation:{chr(10)}{chat_history}" if chat_history else ""}

User Question: {inputs["question"]}

Citation Format Requirements:
1. Begin EACH piece of information with: [Document: "filename.pdf", Page X-Y]
2. Include section if available: [Document: "filename.pdf", Page X, Section Y: "Title"]
3. For OCR-extracted content: [Document: "filename.pdf", Page X, Image/Figure]
4. Be extremely specific about page numbers
5. Never make up or guess page numbers

Answer the question using ONLY the provided context:"""
                
                # Generate response using VertexGenAI
                response_text = self.vertex_gen_ai.generate_content(prompt)
                
                if not response_text:
                    response_text = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
                
                return {
                    "answer": response_text,
                    "source_documents": docs
                }
                
            except Exception as e:
                logger.error(f"Error in custom chain: {str(e)}")
                return {
                    "answer": f"An error occurred while processing your question: {str(e)}",
                    "source_documents": []
                }
        
        return custom_chain

    def _initialize_conversation_chain(self):
        """Initialize the conversation chain with VertexGenAI."""
        if not self.hybrid_retriever or not self.vertex_gen_ai:
            logger.warning("Missing retriever or VertexGenAI instance")
            return

        # Create a custom chain since VertexGenAI has a different interface
        self.conversation_chain = self._create_custom_chain(self.hybrid_retriever.retriever)

    def ask(self, query: str) -> Dict[str, Any]:
        """Process a query and return answer with citations and relevant images."""
        if not self.hybrid_retriever or not self.conversation_chain:
            return {"answer": "Please upload at least one document first."}

        try:
            # Execute the chain
            result = self.conversation_chain({"question": query})

            # Extract answer
            answer = result.get("answer", "No answer generated")
            source_docs = result.get("source_documents", [])

            # Find relevant images from source documents
            relevant_images = []
            for doc in source_docs:
                metadata = doc.metadata
                doc_filename = metadata.get("source", "")
                image_refs = metadata.get("image_refs", [])
                
                # Get the document
                if doc_filename in self.documents:
                    pdf_doc = self.documents[doc_filename]
                    for img_idx in image_refs:
                        if img_idx < len(pdf_doc.images):
                            img_data = pdf_doc.images[img_idx]
                            relevant_images.append({
                                "document": doc_filename,
                                "page": img_data.page_number,
                                "base64": img_data.base64_string,
                                "ocr_text": img_data.ocr_text[:100] + "..." if len(img_data.ocr_text) > 100 else img_data.ocr_text
                            })

            # Extract citation info
            citation_info = self._extract_cited_pages(answer)

            # Add to conversation history
            self.conversation_history.append((query, answer))

            # Get all cited pages
            all_pages = []
            for doc_citations in citation_info.values():
                for citation in doc_citations:
                    all_pages.extend(citation["pages"])

            return {
                "answer": answer,
                "citation_info": citation_info,
                "cited_pages": sorted(list(set(all_pages))),
                "source_docs": source_docs,
                "relevant"relevant_images": relevant_images
           }
       except Exception as e:
           logger.error(f"Error in ask method: {str(e)}")
           return {
               "answer": f"Error processing query: {str(e)}",
               "citation_info": {},
               "cited_pages": [],
               "source_docs": [],
               "relevant_images": []
           }

   def _extract_cited_pages(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
       """Extract citation information from the text."""
       if not text:
           return {}

       citations = {}

       # Updated pattern to handle image citations
       doc_pattern = r'\[Document:\s*"([^"]+)",\s*Page(?:s)?\s*(\d+)(?:-(\d+))?(?:,\s*(?:Section\s*([^:]+):\s*"([^"]+)"|Image/Figure))?\]'

       try:
           for match in re.finditer(doc_pattern, text):
               try:
                   doc_name = match.group(1)
                   start_page = int(match.group(2))
                   end_page = int(match.group(3)) if match.group(3) else start_page
                   pages = list(range(start_page, end_page + 1))

                   section_num = match.group(4) if match.group(4) else None
                   section_title = match.group(5) if match.group(5) else None

                   if doc_name not in citations:
                       citations[doc_name] = []

                   citations[doc_name].append({
                       "pages": pages,
                       "section_num": section_num,
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
       return "Conversation history has been reset."

   def get_document_stats(self) -> Dict[str, Dict[str, Any]]:
       """Get statistics about uploaded documents."""
       stats = {}
       for name, doc in self.documents.items():
           stats[name] = {
               "pages": len(doc.pages),
               "chunks": len(doc.chunks),
               "total_chars": len(doc.content),
               "images": len(doc.images),
               "images_with_ocr": sum(1 for img in doc.images if img.ocr_text)
           }
       return stats

   def save_index(self, path: str):
       """Save the FAISS index to disk."""
       if self.hybrid_retriever:
           return self.hybrid_retriever.save_index(path)
       return "No index to save. Please upload documents first."

class GradioRAGInterface:
   """Gradio interface for the RAG system with image display support."""

   def __init__(self):
       self.rag_system = None
       self.temp_dir = tempfile.mkdtemp()
       self.vertex_gen_ai = None
       self.setup_interface()

   def setup_interface(self):
       """Set up the Gradio interface."""
       with gr.Blocks(theme=gr.themes.Base()) as self.interface:
           gr.Markdown("# Advanced PDF Chat with Document, Page, Section Citations and Image References")

           with gr.Row():
               with gr.Column(scale=1):
                   # System Configuration
                   gr.Markdown("## 🤖 System Configuration")

                   # VertexAI Settings
                   with gr.Group():
                       ca_bundle_path = gr.Textbox(
                           label="CA Bundle Path (optional)",
                           placeholder="Path to PROD CA pem file",
                           value=""
                       )
                       initialize_vertex_btn = gr.Button("Initialize VertexAI", variant="primary")
                       vertex_status = gr.Textbox(label="VertexAI Status", value="Not initialized")

                   # Vector DB settings
                   gr.Markdown("## 💾 Vector Database Settings")
                   use_existing_db = gr.Checkbox(label="Use Existing FAISS Index", value=False)

                   with gr.Group(visible=False) as faiss_settings:
                       faiss_path = gr.Textbox(
                           label="FAISS Index Path",
                           placeholder="Path to existing FAISS index"
                       )

                   # OCR Settings
                   gr.Markdown("## 🔍 OCR Settings")
                   enable_ocr = gr.Checkbox(label="Enable OCR for Images", value=True)

                   initialize_btn = gr.Button("Initialize System")

                   # Document upload section
                   gr.Markdown("## 📄 Upload Documents")
                   pdf_upload = gr.File(
                       label="Upload PDF Documents",
                       file_types=[".pdf"],
                       file_count="multiple"
                   )
                   upload_button = gr.Button("Process PDFs")
                   save_index_button = gr.Button("Save FAISS Index")
                   faiss_save_path = gr.Textbox(
                       label="Save FAISS Index To",
                       placeholder="Path to save the current FAISS index",
                       value="/content/faiss_index"
                   )
                   doc_status = gr.Markdown("No documents uploaded yet.")

                   # System controls
                   gr.Markdown("## ⚙️ System Controls")
                   reset_btn = gr.Button("Reset Conversation")

               with gr.Column(scale=2):
                   # Chat interface
                   gr.Markdown("## 💬 Chat With Your Documents")
                   chatbot = gr.Chatbot(height=400, elem_id="chatbot")
                   msg = gr.Textbox(
                       placeholder="Ask a question about your documents...",
                       label="Your Question",
                       lines=1
                   )
                   submit_btn = gr.Button("Ask", variant="primary")

                   # Citation information with enhanced display
                   gr.Markdown("## 📊 Citation Information")
                   with gr.Accordion("Document References", open=True):
                       citation_display = gr.JSON(label="Citations by Document")

                   # Image references
                   gr.Markdown("## 🖼️ Referenced Images")
                   with gr.Accordion("Images from Documents", open=True):
                       image_gallery = gr.Gallery(
                           label="Relevant Images",
                           show_label=True,
                           elem_id="gallery",
                           columns=2,
                           rows=2,
                           height="auto"
                       )
                       image_info = gr.JSON(label="Image Information")

                   with gr.Accordion("Document Statistics", open=False):
                       doc_stats = gr.JSON(label="Document Statistics")

           # Set up event handlers for showing/hiding FAISS settings
           use_existing_db.change(
               fn=lambda x: gr.update(visible=x),
               inputs=[use_existing_db],
               outputs=[faiss_settings]
           )

           # Initialize VertexAI button
           initialize_vertex_btn.click(
               fn=self.initialize_vertex_ai,
               inputs=[ca_bundle_path],
               outputs=[vertex_status]
           )

           # Initialize system button logic
           initialize_btn.click(
               fn=self.initialize_system,
               inputs=[
                   use_existing_db,
                   faiss_path,
                   enable_ocr
               ],
               outputs=[doc_status]
           )

           # Save FAISS index button
           save_index_button.click(
               fn=self.save_faiss_index,
               inputs=[faiss_save_path],
               outputs=[doc_status]
           )

           # Add event handlers for existing functionality
           upload_button.click(
               fn=self.upload_documents,
               inputs=[pdf_upload],
               outputs=[doc_status, doc_stats]
           )

           # Chat handlers with image support
           def safe_chat_handler(message, history):
               try:
                   if history is None:
                       history = []
                   result = self.chat_with_docs(message, history)
                   return result
               except Exception as e:
                   logger.error(f"Error in chat handler: {str(e)}")
                   if history is None:
                       history = []
                   return history + [[message, f"Error: {str(e)}"]], "", {}, [], {}

           submit_btn.click(
               fn=safe_chat_handler,
               inputs=[msg, chatbot],
               outputs=[chatbot, msg, citation_display, image_gallery, image_info]
           )

           msg.submit(
               fn=safe_chat_handler,
               inputs=[msg, chatbot],
               outputs=[chatbot, msg, citation_display, image_gallery, image_info]
           )

           reset_btn.click(
               fn=self.reset_chat,
               inputs=[],
               outputs=[chatbot, doc_status, citation_display, image_gallery, image_info]
           )

   def initialize_vertex_ai(self, ca_bundle_path):
       """Initialize VertexAI instance."""
       try:
           # Set the CA bundle if provided
           if ca_bundle_path and os.path.exists(ca_bundle_path):
               os.environ["REQUESTS_CA_BUNDLE"] = ca_bundle_path
               logger.info(f"Set CA bundle path to: {ca_bundle_path}")
               
           self.vertex_gen_ai = VertexGenAI()
           
           # Test the connection with both generation and embedding
           test_prompt = "Hello, this is a test."
           test_response = self.vertex_gen_ai.generate_content(test_prompt)
           test_embedding = self.vertex_gen_ai.get_embeddings([test_prompt])
           
           if test_response and test_embedding:
               return "✅ VertexAI initialized successfully! Both generation and embeddings are working."
           elif test_response:
               return "⚠️ VertexAI generation works but embeddings failed. Check your embedding model access."
           elif test_embedding:
               return "⚠️ VertexAI embeddings work but generation failed. Check your generative model access."
           else:
               return "❌ VertexAI initialized but both generation and embeddings failed. Check your credentials and model access."
               
       except Exception as e:
           return f"❌ Failed to initialize VertexAI: {str(e)}"

   def initialize_system(self, use_existing_db, faiss_path, enable_ocr):
       """Initialize the RAG system with settings."""

       if not self.vertex_gen_ai:
           return "Please initialize VertexAI first."

       # Validate FAISS path if using existing DB
       if use_existing_db and (not faiss_path or not os.path.exists(faiss_path)):
           return "Please enter a valid path to an existing FAISS index."

       try:
           # Create the RAG system with appropriate settings
           self.rag_system = AdvancedRAGSystem(
               vertex_gen_ai=self.vertex_gen_ai,
               faiss_index_path=faiss_path if use_existing_db else None,
               enable_ocr=enable_ocr
           )

           ocr_status = "enabled" if enable_ocr else "disabled"
           if use_existing_db:
               return f"✅ System initialized with VertexAI using existing FAISS index at {faiss_path}! OCR is {ocr_status}."
           else:
               return f"✅ System initialized with VertexAI! OCR is {ocr_status}. You can now upload documents."

       except Exception as e:
           return f"❌ Error initializing system: {str(e)}"

   def save_faiss_index(self, save_path):
       """Save the current FAISS index to disk."""
       if not self.rag_system or not self.rag_system.hybrid_retriever:
           return "Please initialize the system and upload documents first."

       if not save_path:
           return "Please provide a valid path to save the FAISS index."

       try:
           result = self.rag_system.hybrid_retriever.save_index(save_path)
           return f"✅ {result}"
       except Exception as e:
           return f"❌ Error saving FAISS index: {str(e)}"

   def upload_documents(self, files):
       """Upload and process documents."""
       if not self.rag_system:
           return "Please initialize the system first.", {}

       if not files:
           return "No files selected for upload.", {}

       results = []

       for file in files:
           try:
               result = self.rag_system.upload_pdf(file.name)
               results.append(f"✅ {result}")
           except Exception as e:
               results.append(f"❌ Error processing {os.path.basename(file.name)}: {str(e)}")

       # Get enhanced document stats for display
       doc_stats = self._get_enhanced_doc_stats()

       status = "## Document Processing Results\n" + "\n".join(results)
       return status, doc_stats

   def _get_enhanced_doc_stats(self):
       """Get enhanced statistics about uploaded documents including image information."""
       if not self.rag_system:
           return {}

       stats = {}
       for name, doc in self.rag_system.documents.items():
           # Count sections found per document
           sections = set()
           for chunk in doc.chunks:
               for section_key, section_title in chunk.section_info.items():
                   sections.add(f"{section_key}: {section_title}")

           # Image statistics
           images_with_text = sum(1 for img in doc.images if img.ocr_text)

           stats[name] = {
               "Total Pages": len(doc.pages),
               "Total Chunks": len(doc.chunks),
               "Characters": len(doc.content),
               "Sections Found": len(sections),
               "Section Examples": list(sections)[:5] if sections else "None found",
               "Total Images": len(doc.images),
               "Images with OCR Text": images_with_text
           }
       return stats

   def chat_with_docs(self, message, history=None):
       """Process a chat message and get a response with images."""
       if history is None:
           history = []

       if not self.rag_system:
           return history + [[message, "Please initialize the system and upload documents first."]], "", {}, [], {}

       if not self.rag_system.documents:
           return history + [[message, "Please upload at least one document before asking questions."]], "", {}, [], {}

       if not message or not message.strip():
           return history + [[message, "Please enter a question."]], "", {}, [], {}

       try:
           # Process the query
           start_time = time.time()
           response = self.rag_system.ask(message)
           process_time = time.time() - start_time

           answer = response["answer"]
           citation_info = response.get("citation_info", {})
           relevant_images = response.get("relevant_images", [])

           # Add processing time info
           answer += f"\n\n_Query processed in {process_time:.2f} seconds._"

           # Update chatbot history
           updated_history = list(history)
           updated_history.append([message, answer])

           # Format citation info for display
           formatted_citations = {}
           for doc_name, citations in citation_info.items():
               doc_citations = []
               for citation in citations:
                   pages_str = "-".join(map(str, [min(citation["pages"]), max(citation["pages"])])) if len(citation["pages"]) > 1 else str(citation["pages"][0])
                   section_info = ""
                   if citation["section_num"] and citation["section_title"]:
                       section_info = f", Section {citation['section_num']}: \"{citation['section_title']}\""
                   elif citation["section_num"]:
                       section_info = f", Section {citation['section_num']}"
                   elif citation["section_title"]:
                       section_info = f", Section: \"{citation['section_title']}\""

                   doc_citations.append(f"Page(s) {pages_str}{section_info}")

               formatted_citations[doc_name] = doc_citations

           # Prepare images for gallery
           gallery_images = []
           image_details = []

           for img_data in relevant_images:
               # Convert base64 to PIL Image for gallery
               img_bytes = base64.b64decode(img_data["base64"])
               img = Image.open(BytesIO(img_bytes))
               
               # Add to gallery
               gallery_images.append(img)
               
               # Add details
               image_details.append({
                   "document": img_data["document"],
                   "page": img_data["page"],
                   "ocr_preview": img_data["ocr_text"]
               })

           # Return results
           page_info = {
               "Citation Information": formatted_citations,
               "Documents Referenced": len(citation_info),
               "Images Found": len(relevant_images)
           }

           return updated_history, "", page_info, gallery_images, {"images": image_details}

       except Exception as e:
           logger.error(f"Error in chat: {str(e)}")
           return history + [[message, f"Error: {str(e)}"]], "", {}, [], {}

   def reset_chat(self):
       """Reset the chat conversation."""
       if self.rag_system:
           self.rag_system.reset_conversation()

       doc_status = "Conversation has been reset."
       if self.rag_system and self.rag_system.documents:
           doc_names = list(self.rag_system.documents.keys())
           doc_status += f"\nLoaded documents: {', '.join(doc_names)}"

       return [], doc_status, {}, [], {}

# Helper function for Google Authentication
def setup_google_auth():
   """Set up Google authentication if using Vertex AI."""
   try:
       from google.colab import auth
       auth.authenticate_user()
       print("Google authentication completed successfully.")
       return True
   except ImportError:
       print("Not running in Google Colab. Please ensure you're authenticated for Vertex AI.")
       return False

# Export the interface for easy import in Colab
def launch_rag_interface():
   """Launch the RAG interface."""
   interface = GradioRAGInterface()
   return interface.interface

# Main entry point for running in Colab
if __name__ == "__main__":
   print("Starting Advanced PDF RAG System with VertexAI and Image Support")
   
   # Check if we're in Colab
   try:
       import google.colab
       print("Running in Google Colab environment")

       # Set up authentication
       print("\n== Setting up Google Authentication ==")
       if setup_google_auth():
           print("Authentication successful!")
       
       # Create directories
       os.makedirs("/content/faiss_index", exist_ok=True)
       print("Created directory for FAISS indices at /content/faiss_index")

       print("\n== IMPORTANT NOTES ==")
       print("1. Click 'Initialize VertexAI' button first")
       print("2. Optionally provide CA Bundle path if needed")
       print("3. Make sure your Google Cloud project has Vertex AI API enabled")
       print("4. The system will use your project's Vertex AI for embeddings and generation")
       print("5. Images in PDFs will be extracted and OCR will be performed if enabled")
       
   except ImportError:
       print("Not running in Google Colab environment")

   # Launch the interface
   interface = launch_rag_interface()
   print("\nRAG System is running. Access the interface through the provided URL.")
