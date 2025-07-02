import os
import gradio as gr
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json
import re
from dataclasses import dataclass, field
from sklearn.metrics.pairwise import cosine_similarity
import logging
from PIL import Image
import io
import base64
import hashlib
import tempfile
import shutil
import time
from tqdm import tqdm

# Your existing VertexAI code
import vertexai
from google.oauth2.credentials import Credentials
from helpers import get_coin_token
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        model = TextEmbeddingModel.from_pretrained(model_name, metadata=self.metadata)
        
        if not texts:
            return []
        
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = model.get_embeddings(texts, metadata=self.metadata)
        return embeddings

@dataclass
class PageChunk:
    """Store chunk information with precise page tracking."""
    text: str
    page_numbers: List[int]
    start_char_idx: int
    end_char_idx: int
    filename: str
    section_info: Dict[str, str] = field(default_factory=dict)
    source_type: str = "text"  # 'text' or 'image'
    image_description: str = ""
    bbox: tuple = None
    image_path: str = None
    chunk_id: str = ""
    embedding: List[float] = None

@dataclass
class PDFDocument:
    """Enhanced class to store PDF document metadata and content."""
    filename: str
    content: str = ""
    pages: Dict[int, str] = field(default_factory=dict)
    char_to_page_map: List[int] = field(default_factory=list)
    chunks: List[PageChunk] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.content)

class EnhancedPDFProcessor:
    """Advanced PDF processor with improved chunking and page tracking."""

    def __init__(self, vertex_ai: VertexGenAI, chunk_size: int = 800, chunk_overlap: int = 200):
        self.vertex_ai = vertex_ai
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.temp_dir = tempfile.mkdtemp()

        # Regex patterns for section detection
        self.section_patterns = [
            r'(?:Section|SECTION|Chapter|CHAPTER)\s+(\d+(?:\.\d+)*)[:\s]+(.*?)(?=\n|$)',
            r'(\d+(?:\.\d+)*)\s+(.*?)(?=\n|$)',
            r'(?:\n|\A)([A-Z][A-Z\s]+)(?:\n|:)'
        ]

    def extract_text_from_pdf(self, file_path: str) -> PDFDocument:
        """Extract text from PDF with enhanced page-level tracking."""
        doc = PDFDocument(filename=os.path.basename(file_path))
        full_text = ""
        char_to_page = []

        try:
            pdf_doc = fitz.open(file_path)
            
            if len(pdf_doc.pages) == 0:
                logger.warning(f"PDF {file_path} has no pages.")
                return doc

            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                page_number = page_num + 1

                # Extract text
                page_text = page.extract_text() or ""
                page_text = self._clean_pdf_text(page_text)

                if page_text.strip():
                    doc.pages[page_number] = page_text
                    full_text += page_text
                    char_to_page.extend([page_number] * len(page_text))

                # Extract images
                image_chunks = self._extract_images_with_ai(page, page_number, file_path)
                doc.chunks.extend(image_chunks)

            doc.content = full_text
            doc.char_to_page_map = char_to_page
            pdf_doc.close()
            return doc

        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            raise

    def _extract_images_with_ai(self, page, page_num: int, pdf_path: str) -> List[PageChunk]:
        """Extract images and analyze them using Vertex AI Vision."""
        chunks = []
        image_list = page.get_images()

        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)
                
                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    img_data = pix.tobytes("png")
                    
                    # Save image to temporary file
                    img_filename = f"page_{page_num}_img_{img_index}.png"
                    img_path = os.path.join(self.temp_dir, img_filename)
                    
                    with open(img_path, "wb") as f:
                        f.write(img_data)
                    
                    # Get image description using Vertex AI
                    img_description = self._get_ai_image_description(img_data)
                    
                    # Get bounding box
                    bbox = page.get_image_bbox(img)
                    
                    chunk_id = f"img_{page_num}_{img_index}"
                    chunk = PageChunk(
                        text=img_description,
                        page_numbers=[page_num],
                        start_char_idx=0,
                        end_char_idx=len(img_description),
                        filename=os.path.basename(pdf_path),
                        section_info={"section": "Image Content"},
                        source_type="image",
                        image_description=img_description,
                        bbox=bbox,
                        image_path=img_path,
                        chunk_id=chunk_id
                    )
                    chunks.append(chunk)
                
                pix = None
            except Exception as e:
                logger.error(f"Error processing image {img_index} on page {page_num}: {e}")
                continue

        return chunks

    def _get_ai_image_description(self, img_data: bytes) -> str:
        """Get image description using Vertex AI Vision."""
        try:
            # Convert to base64 for Vertex AI
            img_base64 = base64.b64encode(img_data).decode()
            
            prompt = """Analyze this image thoroughly and provide a detailed description including:
            1. All visible text content (OCR)
            2. Charts, graphs, diagrams, and their data
            3. Tables and their structure
            4. Key visual elements and their relationships
            5. Context and purpose of the image
            6. Any technical or scientific content
            
            Be comprehensive and include all readable text and data points."""
            
            # Use Vertex AI to analyze the image
            description = self.vertex_ai.generate_content(f"{prompt}\n\nDescribe this image in detail.")
            return description or "Image content could not be analyzed"
            
        except Exception as e:
            logger.error(f"Error getting AI image description: {e}")
            return "Image content analysis failed"

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

    def _semantic_chunking(self, text: str) -> List[str]:
        """Advanced semantic chunking with sentence boundary preservation."""
        if not text.strip():
            return []
            
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _generate_chunk_id(self, text: str, page_num: int) -> str:
        """Generate unique chunk ID."""
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"chunk_{page_num}_{text_hash}"

    def chunk_document(self, doc: PDFDocument) -> PDFDocument:
        """Chunk document with enhanced page number tracking and section detection."""
        if not doc.content or not doc.char_to_page_map:
            logger.warning(f"Document {doc.filename} has no content to chunk.")
            return doc

        try:
            # Get semantic chunks
            semantic_chunks = self._semantic_chunking(doc.content)
            
            for i, chunk_text in enumerate(semantic_chunks):
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
                    logger.warning(f"No pages found for chunk {i}. Assigning to page 1.")

                # Extract section information
                section_info = self._extract_section_info(chunk_text)
                chunk_id = self._generate_chunk_id(chunk_text, min(chunk_pages))

                # Create PageChunk
                page_chunk = PageChunk(
                    text=chunk_text,
                    page_numbers=sorted(list(chunk_pages)),
                    start_char_idx=start_pos,
                    end_char_idx=end_pos,
                    filename=doc.filename,
                    section_info=section_info,
                    source_type="text",
                    chunk_id=chunk_id
                )

                doc.chunks.append(page_chunk)

        except Exception as e:
            logger.error(f"Error chunking document: {str(e)}")

        return doc

    def process_pdf(self, file_path: str) -> PDFDocument:
        """Process PDF file in a single call."""
        doc = self.extract_text_from_pdf(file_path)
        return self.chunk_document(doc)

class VertexAIRetriever:
    """Pure Vertex AI-based retrieval system."""

    def __init__(self, vertex_ai: VertexGenAI):
        self.vertex_ai = vertex_ai
        self.chunks = []
        self.embeddings_matrix = None

    def add_chunks(self, chunks: List[PageChunk]):
        """Add chunks and generate embeddings using Vertex AI."""
        if not chunks:
            return

        self.chunks.extend(chunks)
        
        # Generate embeddings for new chunks
        texts = [chunk.text for chunk in chunks]
        try:
            embeddings = self.vertex_ai.get_embeddings(texts)
            
            # Store embeddings in chunks
            for i, chunk in enumerate(chunks):
                if i < len(embeddings):
                    chunk.embedding = embeddings[i].values
            
            # Update embeddings matrix
            self._update_embeddings_matrix()
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")

    def _update_embeddings_matrix(self):
        """Update the embeddings matrix with current chunks."""
        valid_embeddings = [chunk.embedding for chunk in self.chunks if chunk.embedding is not None]
        if valid_embeddings:
            self.embeddings_matrix = np.array(valid_embeddings)

    def hybrid_retrieval(self, query: str, top_k: int = 10) -> List[Tuple[PageChunk, float]]:
        """Retrieve relevant chunks using hybrid approach."""
        if not self.chunks or self.embeddings_matrix is None:
            return []

        try:
            # Get query embedding
            query_embedding = self.vertex_ai.get_embeddings([query])[0].values
            query_embedding = np.array(query_embedding).reshape(1, -1)

            # Semantic similarity
            semantic_scores = cosine_similarity(query_embedding, self.embeddings_matrix)[0]

            # Keyword matching
            keyword_scores = self._compute_keyword_scores(query)

            # Combine scores
            combined_scores = 0.7 * semantic_scores + 0.3 * keyword_scores

            # Get top candidates
            top_indices = np.argsort(combined_scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if idx < len(self.chunks):
                    results.append((self.chunks[idx], combined_scores[idx]))

            return results

        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            return []

    def _compute_keyword_scores(self, query: str) -> np.ndarray:
        """Compute keyword-based scores."""
        query_words = set(query.lower().split())
        scores = []

        for chunk in self.chunks:
            chunk_words = set(chunk.text.lower().split())
            intersection = len(query_words.intersection(chunk_words))
            union = len(query_words.union(chunk_words))
            score = intersection / union if union > 0 else 0
            scores.append(score)

        return np.array(scores)

class AdvancedRAGSystem:
    """Advanced RAG system using only Vertex AI."""

    def __init__(self):
        try:
            os.environ["REQUESTS_CA_BUNDLE"] = "<Path to PROD CA pem file>"
            self.vertex_ai = VertexGenAI()
            self.pdf_processor = EnhancedPDFProcessor(self.vertex_ai)
            self.retriever = VertexAIRetriever(self.vertex_ai)
            self.documents = {}
            self.conversation_history = []
            logger.info("AdvancedRAGSystem initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing AdvancedRAGSystem: {e}")
            raise

    def upload_pdf(self, file_path: str) -> str:
        """Process and index a PDF document."""
        try:
            # Process the document
            doc = self.pdf_processor.process_pdf(file_path)
            
            # Store the document
            self.documents[doc.filename] = doc
            
            # Add chunks to retriever
            self.retriever.add_chunks(doc.chunks)
            
            return f"✅ Processed {doc.filename}: {len(doc.pages)} pages, {len(doc.chunks)} chunks"
            
        except Exception as e:
            logger.error(f"Error uploading PDF: {str(e)}")
            return f"❌ Error processing {os.path.basename(file_path)}: {str(e)}"

    def ask(self, query: str) -> Dict[str, Any]:
        """Process a query and return response with citations."""
        if not self.retriever.chunks:
            return {"answer": "Please upload at least one document first.", "citation_info": {}, "image_paths": []}

        try:
            # Retrieve relevant chunks
            retrieved_chunks = self.retriever.hybrid_retrieval(query, top_k=5)
            
            if not retrieved_chunks:
                return {
                    "answer": "I couldn't find relevant information to answer your question.",
                    "citation_info": {},
                    "image_paths": []
                }

            # Generate response
            response, citation_info, image_paths = self._generate_response_with_images(query, retrieved_chunks)
            
            # Add to conversation history
            self.conversation_history.append((query, response))
            
            return {
                "answer": response,
                "citation_info": citation_info,
                "image_paths": image_paths
            }

        except Exception as e:
            logger.error(f"Error in ask method: {str(e)}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "citation_info": {},
                "image_paths": []
            }

    def _generate_response_with_images(self, query: str, retrieved_chunks: List[Tuple[PageChunk, float]]) -> Tuple[str, Dict, List[str]]:
        """Generate response with image references."""
        context_parts = []
        citation_info = {}
        image_paths = []

        for i, (chunk, score) in enumerate(retrieved_chunks):
            context_parts.append(f"[Context {i+1}]: {chunk.text}")
            
            # Organize citations by document
            doc_name = chunk.filename
            if doc_name not in citation_info:
                citation_info[doc_name] = []

            # Create citation entry
            citation_entry = {
                "pages": chunk.page_numbers,
                "section_info": chunk.section_info,
                "source_type": chunk.source_type,
                "relevance_score": float(score)
            }

            if chunk.source_type == "image":
                citation_entry["image_description"] = chunk.image_description
                citation_entry["bbox"] = chunk.bbox
                if chunk.image_path and os.path.exists(chunk.image_path):
                    image_paths.append(chunk.image_path)

            citation_info[doc_name].append(citation_entry)

        context = "\n\n".join(context_parts)

        prompt = f"""You are a precise document assistant. Answer the question using ONLY the provided context.

Context from documents:
{context}

Question: {query}

Instructions:
1. Answer based ONLY on the provided context
2. Include specific document, page, and section citations
3. Format citations as: [Document: "filename.pdf", Page X] or [Document: "filename.pdf", Pages X-Y]
4. If information comes from images, mention "from image analysis"
5. If context is insufficient, state clearly
6. Be comprehensive but precise

Answer:"""

        try:
            response = self.vertex_ai.generate_content(prompt)
            return response, citation_info, image_paths
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Error generating response.", citation_info, image_paths

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
        return "Conversation history reset."

    def get_document_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get document statistics."""
        stats = {}
        for name, doc in self.documents.items():
            stats[name] = {
                "pages": len(doc.pages),
                "chunks": len(doc.chunks),
                "total_chars": len(doc.content),
                "text_chunks": len([c for c in doc.chunks if c.source_type == "text"]),
                "image_chunks": len([c for c in doc.chunks if c.source_type == "image"])
            }
        return stats

class ChatbotUI:
    """Gradio interface for the RAG system."""

    def __init__(self):
        self.rag_system = AdvancedRAGSystem()
        self.temp_dir = tempfile.mkdtemp()

    def upload_pdfs(self, files):
        """Upload and process PDF files."""
        if not files:
            return "No files uploaded.", "", []

        results = []
        for file in files:
            try:
                result = self.rag_system.upload_pdf(file.name)
                results.append(result)
            except Exception as e:
                results.append(f"❌ Error processing {os.path.basename(file.name)}: {str(e)}")

        status = "\n".join(results)
        stats = self.rag_system.get_document_stats()
        
        return status, "", []

    def chat(self, message, history):
        """Process chat message."""
        if not message.strip():
            return history, "", []

        history = history or []
        history.append([message, None])

        try:
            start_time = time.time()
            response_data = self.rag_system.ask(message)
            process_time = time.time() - start_time

            answer = response_data["answer"]
            citation_info = response_data.get("citation_info", {})
            image_paths = response_data.get("image_paths", [])

            # Add processing time
            answer += f"\n\n_Processed in {process_time:.2f} seconds._"

            # Update history
            history[-1][1] = answer

            # Format references
            ref_text = self._format_references(citation_info)

            return history, ref_text, image_paths

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            history[-1][1] = error_msg
            return history, "", []

    def _format_references(self, citation_info: Dict) -> str:
        """Format citation information for display."""
        if not citation_info:
            return "No references found."

        ref_text = "## References\n\n"
        ref_count = 1

        for doc_name, citations in citation_info.items():
            ref_text += f"**Document: {doc_name}**\n"
            
            for citation in citations:
                pages = citation["pages"]
                pages_str = f"Page {pages[0]}" if len(pages) == 1 else f"Pages {min(pages)}-{max(pages)}"
                
                ref_text += f"- **Reference {ref_count}:** {pages_str}\n"
                ref_text += f"  - Source Type: {citation['source_type']}\n"
                ref_text += f"  - Relevance Score: {citation['relevance_score']:.3f}\n"
                
                if citation['source_type'] == 'image':
                    ref_text += f"  - Image Description: {citation.get('image_description', 'N/A')}\n"
                    ref_text += f"  - **Image displayed in gallery**\n"
                
                if citation.get('section_info'):
                    section_info = citation['section_info']
                    if section_info:
                        ref_text += f"  - Sections: {', '.join([f'{k}: {v}' for k, v in section_info.items()])}\n"
                
                ref_text += "\n"
                ref_count += 1

        return ref_text

    def clear_chat(self):
        """Clear chat and reset."""
        self.rag_system.reset_conversation()
        return [], "Chat cleared. Upload documents to start.", []

    def create_interface(self):
        """Create the Gradio interface."""
        with gr.Blocks(title="Advanced Vertex AI RAG Chatbot", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 📚 Advanced Vertex AI RAG PDF Chatbot")
            gr.Markdown("Upload PDFs and ask questions. Uses only Vertex AI for embeddings, vision analysis, and response generation.")

            with gr.Row():
                with gr.Column(scale=2):
                    # Upload section
                    with gr.Group():
                        gr.Markdown("### 📄 Upload Documents")
                        file_upload = gr.File(
                            label="Upload PDF Files",
                            file_count="multiple",
                            file_types=[".pdf"],
                            height=120
                        )
                        upload_btn = gr.Button("Process PDFs", variant="primary", size="lg")
                        upload_status = gr.Textbox(
                            label="Processing Status",
                            interactive=False,
                            placeholder="Upload status will appear here..."
                        )

                    # Chat section
                    with gr.Group():
                        gr.Markdown("### 💬 Chat")
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=450,
                            avatar_images=("👤", "🤖"),
                            show_copy_button=True
                        )

                        msg = gr.Textbox(
                            label="Ask about your documents",
                            placeholder="Type your question here...",
                            lines=2
                        )

                        with gr.Row():
                            submit_btn = gr.Button("Send", variant="primary", scale=2)
                            clear_btn = gr.Button("Clear Chat", variant="secondary", scale=1)

                with gr.Column(scale=1):
                    # References section
                    with gr.Group():
                        gr.Markdown("### 📋 References")
                        references = gr.Markdown(
                            value="References will appear here after asking questions.",
                            height=300
                        )

                    # Image gallery
                    with gr.Group():
                        gr.Markdown("### 🖼️ Reference Images")
                        reference_images = gr.Gallery(
                            label="Images from sources",
                            columns=2,
                            rows=3,
                            height=400,
                            show_download_button=True
                        )

            # Event handlers
            upload_btn.click(
                fn=self.upload_pdfs,
                inputs=[file_upload],
                outputs=[upload_status, references, reference_images],
                show_progress=True
            )

            submit_btn.click(
                fn=self.chat,
                inputs=[msg, chatbot],
                outputs=[chatbot, references, reference_images],
                show_progress=True
            ).then(
                lambda: gr.update(value=""),
                outputs=[msg]
            )

            msg.submit(
                fn=self.chat,
                inputs=[msg, chatbot],
                outputs=[chatbot, references, reference_images],
                show_progress=True
            ).then(
                lambda: gr.update(value=""),
                outputs=[msg]
            )

            clear_btn.click(
                fn=self.clear_chat,
                outputs=[chatbot, references, reference_images]
            )

            # Example questions
            gr.Markdown("""
            ### 💡 Example Questions:
            - "What are the main findings in this document?"
            - "Explain the chart on page X"
            - "What does the methodology section say?"
            - "Summarize the key recommendations"
            """)

        return demo

def main():
    """Main function to launch the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        chatbot_ui = ChatbotUI()
        demo = chatbot_ui.create_interface()
        
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            debug=False,
            show_error=True
        )

    except Exception as e:
        logger.error(f"Error launching application: {e}")
        print(f"Failed to launch application: {e}")

if __name__ == "__main__":
    main()
