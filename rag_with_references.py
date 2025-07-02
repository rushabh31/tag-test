import os
import gradio as gr
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import json
import re
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
import logging
from PIL import Image
import io
import base64
from sentence_transformers import CrossEncoder
import hashlib
import tempfile
import shutil

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
        from vertexai.language_models import TextEmbeddingModel
        
        model = TextEmbeddingModel.from_pretrained(model_name, metadata=self.metadata)
        
        if not texts:
            return []
        
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = model.get_embeddings(texts, metadata=self.metadata)
        return embeddings

@dataclass
class DocumentChunk:
    text: str
    page_number: int
    section: str
    subsection: str
    chunk_id: str
    source_type: str  # 'text' or 'image'
    image_description: str = ""
    bbox: tuple = None  # bounding box for images
    embedding: List[float] = None
    image_path: str = None  # Path to saved image file
    image_data: bytes = None  # Raw image data

class AdvancedPDFProcessor:
    def __init__(self, vertex_ai: VertexGenAI):
        self.vertex_ai = vertex_ai
        self.chunks = []
        self.chunk_embeddings = []
        self.temp_dir = tempfile.mkdtemp()  # Directory for storing extracted images
        
    def extract_images_and_text(self, pdf_path: str) -> List[DocumentChunk]:
        """Extract text and images from PDF with detailed metadata"""
        doc = fitz.open(pdf_path)
        chunks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text with structure
            text_chunks = self._extract_structured_text(page, page_num + 1)
            chunks.extend(text_chunks)
            
            # Extract images
            image_chunks = self._extract_images_with_ocr(page, page_num + 1, pdf_path)
            chunks.extend(image_chunks)
        
        doc.close()
        return chunks
    
    def _extract_structured_text(self, page, page_num: int) -> List[DocumentChunk]:
        """Extract text with hierarchical structure detection"""
        blocks = page.get_text("dict")
        chunks = []
        current_section = "Unknown Section"
        current_subsection = "Unknown Subsection"
        
        for block in blocks["blocks"]:
            if "lines" not in block:
                continue
                
            block_text = ""
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text:
                        # Detect headers based on font size and style
                        font_size = span["size"]
                        font_flags = span["flags"]
                        
                        if font_size > 14 or font_flags & 2**4:  # Large or bold text
                            if self._is_section_header(text):
                                current_section = text
                                current_subsection = "Unknown Subsection"
                            elif self._is_subsection_header(text):
                                current_subsection = text
                        
                        block_text += text + " "
            
            if block_text.strip():
                # Apply semantic chunking
                semantic_chunks = self._semantic_chunking(block_text.strip())
                
                for chunk_text in semantic_chunks:
                    chunk_id = self._generate_chunk_id(chunk_text, page_num)
                    chunk = DocumentChunk(
                        text=chunk_text,
                        page_number=page_num,
                        section=current_section,
                        subsection=current_subsection,
                        chunk_id=chunk_id,
                        source_type="text"
                    )
                    chunks.append(chunk)
        
        return chunks
    
    def _extract_images_with_ocr(self, page, page_num: int, pdf_path: str) -> List[DocumentChunk]:
        """Extract images and perform OCR - Enhanced to save actual images"""
        chunks = []
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            try:
                # Get image
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)
                
                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    img_data = pix.tobytes("png")
                    
                    # Save image to temporary file
                    img_filename = f"page_{page_num}_img_{img_index}.png"
                    img_path = os.path.join(self.temp_dir, img_filename)
                    
                    with open(img_path, "wb") as f:
                        f.write(img_data)
                    
                    # Convert to PIL Image for processing
                    pil_image = Image.open(io.BytesIO(img_data))
                    
                    # Get image description using Vision API
                    img_description = self._get_image_description(pil_image)
                    
                    # Get bounding box
                    bbox = page.get_image_bbox(img)
                    
                    chunk_id = f"img_{page_num}_{img_index}"
                    chunk = DocumentChunk(
                        text=img_description,
                        page_number=page_num,
                        section="Image Content",
                        subsection=f"Image {img_index + 1}",
                        chunk_id=chunk_id,
                        source_type="image",
                        image_description=img_description,
                        bbox=bbox,
                        image_path=img_path,
                        image_data=img_data
                    )
                    chunks.append(chunk)
                
                pix = None
            except Exception as e:
                logging.error(f"Error processing image {img_index} on page {page_num}: {e}")
                continue
        
        return chunks
    
    def _get_image_description(self, image: Image.Image) -> str:
        """Get image description using Vertex AI Vision"""
        try:
            # Convert PIL image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            prompt = """Analyze this image and provide a detailed description including:
            1. Main objects, text, charts, or diagrams visible
            2. Any readable text content
            3. Context and purpose of the image
            4. Key information that might be relevant for document search
            
            Be comprehensive but concise."""
            
            # Enhanced prompt with image analysis
            full_prompt = f"{prompt}\n\nImage Analysis: Describe what you see in this image in detail."
            description = self.vertex_ai.generate_content(full_prompt)
            return description or "Image content could not be analyzed"
            
        except Exception as e:
            logging.error(f"Error getting image description: {e}")
            return "Image content analysis failed"
    
    def _semantic_chunking(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """Advanced semantic chunking with sentence boundary preservation"""
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with improved accuracy"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _is_section_header(self, text: str) -> bool:
        """Detect if text is a section header"""
        patterns = [
            r'^\d+\.\s+[A-Z]',  # 1. Introduction
            r'^[A-Z][A-Z\s]{2,}$',  # ALL CAPS
            r'^Chapter\s+\d+',  # Chapter 1
            r'^Section\s+\d+',  # Section 1
        ]
        return any(re.match(pattern, text) for pattern in patterns)
    
    def _is_subsection_header(self, text: str) -> bool:
        """Detect if text is a subsection header"""
        patterns = [
            r'^\d+\.\d+\s+[A-Z]',  # 1.1 Subsection
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+',  # Title Case
        ]
        return any(re.match(pattern, text) for pattern in patterns)
    
    def _generate_chunk_id(self, text: str, page_num: int) -> str:
        """Generate unique chunk ID"""
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"chunk_{page_num}_{text_hash}"

class AdvancedRAGSystem:
    def __init__(self, vertex_ai: VertexGenAI):
        self.vertex_ai = vertex_ai
        self.pdf_processor = AdvancedPDFProcessor(vertex_ai)
        self.chunks = []
        self.embeddings_matrix = None
        try:
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-2-v2')
        except Exception as e:
            logging.warning(f"Could not load cross-encoder: {e}")
            self.cross_encoder = None
        
    def process_pdfs(self, pdf_files: List[str]) -> str:
        """Process multiple PDF files"""
        all_chunks = []
        
        for pdf_file in pdf_files:
            try:
                chunks = self.pdf_processor.extract_images_and_text(pdf_file)
                all_chunks.extend(chunks)
            except Exception as e:
                logging.error(f"Error processing {pdf_file}: {e}")
                continue
        
        if not all_chunks:
            return "No content could be extracted from the uploaded PDFs."
        
        # Generate embeddings for all chunks
        texts = [chunk.text for chunk in all_chunks]
        try:
            embeddings = self.vertex_ai.get_embeddings(texts)
            
            # Store embeddings in chunks
            for i, chunk in enumerate(all_chunks):
                if i < len(embeddings):
                    chunk.embedding = embeddings[i].values
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            return f"Error generating embeddings: {str(e)}"
        
        self.chunks = all_chunks
        valid_embeddings = [chunk.embedding for chunk in all_chunks if chunk.embedding is not None]
        
        if valid_embeddings:
            self.embeddings_matrix = np.array(valid_embeddings)
        else:
            return "No valid embeddings could be generated."
        
        return f"Successfully processed {len(pdf_files)} PDFs with {len(all_chunks)} chunks extracted."
    
    def hybrid_retrieval(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """Hybrid retrieval combining semantic search and keyword matching"""
        if not self.chunks or self.embeddings_matrix is None:
            return []
        
        try:
            # Get query embedding
            query_embedding = self.vertex_ai.get_embeddings([query])[0].values
            query_embedding = np.array(query_embedding).reshape(1, -1)
            
            # Semantic similarity
            semantic_scores = cosine_similarity(query_embedding, self.embeddings_matrix)[0]
            
            # Keyword matching score
            keyword_scores = self._compute_keyword_scores(query)
            
            # Combine scores (weighted)
            combined_scores = 0.7 * semantic_scores + 0.3 * keyword_scores
            
            # Get top candidates
            top_indices = np.argsort(combined_scores)[::-1][:top_k * 2]  # Get more for reranking
            
            candidates = [(self.chunks[idx], combined_scores[idx]) for idx in top_indices if idx < len(self.chunks)]
            
            # Rerank using cross-encoder if available
            if self.cross_encoder:
                reranked_candidates = self._rerank_with_cross_encoder(query, candidates)
            else:
                reranked_candidates = candidates
            
            return reranked_candidates[:top_k]
        
        except Exception as e:
            logging.error(f"Error in hybrid retrieval: {e}")
            return []
    
    def _compute_keyword_scores(self, query: str) -> np.ndarray:
        """Compute keyword-based scores for all chunks"""
        query_words = set(query.lower().split())
        scores = []
        
        for chunk in self.chunks:
            chunk_words = set(chunk.text.lower().split())
            # Jaccard similarity
            intersection = len(query_words.intersection(chunk_words))
            union = len(query_words.union(chunk_words))
            score = intersection / union if union > 0 else 0
            scores.append(score)
        
        return np.array(scores)
    
    def _rerank_with_cross_encoder(self, query: str, candidates: List[Tuple[DocumentChunk, float]]) -> List[Tuple[DocumentChunk, float]]:
        """Rerank candidates using cross-encoder"""
        if not candidates or not self.cross_encoder:
            return candidates
        
        try:
            # Prepare pairs for cross-encoder
            pairs = [(query, chunk.text) for chunk, _ in candidates]
            
            # Get scores from cross-encoder
            ce_scores = self.cross_encoder.predict(pairs)
            
            # Combine with original scores
            reranked = []
            for i, (chunk, original_score) in enumerate(candidates):
                combined_score = 0.6 * ce_scores[i] + 0.4 * original_score
                reranked.append((chunk, combined_score))
            
            # Sort by combined score
            reranked.sort(key=lambda x: x[1], reverse=True)
            return reranked
            
        except Exception as e:
            logging.error(f"Error in cross-encoder reranking: {e}")
            return candidates
    
    def generate_response_with_images(self, query: str, retrieved_chunks: List[Tuple[DocumentChunk, float]]) -> Tuple[str, List[Dict], List[str]]:
        """Generate response with detailed references AND image files"""
        if not retrieved_chunks:
            return "I couldn't find relevant information to answer your question.", [], []
        
        # Prepare context
        context_parts = []
        references = []
        image_paths = []
        
        for i, (chunk, score) in enumerate(retrieved_chunks[:5]):  # Top 5 chunks
            context_parts.append(f"[Context {i+1}]: {chunk.text}")
            
            ref = {
                "chunk_id": chunk.chunk_id,
                "page": chunk.page_number,
                "section": chunk.section,
                "subsection": chunk.subsection,
                "source_type": chunk.source_type,
                "relevance_score": float(score),
                "text_preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
            }
            
            if chunk.source_type == "image":
                ref["image_description"] = chunk.image_description
                ref["bbox"] = chunk.bbox
                ref["image_path"] = chunk.image_path
                # Add image path to the list for display
                if chunk.image_path and os.path.exists(chunk.image_path):
                    image_paths.append(chunk.image_path)
            
            references.append(ref)
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are a helpful assistant that answers questions based on the provided document context. 

Context from documents:
{context}

Question: {query}

Instructions:
1. Answer the question based ONLY on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Be specific and detailed in your response
4. Reference specific sections or pages when relevant
5. If information comes from an image, mention that explicitly
6. Do not hallucinate or add information not present in the context

Answer:"""
        
        try:
            response = self.vertex_ai.generate_content(prompt)
            return response, references, image_paths
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return "I encountered an error while generating the response.", references, image_paths

class ChatbotUI:
    def __init__(self):
        try:
            os.environ["REQUESTS_CA_BUNDLE"] = "<Path to PROD CA pem file>"
            self.vertex_ai = VertexGenAI()
            self.rag_system = AdvancedRAGSystem(self.vertex_ai)
            self.chat_history = []
        except Exception as e:
            logging.error(f"Error initializing ChatbotUI: {e}")
            raise
        
    def upload_pdfs(self, files):
        if not files:
            return "No files uploaded.", "", []
        
        try:
            file_paths = [file.name for file in files]
            result = self.rag_system.process_pdfs(file_paths)
            return result, "", []
        except Exception as e:
            error_msg = f"Error processing files: {str(e)}"
            logging.error(error_msg)
            return error_msg, "", []
    
    def chat(self, message, history):
        if not message.strip():
            return history, "", []
        
        # Add user message to history
        history = history or []
        history.append([message, None])
        
        try:
            # Retrieve relevant chunks
            retrieved_chunks = self.rag_system.hybrid_retrieval(message)
            
            if not retrieved_chunks:
                response = "I don't have any relevant information to answer your question. Please make sure you've uploaded PDFs first."
                references = []
                image_paths = []
            else:
                # Generate response with images
                response, references, image_paths = self.rag_system.generate_response_with_images(message, retrieved_chunks)
            
            # Add bot response to history
            history[-1][1] = response
            
            # Format references
            ref_text = self._format_references(references)
            
            return history, ref_text, image_paths
            
        except Exception as e:
            error_msg = f"Error processing your question: {str(e)}"
            logging.error(error_msg)
            history[-1][1] = error_msg
            return history, "", []
    
    def _format_references(self, references: List[Dict]) -> str:
        if not references:
            return "No references found."
        
        ref_text = "## References\n\n"
        for i, ref in enumerate(references, 1):
            ref_text += f"**Reference {i}:**\n"
            ref_text += f"- Page: {ref['page']}\n"
            ref_text += f"- Section: {ref['section']}\n"
            ref_text += f"- Subsection: {ref['subsection']}\n"
            ref_text += f"- Source Type: {ref['source_type']}\n"
            ref_text += f"- Relevance Score: {ref['relevance_score']:.3f}\n"
            
            if ref['source_type'] == 'image':
                ref_text += f"- Image Description: {ref.get('image_description', 'N/A')}\n"
                ref_text += f"- **Image displayed in gallery below**\n"
            
            ref_text += f"- Preview: {ref['text_preview']}\n\n"
        
        return ref_text
    
    def clear_chat(self):
        """Clear chat history and references"""
        return [], "References will appear here after asking questions.", []
    
    def create_interface(self):
        with gr.Blocks(title="Advanced RAG PDF Chatbot", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 📚 Advanced RAG PDF Chatbot with Image References")
            gr.Markdown("Upload PDFs and ask questions about their content. The system handles text and images with precise referencing, showing actual images when they contain relevant information.")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # File upload section
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
                    
                    # Chat interface
                    with gr.Group():
                        gr.Markdown("### 💬 Chat")
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=450,
                            avatar_images=("👤", "🤖"),
                            bubble_full_width=False,
                            show_copy_button=True
                        )
                        
                        msg = gr.Textbox(
                            label="Ask a question about your documents",
                            placeholder="Type your question here...",
                            lines=2,
                            max_lines=5
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
                    
                    # Reference images gallery
                    with gr.Group():
                        gr.Markdown("### 🖼️ Reference Images")
                        reference_images = gr.Gallery(
                            label="Images from relevant sources",
                            columns=2,
                            rows=3,
                            height=400,
                            show_label=False,
                            show_download_button=True,
                            interactive=False
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
            
            # Add some example questions
            gr.Markdown("""
            ### 💡 Example Questions:
            - "What does this document say about [topic]?"
            - "Can you explain the chart/diagram on page X?"
            - "What are the key findings mentioned?"
            - "Summarize the methodology section"
            """)
        
        return demo

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create and launch the chatbot
        chatbot_ui = ChatbotUI()
        demo = chatbot_ui.create_interface()
        
        # Launch with sharing enabled
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            debug=False,
            show_error=True
        )
    
    except Exception as e:
        logging.error(f"Error launching application: {e}")
        print(f"Failed to launch application: {e}")

if __name__ == "__main__":
    main()
