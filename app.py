import nltk
import gradio as gr

from assistant import Assistant
from citation import Citation
from retrievers import KeyWordRetriever, SemanticRetriever, HybridRetriever
from text_processing import extract_text_from_pdf, clean_text
from embeddings import process_pdf_for_rag
from reranker import Reranker

nltk.download('punkt', quiet=True)
nltk.download('punkt')
nltk.download('punkt_tab')

# Load and preprocess PDF
data_path = "data"
pdf_path = data_path + "/sherlock.pdf"

chunks = process_pdf_for_rag(pdf_path, chunk_size=500)
pdf_text = extract_text_from_pdf(pdf_path)

cleaned_text = clean_text(pdf_text)
citation = Citation(cleaned_text)

# Initialize retrievers
keyword_retriever = KeyWordRetriever(chunks)
semantic_retriever = SemanticRetriever(chunks)
hybrid_retriever = HybridRetriever(keyword_retriever, semantic_retriever)

# Initialize assistant
reranker = Reranker()
assistant = Assistant(hybrid_retriever, reranker, citation=citation)

# Gradio UI
def run_rag_ui(api_key, query, retriever_type, top_k, use_reranker):
    if retriever_type.lower() == "keyword":
        retriever = keyword_retriever
    elif retriever_type.lower() == "semantic":
        retriever = semantic_retriever
    elif retriever_type.lower() == "hybrid":
        retriever = hybrid_retriever
    else:
        return "Invalid retrieval method selected."

    reranker = Reranker() if use_reranker else None
    pdf_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(pdf_text)
    citation = Citation(cleaned_text)
    assistant = Assistant(retriever, reranker, citation=citation)
    response, retrieved_chunks, citations = assistant.handle_query(
        query, api_key, 
        retriever_type=retriever_type, 
        top_k=top_k, 
        use_reranker=use_reranker
    )

    return response, citations, retrieved_chunks

iface = gr.Interface(
    fn=run_rag_ui,
    inputs=[
        gr.Textbox(label="API Key", placeholder="Enter your Groq API Key", type="password"),
        gr.Textbox(label="Query", placeholder="Enter your query", type="text"),
        gr.Radio(choices=["keyword", "semantic", "hybrid"], label="Retrieval Method", value='hybrid'),
        gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Number of chunks in context"),
        gr.Radio(choices=[True, False], label="Use Reranker", value=False)
    ],
    outputs=[
        gr.Textbox(label="LLM Response", interactive=False),
        gr.Textbox(label="Citations", interactive=False),
        gr.Textbox(label="Retrieved Chunks", interactive=False)
    ],
    title="RAG System with Gradio UI",
    description="Enter your query, select the retrieval method, and get retrieved chunks along with LLM responses."
)

iface.launch(share=True)
