# RAG Project with FAISS and Flan-T5

This project implements a Retrieval Augmented Generation (RAG) system using FAISS for efficient similarity search and a Flan-T5 model for text generation. The system is designed to answer questions based on a provided knowledge base, which in this case is a PDF document about statistics.

## Components:

1.  **PDF Document Loading and Text Extraction**: Uses `pypdf` to load a PDF file and extract its textual content.
2.  **Text Chunking**: Employs `langchain-text-splitters` (specifically `RecursiveCharacterTextSplitter`) to break down the extracted text into smaller, manageable chunks. This is crucial for effective retrieval and to avoid exceeding the model's token limit.
3.  **Sentence Embeddings**: Utilizes the `sentence-transformers` library with the 'all-MiniLM-L6-v2' model to convert text chunks into dense vector representations (embeddings). These embeddings capture the semantic meaning of the text.
4.  **FAISS Indexing**: Creates an in-memory FAISS (Facebook AI Similarity Search) index (`IndexFlatL2`) to store the chunk embeddings. FAISS enables fast and efficient similarity search, allowing the system to quickly find relevant text chunks given a query.
5.  **Flan-T5 Language Model**: Integrates 'google/flan-t5-small' from `transformers` as the generative model. This model is responsible for generating human-like answers based on the retrieved context and the user's query.
6.  **RAG Pipeline**: A custom `answer_question` function orchestrates the RAG process:
    *   **Retrieval**: Embeds the user's query and searches the FAISS index to retrieve the top `k` most similar text chunks from the knowledge base.
    *   **Augmentation**: Constructs a prompt that combines the retrieved context with the user's question, instructing the generative model to answer *only* based on the provided context.
    *   **Generation**: Feeds the augmented prompt to the Flan-T5 model to generate a coherent answer.

## How it Works:

When a user asks a question, the system first converts the question into an embedding. This query embedding is then used to search the FAISS index to find the most semantically similar chunks from the loaded PDF document. These relevant chunks are then provided as context to the Flan-T5 model, along with the original question. The Flan-T5 model then generates an answer, ensuring that the response is grounded in the information extracted from the PDF.

This approach helps in reducing hallucinations common in large language models by explicitly providing relevant information from a specific knowledge base.
"""

future_improvements_md = """
## Future Improvements

1.  **Dynamic PDF Loading**: Implement functionality to dynamically upload and process different PDF documents, allowing users to switch knowledge bases easily.
2.  **Persistent FAISS Index**: Instead of rebuilding the FAISS index every time, save and load the index to disk for faster startup times and more efficient resource usage.
3.  **Advanced Chunking Strategies**: Experiment with different text chunking strategies and overlap values to optimize retrieval performance and context quality.
4.  **Hybrid Search**: Explore combining vector similarity search (FAISS) with keyword-based search for improved retrieval accuracy.
5.  **Evaluation Metrics**: Implement quantitative evaluation metrics (e.g., ROUGE, BLEU, custom relevance scores) to systematically assess the quality of generated answers.
6.  **User Interface**: Develop a simple web-based or command-line interface for easier interaction with the RAG system.
7.  **Larger Language Model**: Upgrade to a more powerful Flan-T5 model (e.g., `flan-t5-base`, `flan-t5-large`) or other suitable LLMs for potentially better generation quality.
8.  **Error Handling and Edge Cases**: Improve error handling for malformed PDFs, empty contexts, and queries that cannot be answered from the provided knowledge base.
9.  **Deployment**: Containerize the application using Docker and explore deployment options (e.g., Hugging Face Spaces, Google Cloud Run) for broader accessibility.
"""
