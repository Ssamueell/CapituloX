# Import necessary classes and modules for PDF loading, text splitting, embeddings, and vector storage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import tempfile  # For creating temporary files

# PDFLoader class is responsible for processing PDF files and extracting relevant content
class PDFLoader:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = InMemoryVectorStore(self.embeddings)

    def art_load_pdf_and_extract_key_content(self, uploaded_files: list) -> list[str]:
        extracted_context = []  
        
        for uploaded_file in uploaded_files:
            try:
                # Save the uploaded file as a temporary PDF on disk
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                
                # Load the PDF content using LangChain's PyPDFLoader
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
            except Exception as e:
                raise ValueError(f"Error loading file {uploaded_file.name}: {e}")

            try:
                # Split the loaded document into smaller chunks for better processing
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,       # Maximum number of characters per chunk
                    chunk_overlap=200,     # Number of overlapping characters between chunks
                    add_start_index=True   # Include index of chunk in metadata
                )
                all_splits = text_splitter.split_documents(docs)
            except Exception as e:
                raise ValueError(f"Error splitting text from PDF {uploaded_file.name}: {e}")

            try:
                # Add the document chunks to the vector store
                self.vector_store.add_documents(documents=all_splits)

                # Perform a semantic similarity search using a fixed query
                results = self.vector_store.similarity_search(
                    "What is the main objective of the article?\n\n\nGeneral summary of the presented content\n\n\nWhat is the author's main message?"
                )
            except Exception as e:
                raise RuntimeError(f"Similarity search error in file {uploaded_file.name}: {e}")

            # Join the search results into a single string and append to the result list
            content = "\n".join([r.page_content for r in results])
            extracted_context.append(content)

        return extracted_context  # Return a list of summaries, one for each uploaded file

# Placeholder for a future ExcelLoader implementation
class ExcelLoader:
    pass
