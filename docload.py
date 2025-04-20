from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import tempfile

class PDFLoader:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = InMemoryVectorStore(self.embeddings)

    def art_load_pdf_and_extract_key_content(self, uploaded_files: list) -> list[str]:
        extracted_context = []
        for uploaded_file in uploaded_files:
            try:
                # Salva o arquivo temporariamente
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
            except Exception as e:
                raise ValueError(f"Erro ao carregar o arquivo {uploaded_file.name}: {e}")

            try:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200, add_start_index=True
                )
                all_splits = text_splitter.split_documents(docs)
            except Exception as e:
                raise ValueError(f"Erro ao dividir o texto do PDF {uploaded_file.name}: {e}")

            try:
                self.vector_store.add_documents(documents=all_splits)
                results = self.vector_store.similarity_search(
                    "Qual é o objetivo principal do artigo?\n\n\nResumo geral do conteúdo apresentado\n\n\nO que o autor quer transmitir?"
                )
            except Exception as e:
                raise RuntimeError(f"Erro na busca por similaridade no arquivo {uploaded_file.name}: {e}")
            content = "\n".join([r.page_content for r in results])
            extracted_context.append(content)
        return extracted_context

class ExcelLoader:
    pass

