# Required imports
import os
from langchain_openai import ChatOpenAI  
from loaders.docload import PDFLoader    
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv           
import streamlit as st                
load_dotenv()
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "CapituloX"

# Main application function
def main():
    st.set_page_config(page_title="Chat with PDF files", layout="centered")
    st.title("🤖 Ask ChapterX – ")

    # File uploader to upload multiple PDF files
    uploaded_files = st.file_uploader("**Upload your pdf Here**", type="pdf", accept_multiple_files=True)
    if not uploaded_files:
        st.info("Please upload at least one PDF file.")
        st.stop()
        
    # Function to process and extract content from uploaded PDFs (cached to avoid reprocessing)
    @st.cache_data
    def load_context_from_pdfs(uploaded_files):
        pdf_loader = PDFLoader()  # Instantiate custom loader
        extracted_context = pdf_loader.art_load_pdf_and_extract_key_content(uploaded_files)
        return "\n\n\n".join(extracted_context)  # Combine all extracted contents into one string
    
    # Show loading spinner while extracting content
    with st.spinner("Extracting content from files..."):
        combined_contexts = load_context_from_pdfs(uploaded_files)
    
    # Stop if no content was extracted
    if not combined_contexts:
        st.stop()
    
    # Instantiate the LLM (GPT-4o-mini in this case)
    llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")

    # Define system message template (guides the AI’s behavior)
    system_template = (
        "You are an AI that answers questions based on the following context:\n{context}\n"
        "Always relate your response to the user's question."
    )
    
    # Prompt templates for system and user inputs
    sys_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_prompt = HumanMessagePromptTemplate.from_template("{input}")
    
    # Combine prompts and link them to the model
    chat_template = ChatPromptTemplate.from_messages([sys_prompt, human_prompt])
    chain = chat_template | llm  # Pipeline: prompt -> LLM

    # Input field for user to type questions based on the uploaded PDFs
    human_question = st.chat_input("Type your question based on the uploaded PDFs:")
    
    # If user submits a question, generate and display the answer
    if human_question:
        with st.spinner("Generating answer..."):
            try:
                response = chain.invoke({
                    "context": combined_contexts,
                    "input": human_question
                })
                # Display the AI-generated response
                st.markdown("### 💬 Answer")
                st.write(response.content)
            except Exception as e:
                # Display error message in case of failure
                st.error(f"An error occurred while generating the answer: {e}")
                
    with st.expander("📚 View extracted PDF content"):
        st.write(combined_contexts)

# Run the main function
if __name__ == "__main__":
    main()
