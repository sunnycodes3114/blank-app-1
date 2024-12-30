import logging
import streamlit as st
from io import BytesIO
import fitz  # PyMuPDF for PDF processing
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema import Document  # Import the Document class for storing text content


# Load PDF and split into chunks using PyMuPDF
def load_pdf_and_split(uploaded_file):
    # Convert the uploaded file to a byte stream
    pdf_data = BytesIO(uploaded_file.read())
    
    # Open the PDF with PyMuPDF (fitz)
    doc = fitz.open(stream=pdf_data, filetype="pdf")  # Specify the stream and filetype explicitly
    
    # Extract text from each page (assuming single-page here)
    text_chunks = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text_chunks += page.get_text("text")  # Extract text from the page
    
    return text_chunks


# Split text into smaller chunks
def split_text_into_chunks(text_chunks, chunk_size):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size)
    chunks = text_splitter.create_documents([text_chunks])
    return chunks


# Embed documents using GoogleGenerativeAI embeddings
def embed_documents(text_chunks, api_key, model):
    embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model=model)
    vectors = embedding_model.embed_documents(text_chunks)
    return vectors


# Create the Chroma vector store
def create_vector_store(chunks, embedding_model):
    db = Chroma.from_documents(chunks, embedding_model)
    return db


# Load conversation history from the vector store
def load_conversation_history(db_connection):
    # Create a retriever from the vector store
    retriever = db_connection.as_retriever(search_kwargs={"k": 5})
    
    # Use the retriever to retrieve relevant documents based on the query
    history_docs = retriever.get_relevant_documents("conversation")  # Pass the query for conversation history
    
    # Extract the content from the retrieved documents
    conversation_history = [doc.page_content for doc in history_docs]
    return conversation_history


# Create the RAG chain with the retriever and prompt template
def build_rag_chain(retriever, chat_template, api_key, model):
    output_parser = StrOutputParser()
    
    # Create a direct call to retrieve documents and generate response
    def rag_chain(query):
        # Retrieve relevant documents based on the query
        docs = retriever.get_relevant_documents(query)
        
        # Extract content from the retrieved documents
        context = "\n\n".join(doc.page_content for doc in docs)
        
        # Prepare the prompt with the context and the user's question
        prompt = chat_template.format(context=context, question=query)
        
        # Get the response from the generative AI model
        response = ChatGoogleGenerativeAI(google_api_key=api_key, model=model).invoke(prompt)
        
        # Parse the response and return the answer
        return output_parser.invoke(response)
    
    return rag_chain


# Streamlit app
def main():
    st.title("Document-based Q&A Assistant")
    st.write("Upload a PDF and ask questions about its content.")
    
    # Upload PDF
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    # Read the API key
    api_key = st.secrets["google_api_key"]
    
    # Only run if a file is uploaded
    if uploaded_file is not None:
        # Load PDF and split into chunks
        text_chunks = load_pdf_and_split(uploaded_file)
        
        # Split the text into chunks (adjust chunk_size as needed)
        chunks = split_text_into_chunks(text_chunks, chunk_size=300)
        
        # Embed the documents
        vectors = embed_documents(text_chunks, api_key, model="models/embedding-001")
        
        # Create the vector store
        db = create_vector_store(chunks, GoogleGenerativeAIEmbeddings(google_api_key=api_key, model="models/embedding-001"))
        
        # Load conversation history (optional, if you're doing continuous chat)
        db_connection = Chroma(embedding_function=GoogleGenerativeAIEmbeddings(google_api_key=api_key, model="models/embedding-001"))
        conversation_history = load_conversation_history(db_connection)
        
        # Set up the chat template for context-based conversation
        chat_template = ChatPromptTemplate.from_messages([SystemMessage(content="I'm a helpful AI assistant. I'll use the provided document to answer your questions."),
                                                         HumanMessagePromptTemplate.from_template("""Answer the following question based on the provided context:

        Context:
        {context}

        Question:
        {question}

        Answer:""")])
        
        # Define the model to use
        model = "gemini-1.5-pro-latest"
        
        # Build the RAG chain
        rag_chain = build_rag_chain(db_connection.as_retriever(), chat_template, api_key, model)
        
        # Get user input
        user_question = st.text_input("Ask a question:")
        
        if user_question:
            # Log the user question
            logging.info(f"User question: {user_question}")
            
            # Run the RAG chain and get the response
            response = rag_chain(user_question)
            
            # Display the response
            st.write("Answer:", response)


if __name__ == "__main__":
    main()
