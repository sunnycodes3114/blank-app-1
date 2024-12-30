import logging
import streamlit as st
from io import BytesIO
import fitz  # PyMuPDF for PDF processing
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema import Document  # Import the Document class for storing text content
import faiss  # Correctly imported from faiss
import numpy as np

# Load PDF and split into chunks using PyMuPDF
def load_pdf_and_split(uploaded_file):
    pdf_data = BytesIO(uploaded_file.read())
    doc = fitz.open(stream=pdf_data, filetype="pdf")
    text_chunks = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text_chunks += page.get_text("text")
    return text_chunks

# Split text into smaller chunks
def split_text_into_chunks(text_chunks, chunk_size=300):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size)
    chunks = text_splitter.create_documents([text_chunks])
    return chunks

# Embed documents using GoogleGenerativeAI embeddings
def embed_documents(text_chunks, api_key, model="models/embedding-001"):
    embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model=model)
    vectors = embedding_model.embed_documents(text_chunks)
    return vectors

# Create the FAISS vector store
def create_vector_store(chunks, embedding_model):
    embeddings = [embedding_model.embed_documents([chunk.page_content])[0] for chunk in chunks]
    index = faiss.IndexFlatL2(len(embeddings[0]))
    
    # Store documents and their embeddings together
    document_store = [(chunk, embeddings[i]) for i, chunk in enumerate(chunks)]
    
    index.add(np.array(embeddings, dtype=np.float32))
    
    return index, document_store

# Load conversation history from the vector store (optional, if you're doing continuous chat)
def load_conversation_history(db_connection, query="conversation"):
    conversation_history = []  # Return empty history for simplicity
    return conversation_history

# Create the RAG chain with the retriever and prompt template
def build_rag_chain(retriever, document_store, chat_template, api_key, model):
    output_parser = StrOutputParser()
    
    # Create a direct call to retrieve documents and generate a response
    def rag_chain(query, history):
        # Embed the query to get its vector representation
        embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model="models/embedding-001")
        query_vector = embedding_model.embed_documents([query])[0]
        
        # Search for the nearest vectors in the FAISS index
        k = 5  # Number of nearest neighbors to retrieve
        D, I = retriever.search(np.array([query_vector], dtype=np.float32), k)
        
        # Retrieve the corresponding documents from the document_store
        docs = [document_store[i][0] for i in I[0]]  # Accessing the documents from the store
        
        # Extract content from the retrieved documents
        context = "\n\n".join(doc.page_content for doc in docs)
        
        # Add previous messages to context to maintain conversation continuity
        full_context = context + "\n\n" + "\n".join([msg['content'] for msg in history])
        
        # Prepare the prompt with the full context and the user's question
        prompt = chat_template.format(context=full_context, question=query)
        
        # Get the response from the generative AI model
        response = ChatGoogleGenerativeAI(google_api_key=api_key, model=model).invoke(prompt)
        
        # Parse the response and return the answer
        return output_parser.invoke(response)
    
    return rag_chain

# Streamlit app
def main():
    st.title("Document-based Q&A Assistant")
    st.write("Upload a PDF and ask questions about its content.")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    api_key = st.secrets["google_api_key"]
    
    # Store chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if uploaded_file is not None:
        text_chunks = load_pdf_and_split(uploaded_file)
        chunks = split_text_into_chunks(text_chunks, chunk_size=300)
        
        embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model="models/embedding-001")
        db, document_store = create_vector_store(chunks, embedding_model)
        
        chat_template = ChatPromptTemplate.from_messages([ 
            SystemMessage(content="I'm a helpful AI assistant. I'll use the provided document to answer your questions."),
            HumanMessagePromptTemplate.from_template(""" 
Answer the following question based on the provided context:

Context:
{context}

Question:
{question}

Answer:""")
        ])
        
        model = "gemini-1.5-pro-latest"
        
        rag_chain = build_rag_chain(db, document_store, chat_template, api_key, model)
        
        # Show previous messages (if any)
        for msg in st.session_state.messages:
            if msg['role'] == 'user':
                st.chat_message("user").markdown(msg['content'])
            else:
                st.chat_message("Document AI").markdown(msg['content'])
        
        user_question = st.text_input("Ask a question:")

        if user_question:
            logging.info(f"User question: {user_question}")
            # Append the user's message to the session's chat history
            st.session_state.messages.append({"role": "user", "content": user_question})
            
            # Get the response from the RAG chain
            response = rag_chain(user_question, st.session_state.messages)
            
            # Append the response from the Document AI to the chat history
            st.session_state.messages.append({"role": "Document AI", "content": response})
            
            # Display the bot's response
            st.chat_message("Document AI").markdown(response)

if __name__ == "__main__":
    main()
