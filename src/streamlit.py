import os
os.environ['STREAMLIT_DISABLE_TORCH_PATHS'] = '1'

import streamlit as st
from r1_smolagent_rag import primary_agent

def init_chat_history():
    """
    Initialize chat history in the session state if not present.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []

def display_chat_history():
    """
    Display each message in the chat history on the Streamlit page.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input(prompt: str):
    """
    Handle user input by sending it to the agent and appending responses to 
    the session state chat history.
    """
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = primary_agent.run(prompt, reset=False)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

def display_sidebar():
    """
    Display the sidebar with an 'About' section and a button to clear chat history.
    """
    with st.sidebar:
        st.title("About")
        st.markdown("""
        This Q&A bot uses Retrieval-Augmented Generation (RAG) to answer questions about your documents.
        
        **How it works:**
        1. Your query is matched against chunked embeddings of the documents.
        2. The most relevant chunks are retrieved.
        3. A reasoning model generates an answer based on the context.
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

def main():
    """
    Main Streamlit application entry point.
    """
    st.set_page_config(page_title="Document Q&A Bot", layout="wide")
    st.title("Document Q&A Bot")

    init_chat_history()
    display_chat_history()
    display_sidebar()

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        handle_user_input(prompt)

if __name__ == "__main__":
    main()
