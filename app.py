import streamlit as st

from pypdf import PdfReader

from utils import (
    create_chunks,
    create_vector_store,
    get_answer
)

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 INTELLIGENT RAG-BASED DOC ASSISSTANT")

st.markdown(
    "Upload a PDF and ask questions about its content."
)

# -----------------------------
# SESSION STATE
# -----------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:

    st.header("Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf"
    )

# -----------------------------
# PROCESS PDF
# -----------------------------

if uploaded_file is not None:

    with st.spinner("Processing PDF..."):

        text = ""

        reader = PdfReader(uploaded_file)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text

        chunks = create_chunks(text)

        vector_store = create_vector_store(chunks)

        st.session_state.vector_store = vector_store

    st.sidebar.success(
        "PDF uploaded successfully!"
    )

# -----------------------------
# DISPLAY CHAT HISTORY
# -----------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# -----------------------------
# USER INPUT
# -----------------------------

query = st.chat_input(
    "Ask a question from the PDF..."
)

# -----------------------------
# GENERATE RESPONSE
# -----------------------------

if query:

    # USER MESSAGE

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):

        st.markdown(query)

    # ASSISTANT RESPONSE

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer = get_answer(
                query,
                st.session_state.vector_store
            )

            st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )