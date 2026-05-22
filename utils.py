from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from transformers import pipeline


# -----------------------------
# CREATE TEXT CHUNKS
# -----------------------------

def create_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = text_splitter.split_text(text)

    return chunks


# -----------------------------
# CREATE VECTOR STORE
# -----------------------------

def create_vector_store(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(
        chunks,
        embedding=embeddings
    )

    return vector_store

from functools import lru_cache


# -----------------------------
# LOAD MODEL ONLY ONCE
# -----------------------------

from transformers import pipeline
from functools import lru_cache


# -----------------------------
# LOAD MODEL
# -----------------------------

@lru_cache(maxsize=1)
def load_model():

    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )

    return generator
# -----------------------------
# GENERATE ANSWER
# -----------------------------

def get_answer(query, vector_store):

    generator = load_model()

    docs = vector_store.similarity_search(
        query,
        k=2
    )

    context = "\n".join(
        [doc.page_content[:300] for doc in docs]
    )

    prompt = f"""
    Answer the question using only the provided context.

    Context:
    {context}

    Question:
    {query}

    Give a short and direct answer.
    """

    response = generator(
        prompt,
        max_new_tokens=50
    )

    answer = response[0]["generated_text"]

    return answer