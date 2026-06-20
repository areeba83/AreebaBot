"""
rag_engine.py
--------------
Core RAG (Retrieval-Augmented Generation) engine for AreebaBot.

Pipeline:
1. Load CV (PDF) from data/
2. Split into chunks
3. Generate embeddings (HuggingFace - free, local, no API key needed)
4. Store/retrieve embeddings using FAISS vector database
5. Retrieve relevant chunks for a user query
6. Pass retrieved context + chat history + question to Groq LLM
7. Return generated answer

This file builds the vector store ONCE and caches it on disk (faiss_index/)
so subsequent runs are fast.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

DATA_PATH = "data/CV.pdf"
INDEX_PATH = "faiss_index"

# Custom prompt so the bot answers in AreebaBot's persona and stays grounded
# in the retrieved context instead of hallucinating.
CUSTOM_PROMPT_TEMPLATE = """You are AreebaBot, a friendly and professional AI assistant representing Areeba Amar,
a Software Engineering student at UMT Lahore. You answer questions about Areeba based ONLY on the
context provided below, which comes from her CV/resume.

Rules:
- Answer naturally and conversationally, as if you personally know Areeba's background.
- Use ONLY the information given in the context. If the answer is not in the context, politely say
  you don't have that information about Areeba, and don't make anything up.
- Keep answers concise and relevant to what was asked.
- You may refer to the chat history to understand follow-up questions.

Context about Areeba:
{context}

Chat History:
{chat_history}

Question: {question}

Answer as AreebaBot:"""


def build_or_load_vectorstore():
    """
    Loads the CV PDF, splits it into chunks, embeds the chunks, and stores
    them in a FAISS vector index. If an index already exists on disk, it is
    loaded instead of being rebuilt (saves time on every app restart).
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(INDEX_PATH):
        vectorstore = FAISS.load_local(
            INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
        return vectorstore

    # 1. Load document
    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()

    # 2. Split into overlapping chunks for better retrieval context
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documents)

    # 3. Embed + 4. Store in FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)

    return vectorstore


def get_conversational_chain(vectorstore, groq_api_key):
    """
    Builds a ConversationalRetrievalChain:
    - Retriever: pulls top-k relevant chunks from FAISS for a given query
    - Memory: maintains chat history across turns
    - LLM: Groq's hosted Llama model generates the final answer
    """
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "chat_history", "question"],
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    return chain
