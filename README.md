# AreebaBot 🤖

**Student Name:** Areeba Amar
**Student ID:** S2023065066
**Project Title:** RAG-based Personal Chatbot (NLP Semester Project - CC438)
**Chatbot Name:** AreebaBot
**Course Instructor:** Mr. Waqar Ashiq

---

## Brief Project Description

AreebaBot is a **Retrieval-Augmented Generation (RAG)** chatbot that answers questions
about Areeba Amar (education, skills, certifications, academic projects) using her
**CV/Resume** as a personal, custom knowledge base — as required by the project task.

**Pipeline:**
1. **Document Loading** — Areeba's CV (PDF) is loaded with `PyPDFLoader`.
2. **Preprocessing / Chunking** — Text is split into overlapping chunks using
   `RecursiveCharacterTextSplitter` for better retrieval granularity.
3. **Embedding Generation** — Each chunk is embedded using the free, local
   `sentence-transformers/all-MiniLM-L6-v2` model (HuggingFace) — no paid API needed.
4. **Vector Database** — Embeddings are stored and searched using **FAISS**.
5. **Retrieval** — On each user query, the top-3 most relevant chunks are retrieved.
6. **LLM Response Generation** — Retrieved context + conversation history + the
   question are sent to **Groq's free LLM API** (`llama-3.3-70b-versatile`) using a
   custom prompt template so AreebaBot answers in-character and stays grounded in
   the CV content (no hallucination).
7. **History Maintenance** — `ConversationBufferMemory` keeps track of the full
   chat history so AreebaBot can handle follow-up questions naturally.
8. **Interface** — A clean **Streamlit** chat UI (ChatGPT-style).

**Tech stack:** LangChain · FAISS · HuggingFace Embeddings (local/free) · Groq API (free) · Streamlit

---

## Deployment Link

🔗 **[ADD YOUR STREAMLIT CLOUD LINK HERE AFTER DEPLOYING]**

(See `DEPLOYMENT_STEPS.md` for exact step-by-step instructions to get this link.)

---

## Project Structure

```
AreebaBot/
├── app.py                  # Streamlit chat interface
├── rag_engine.py           # RAG pipeline (loading, embeddings, FAISS, chain)
├── requirements.txt        # Python dependencies
├── data/
│   └── CV.pdf              # Personal dataset (Areeba's CV)
├── README.md
└── DEPLOYMENT_STEPS.md      # Step-by-step deployment guide
```

---

## How to Run Locally

### 1. Clone / extract the project folder
```bash
cd AreebaBot
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get a FREE Groq API key
1. Go to https://console.groq.com/keys
2. Sign up (free) and click **"Create API Key"**
3. Copy the key (starts with `gsk_...`)

### 5. Run the app
```bash
streamlit run app.py
```

### 6. Use the chatbot
- The app opens in your browser at `http://localhost:8501`
- Paste your Groq API key in the **sidebar**
- Start chatting with AreebaBot! Example questions:
  - "What programming languages does Areeba know?"
  - "Tell me about her academic projects"
  - "What certifications does she have?"
  - "Where is she from?"

> Note: On first run, the app builds a FAISS index from `data/CV.pdf` and
> caches it in a `faiss_index/` folder. Subsequent runs load instantly from cache.

---

## Notes

- Only **personal/custom data** (Areeba's own CV) is used as the knowledge base, per
  assignment requirements — no generic datasets (e.g. Wikipedia) are used.
- The Groq API key is entered by the user at runtime (sidebar) and is **never hardcoded**
  in the source code, so the project is safe to share/submit publicly.
- Embeddings use a free local HuggingFace model — **zero cost**, no embedding API key needed.
- Groq's LLM API is **free** (generous free tier) — no paid API key required.
