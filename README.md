# rag-video-intelligence-assistant

A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about any YouTube video using its transcript.

The application extracts the video's transcript, converts it into vector embeddings, stores them in a FAISS vector database, retrieves the most relevant context, and generates accurate answers using a Hugging Face Large Language Model.

## 🚀 Features

- Extract transcripts directly from YouTube videos
- Generate semantic embeddings using BGE embeddings
- Store and retrieve content with FAISS Vector Store
- Context-aware question answering using RAG
- Powered by Hugging Face LLMs
- Simple Streamlit user interface
- Fast and efficient retrieval pipeline

## 🛠️ Tech Stack

### Frontend
- Streamlit

### Backend
- Python
- LangChain

### Embeddings
- BAAI/bge-small-en-v1.5

### Vector Database
- FAISS

### LLM
- Qwen2.5-7B-Instruct (via Hugging Face)

### Other Libraries
- YouTube Transcript API
- LangChain Community
- LangChain Hugging Face
- Python Dotenv

---

## 📂 Project Workflow

1. User enters a YouTube video URL.
2. The application extracts the video ID.
3. Transcript is fetched using YouTube Transcript API.
4. Transcript is split into smaller chunks.
5. Embeddings are generated for each chunk.
6. Chunks are stored in FAISS.
7. Relevant chunks are retrieved based on the user's query.
8. Retrieved context is passed to the LLM.
9. The model generates an answer based only on the retrieved context.

---

## 📦 Installation

### Clone the repository

```bash
git clone https://github.com/muneeb1505/rag-video-intelligence-assistant
cd rag-video-intelligence-assistant
```

### Create a virtual environment

```bash
python -m venv venv
```

### Activate the environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory:

```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_key
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📸 Usage

1. Open the Streamlit application.
2. Paste a YouTube video URL.
3. Submit the URL.
4. Ask questions related to the video.
5. Receive context-aware answers generated from the transcript.

### Example Questions

- What is the main topic of the video?
- Summarize the video.
- What key points were discussed?
- Explain a specific concept mentioned in the video.

---

## 🏗️ Project Structure

```text
project/
│
├── app.py                 # Streamlit frontend
├── backend.py             # RAG pipeline
├── .env                   # Environment variables
├── requirements.txt
└── README.md
```

---

## 🔍 Retrieval-Augmented Generation (RAG) Pipeline

```text
YouTube Video
      │
      ▼
Transcript Extraction
      │
      ▼
Text Chunking
      │
      ▼
Embeddings Generation
      │
      ▼
FAISS Vector Store
      │
      ▼
Retriever
      │
      ▼
Prompt Template
      │
      ▼
Qwen 2.5 LLM
      │
      ▼
Answer Generation
```

---

## 🎯 Future Improvements

- Support multilingual transcripts
- Chat history memory
- Source citations in responses
- Multiple video analysis
- Video summarization feature
- Export conversation history

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.

Feel free to fork the repository and submit a pull request.

---

---

## 👨‍💻 Author

Mirza Muneeb Baig
