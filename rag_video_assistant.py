from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace
)

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)

from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# LLM
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.2,
    max_new_tokens=400,
    streaming=True
)

model = ChatHuggingFace(llm=llm)


def extract_video_id(url):

    parsed = urlparse(url)

    if "youtube.com" in parsed.netloc:
        return parse_qs(parsed.query)["v"][0]

    if "youtu.be" in parsed.netloc:
        return parsed.path[1:]

    raise ValueError("Invalid YouTube URL")


def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


def build_chain(video_url, progress_callback=None):

    def update(message, progress):
        if progress_callback:
            progress_callback(message, progress)

    update("🎙️ Extracting transcript...", 25)

    video_id = extract_video_id(video_url)

    transcript = YouTubeTranscriptApi().fetch(video_id)

    update("✂️ Processing content...", 50)

    transcript_text = " ".join(
        chunk.text
        for chunk in transcript
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    docs = splitter.create_documents(
        [transcript_text]
    )

    update("🧠 Understanding video...", 75)

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    prompt = PromptTemplate(
        template="""
You are an AI assistant answering questions about a YouTube video.

Use ONLY the provided context.

If the answer is not present in the context, say:
"I could not find this information in the video transcript."

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=[
            "context",
            "question"
        ]
    )

    chain = (
        RunnableParallel(
            {
                "context": retriever
                | RunnableLambda(format_docs),

                "question": RunnablePassthrough()
            }
        )
        | prompt
        | model
        | StrOutputParser()
    )

    update("🔍 Preparing search...", 100)

    return chain



# from urllib.parse import urlparse, parse_qs

# from youtube_transcript_api import YouTubeTranscriptApi

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS

# from langchain_huggingface import (
#     HuggingFaceEmbeddings,
#     HuggingFaceEndpoint,
#     ChatHuggingFace
# )

# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import (
#     RunnableParallel,
#     RunnablePassthrough,
#     RunnableLambda
# )

# from langchain_core.output_parsers import StrOutputParser
# from dotenv import load_dotenv

# load_dotenv()

# # Embedding Model
# embeddings = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-small-en-v1.5"
# )

# # LLM
# llm = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen2.5-7B-Instruct",
#     temperature=0.2,
#     max_new_tokens=400,
#     streaming=True
# )

# model = ChatHuggingFace(llm=llm)


# def extract_video_id(url):

#     parsed = urlparse(url)

#     if "youtube.com" in parsed.netloc:
#         return parse_qs(parsed.query)["v"][0]

#     if "youtu.be" in parsed.netloc:
#         return parsed.path[1:]

#     raise ValueError("Invalid YouTube URL")


# def format_docs(docs):

#     return "\n\n".join(
#         doc.page_content
#         for doc in docs
#     )


# def build_chain(video_url):

#     video_id = extract_video_id(video_url)

#     transcript = YouTubeTranscriptApi().fetch(video_id)

#     transcript_text = " ".join(
#         chunk.text
#         for chunk in transcript
#     )

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=800,
#         chunk_overlap=150
#     )

#     docs = splitter.create_documents(
#         [transcript_text]
#     )

#     vectorstore = FAISS.from_documents(
#         docs,
#         embeddings
#     )

#     retriever = vectorstore.as_retriever(
#         search_kwargs={"k": 4}
#     )

#     prompt = PromptTemplate(
#         template="""
# Answer the question using only the provided context.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """,
#         input_variables=[
#             "context",
#             "question"
#         ]
#     )

#     chain = (
#         RunnableParallel(
#             {
#                 "context": retriever
#                 | RunnableLambda(format_docs),

#                 "question": RunnablePassthrough()
#             }
#         )
#         | prompt
#         | model
#         | StrOutputParser()
#     )

#     return chain