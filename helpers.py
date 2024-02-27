import os

from dotenv import load_dotenv

# from langchain.document_loaders import YoutubeLoader
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI


from langchain_openai.embeddings import OpenAIEmbeddings

# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

load_dotenv()


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings()

url = "https://www.youtube.com/watch?v=_qJphZ1Zajw"


def create_vector_db(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)

    return db


def get_response_from_query(db, query, k=4):
    """
    gpt-3.5-turbo-instruct handles 4096 tokens. setting chink size to 1000 and
    4 maximizes the tokens to check
    """

    documents = db.similarity_search(query, k=k)
    docs_page_content = " ".join([doc.page_content for doc in documents])

    llm = OpenAI(model="gpt-3.5-turbo-instruct")

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
            You are a helpful assistant that that can answer questions about youtube videos 
            based on the video's transcript.

            Answer the following question: {question}
            By searching the following video transcript: {docs}

            Only use the factual information from the transcript to answer the question.

            If you feel like you don't have enough information to answer the question, say "I don't know".

            Your answers should be verbose and detailed.
            """,
    )

