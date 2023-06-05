import os
from dotenv import load_dotenv
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from consts import INDEX_NAME

load_dotenv()
pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENVIRONMENT_REGION"),
)


def ingest_docs() -> None:
    loader = ReadTheDocsLoader(
        path="langchain-docs/langchain.readthedocs.io/en/latest",
        encoding="utf8",
        features="html.parser",
    )
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents) } documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} chunks")

    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to insert {len(documents)} to Pinecone")
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    Pinecone.from_documents(documents[3969:], embeddings, index_name=INDEX_NAME)
    print("****** Added to Pinecone vectorstore vectors")


if __name__ == "__main__":
    ingest_docs()
