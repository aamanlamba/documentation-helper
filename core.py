from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain

import pinecone
import os
from dotenv import load_dotenv
from typing import Any

from consts import INDEX_NAME

load_dotenv()
pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENVIRONMENT_REGION"),
)


def run_llm(query: str) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(
        index_name="langchain-doc-index", embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0)
    ### Retrieval QA Chain example
    qa = RetrievalQA.from_chain_type(
            llm=chat,
            chain_type="stuff",
            retriever=docsearch.as_retriever(),
            return_source_documents=True,
            verbose=True,
    )
    qa2 = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )
    return qa({"query": query})
    #return qa({"query": query, "chat_history": chat_history})


if __name__ == "__main__":
    print(run_llm(query="What is RetrievalQA Chain?"))
