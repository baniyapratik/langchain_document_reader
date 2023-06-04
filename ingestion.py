import requests
import os
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import pinecone

from consts import INDEX_NAME

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


def ingest_ts_docs():
    html_files = []
    # all the html files have been stored in the tmp directory
    # command: wget -r --no-parent
    # https://docs.thoughtspot.com/software/latest/ -P thoughtspot-docs --html-extension
    directory = "/tmp/thoughtspot-docs/docs.thoughtspot.com/software/latest/"
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".html"):
                html_files.append(os.path.join(root, filename))

    documents = []
    for html_file in html_files:
        loader = UnstructuredHTMLLoader(file_path=html_file)
        raw_documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
        )
        document = text_splitter.split_documents(documents=raw_documents)
        # print(f"Split into {len(document)} chunks")
        documents.extend(document)

    for document in documents:
        old_path = document.metadata["source"]
        new_path = old_path.replace("/tmp/thoughtspot-docs", "https:/")
        document.metadata.update({"source": new_path})

    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(
        documents=documents, embedding=embeddings, index_name=INDEX_NAME
    )


def ingest_pdf(pdf_url):
    # check if the url has .pdf
    r = requests.get(pdf_url, stream=True)
    content_type = r.headers.get("content-type")
    if "application/pdf" not in content_type:
        raise TypeError("url is not a pdf")
    # create a tmp file
    pdf_path = "/tmp/" + pdf_url.rsplit("/", 1)[-1]
    with open(pdf_path, "wb") as file:
        file.write(r.content)
        file.write(r.raw.read())
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local("faiss_index_test")
    # delete file
    if os.path.exists(pdf_path):
        os.remove(pdf_path)


def get_data(query):
    if not query:
        query = "Summarize the document for me in one paragraph"
    embeddings = OpenAIEmbeddings()
    new_vector_store = FAISS.load_local("faiss_index_test", embeddings)
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), retriever=new_vector_store.as_retriever()
    )
    res = qa.run(query)
    print(res)


def ingest_text(text_file_path):
    """ingests text files"""
    loader = TextLoader(text_file_path)
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    # print(len(texts))

    embeddings = OpenAIEmbeddings(openai_api_keys=os.environ.get("OPENAI_API_KEYS"))
    Pinecone.from_documents(texts, embeddings, index_name=INDEX_NAME)


if __name__ == "__main__":
    # pdf reader
    # ingest_pdf(pdf_url="https://arxiv.org/pdf/2305.07185.pdf")
    # print(get_data())

    # ingest ts documentation reader
    ingest_ts_docs()
