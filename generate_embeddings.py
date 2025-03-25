from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loaders=[PyPDFLoader('crypt.pdf')]

docs=[]

for file in loaders:
    docs.extend(file.load())

# docs list will contain all of data from all the pdfs

# split all of contents into manageable chunks and for each chunks generate embeddings

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs=text_splitter.split_documents(docs)
embeddings_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device':'cpu'})

vectorstore=Chroma.from_documents(docs, embeddings_function, persist_directory="./chroma_db_nccn")

print(vectorstore._collection.count())
