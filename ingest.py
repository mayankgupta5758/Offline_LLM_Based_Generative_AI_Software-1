import time
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

def create_vector_db():
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    documents = loader.load()
    print(f"Loaded {len(documents)} PDF documents.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"Split documents into {len(texts)} text chunks.")

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    # Start measuring time
    start_time = time.time()

    db = FAISS.from_documents(texts, embeddings)
    print("Created the FAISS vector database.")

    db.save_local(DB_FAISS_PATH)
    print(f"Vector database saved to: {DB_FAISS_PATH}")

    # Calculate and print the total time taken in minutes
    end_time = time.time()
    total_time_seconds = end_time - start_time
    total_time_minutes = total_time_seconds / 60
    print(f"Total time taken: {total_time_minutes:.2f} minutes")

if __name__ == "__main__":
    create_vector_db()
