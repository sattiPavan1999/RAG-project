import os
from dotenv import load_dotenv

# 1. Load text file
from langchain_community.document_loaders import TextLoader, DirectoryLoader

# 2. Split text into chunks
from langchain_text_splitters import CharacterTextSplitter

# 3. Embeddings
from langchain_openai import OpenAIEmbeddings

# 4. Store in Chroma DB
from langchain_chroma import Chroma

load_dotenv()

def load_documents(docs_path):
    """Load all the text files from the path"""

    # if the path is not correct or not found
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist")

    # Load all the .txt files from the path
    loader = DirectoryLoader(path=docs_path, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    if(len(documents) == 0):
        raise FileNotFoundError("No files or File Not Found")
    
    return documents
    

def split_documents(documents, chunk_size=500, chunk_overlap=0):
    """Split loaded documents into smaller chunks."""
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create or update a Chroma vector store with chunk embeddings."""

    # 1. Create embeddings
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")   # Needs OPENAI_API_KEY

    # 2. Create vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"✅ Vector DB created with {vector_store._collection.count()} embeddings")
    return vector_store



def main():
    # load all the files
    documents = load_documents(docs_path="docs")

    # chunk the files
    chunks = split_documents(documents)

    # embedding and storing the files
    vector_store = create_vector_store(chunks)

if __name__ == "__main__":
    main()