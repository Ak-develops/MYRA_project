import os
import shutil
import warnings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

warnings.filterwarnings("ignore")

# ---------- PATH SETUP ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data")
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")

REBUILD = True  # Set False to skip rebuild if index exists


# ---------- LOAD PDFs ----------
def load_pdfs(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data folder not found: {data_path}")

    docs = []

    for file in os.listdir(data_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(data_path, file)
            loader = PyPDFLoader(file_path)
            pages = loader.load()

            # Attach metadata (source + page)
            for i, p in enumerate(pages):
                p.metadata["source"] = file
                p.metadata["page"] = p.metadata.get("page", i)

            docs.extend(pages)

    if len(docs) == 0:
        raise ValueError("No PDFs found in data folder")

    return docs


# ---------- TEXT CLEANING ----------
def clean_text(text):
    return " ".join(text.split())


# ---------- CHUNKING ----------
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    # Clean text
    for c in chunks:
        c.page_content = clean_text(c.page_content)

    return chunks


# ---------- FILTER JUNK ----------
def filter_chunks(chunks):
    clean_chunks = []
    for c in chunks:
        text = c.page_content.strip()
        if len(text) > 50:  # remove tiny useless chunks
            clean_chunks.append(c)
    return clean_chunks


# ---------- EMBEDDINGS ----------
def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ---------- BUILD INDEX ----------
def build_faiss_index(chunks, embedding_model):
    return FAISS.from_documents(chunks, embedding_model)


# ---------- MAIN PIPELINE ----------
def main():

    # Handle rebuild logic
    if os.path.exists(INDEX_PATH):
        if REBUILD:
            print("Rebuilding index...")
            shutil.rmtree(INDEX_PATH)
        else:
            print("FAISS index already exists. Skipping.")
            return

    print(" Loading PDFs...")
    docs = load_pdfs(DATA_PATH)
    print(f"Loaded {len(docs)} pages")

    print(" Splitting into chunks...")
    chunks = split_documents(docs)
    print(f"Chunks before filtering: {len(chunks)}")

    chunks = filter_chunks(chunks)
    print(f"Chunks after filtering: {len(chunks)}")

    # Debug sample
    print("\n Sample chunk preview:")
    print(chunks[0].page_content[:300])
    print("Metadata:", chunks[0].metadata)

    print("\n Creating embeddings...")
    embedding_model = create_embeddings()

    print(" Building FAISS index...")
    vectorstore = build_faiss_index(chunks, embedding_model)

    print(" Saving index...")
    vectorstore.save_local(INDEX_PATH)

    print("\nIngestion complete!")
    print(f"Total chunks stored: {len(chunks)}")


if __name__ == "__main__":
    main()