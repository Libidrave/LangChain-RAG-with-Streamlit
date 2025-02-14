from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.parsers import PyMuPDFParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma

from pathlib import Path

def load_data(documents):
    """
    Load and parse data from a list of PDF files.

    Args:
    documents Union[UploadedFile, list(UploadedFile)]: A single UploadedFile or list of UploadedFile objects. Strict for PDF only.

    Returns:
    List[Document]: A list of parsed LangChain Document class.
    """
    # Write PDF file to current working directory
    for file in documents:
        with open(f"./{file.name}", 'wb') as f:
            f.write(file.getbuffer())

    # Load and parse the data        
    loader = GenericLoader(blob_loader=FileSystemBlobLoader(path="./", glob="*.pdf"),
                           blob_parser=PyMuPDFParser(mode='page'))
    loaded_docs = loader.load()

    # Remove temporary PDF files after loading
    pdf_files = Path.cwd().glob("*.pdf")
    for pdf in pdf_files:
        pdf.unlink()

    return loaded_docs

def split_data(loaded_docs):
    """
    Split a list of loaded documents into smaller chunks.

    Args:
    loaded_docs List[Document]: A list of loaded LangChain Document class.

    Returns:
    List[Document]: A list of smaller chunks of parsed document.
    """
    splitter = RecursiveCharacterTextSplitter(
                        separators=["\n\n", "\n", " ", ".", ",", ""
                                    "\u200b",  # Zero-width space
                                    "\uff0c",  # Fullwidth comma
                                    "\u3001",  # Ideographic comma
                                    "\uff0e",  # Fullwidth full stop
                                    "\u3002",  # Ideographic full stop
                                    ],
                        chunk_size=1000,
                        chunk_overlap=200,
                        add_start_index=True,
                        is_separator_regex=False)
            
    splitted_docs = splitter.split_documents(loaded_docs)
    return splitted_docs

def upsert_chromadb(splitted_docs, embedding, idx, collection_name, db_name):
    """
    Upserts data into Chromadb

    Args:
    splitted_docs List[Document]: A list of smaller chunks of parsed document.
    embedding: The embedding model.
    idx List[str]: A list of unique identifiers for each document.
    collection_name str: The name of the Chroma collection.
    db_name str: The name of the database.
    """
    vector_store = Chroma.from_documents(splitted_docs, embedding, ids=idx,
                          collection_name=collection_name,
                          persist_directory="./" + db_name
                         )
    return vector_store
