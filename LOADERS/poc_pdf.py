from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

pdf_path = Path(__file__).resolve().parent / "sample.pdf"

loader = PyPDFLoader(file_path=str(pdf_path))

docs = loader.load()

print(docs[0].page_content)