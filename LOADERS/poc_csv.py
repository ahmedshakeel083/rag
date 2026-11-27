from pathlib import Path

from langchain_community.document_loaders import CSVLoader

file_path = Path(__file__).resolve().parent / "customers.csv"

loader = CSVLoader(file_path=file_path)

docs = loader.load()

for rows in docs:
    print(rows, '\n')
    