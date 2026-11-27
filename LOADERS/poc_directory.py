from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="books", # Go in the books directory
    glob="*.pdf",   # Load all pdf files, If just "*" means all files
    loader_cls=PyPDFLoader
)

# docs = loader.load()  this will create 1 document object for 1 page pdf Example if u have 100 pages in book 1 and 200 pages in book 2 than output will be 300 document objects all together at once and this is not a CORRECT WAY to do it use lazy_load to resolve this


# print(docs[0].page_content)

# print(docs[0].metadata)

docs = loader.lazy_load()

for document in docs:
    print(document.metadata)