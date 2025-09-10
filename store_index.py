from src.helper import repo_ingestion, load_repo, text_splitter, load_embedding
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
import os
from huggingface_hub import login
# Load token từ .env
load_dotenv()
token = os.environ["HUGGINGFACEHUB_API_TOKEN"]

# Login trực tiếp bằng code
login(token=token)



# url = "https://github.com/Vinhnv0901/universal-document-QA_with_Llama2.git"

# repo_ingestion(url)


documents = load_repo("repo/")
text_chunks = text_splitter(documents)
embeddings = load_embedding()



#storing vector in choramdb
vectordb = Chroma.from_documents(text_chunks, embedding=embeddings, persist_directory='./db')
vectordb.persist()