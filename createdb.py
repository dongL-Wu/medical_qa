from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter,CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import os

# Define the metadata extraction function.
def metadata_func(record: dict, metadata: dict) -> dict:

    metadata["year"] = record.get("pub_date").get('year')
    metadata["month"] = record.get("pub_date").get('month')
    metadata["day"] = record.get("pub_date").get('day')
    metadata["title"] = record.get("article_title")
    
    return metadata

loader = JSONLoader(
    file_path='./data.json',
    jq_schema='.[]',
    content_key='article_abstract',
    metadata_func=metadata_func)
data = loader.load()

print(f"{len(data)} pubmed articles are loaded!")


text_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=50)
chunks = text_splitter.split_documents(data)

# 加载开源词向量模型
embeddings = HuggingFaceEmbeddings(model_name="/root/model/sentence-transformer")


# 构建向量数据库
# 定义持久化路径
persist_directory = 'data_base/vector_db/chroma'
# 加载数据库
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)
# 将加载的向量数据库持久化到磁盘上
vectordb.persist()