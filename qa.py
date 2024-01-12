from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
from llm import InternLM_LLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# 定义 Embeddings
embeddings = HuggingFaceEmbeddings(model_name="/root/model/sentence-transformer")

# 向量数据库持久化路径
persist_directory = 'data_base/vector_db/chroma'

# 加载数据库
vectordb = Chroma(
    persist_directory=persist_directory, 
    embedding_function=embeddings
)


llm = InternLM_LLM(model_path = "/root/model/Shanghai_AI_Laboratory/internlm-chat-7b")


# 我们所构造的 Prompt 模板
template = """Answer the question based only on the following context:
{context}
You are allowed to rephrase the answer based on the context. 
Question: {question}
"""

# 调用 LangChain 的方法来实例化一个 Template 对象，该对象包含了 context 和 question 两个变量，在实际调用时，这两个变量会被检索到的文档片段和用户提问填充
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],template=template)

qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectordb.as_retriever(),return_source_documents=True,chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})



# 检索问答链回答效果
question = "What are the safest cryopreservation methods?"
result = qa_chain({"query": question})
print("检索问答链回答 question 的结果：")
print(result["result"])

# 仅 LLM 回答效果
result_2 = llm(question)
print("大模型回答 question 的结果：")
print(result_2)