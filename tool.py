from dotenv import load_dotenv
from pydantic import BaseModel
load_dotenv()
import vertexai

vertexai.init(project="Generative Language API Key", location="us-central1")
import hashlib
import os
from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader,WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage,ToolMessage
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import Annotated, TypedDict
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_community.tools import DuckDuckGoSearchResults

@tool(description="Search the web for information related to the job description or resume.")
def search_tool(query: str) -> str:
    return DuckDuckGoSearchResults().run(query)


llm=init_chat_model(model="gpt-4o-mini")

model=llm.bind_tools([search_tool])
print(model.invoke("what is info about apple?"))