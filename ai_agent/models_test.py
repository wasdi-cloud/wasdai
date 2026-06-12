from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent


import os
import logging
import asyncio
from dotenv import load_dotenv
from openai import OpenAI

from .RAGChain import RAGChain
from utils.WasdiConfig import WasdiConfig

logging.basicConfig(level=logging.INFO)

load_dotenv()

sConfigFilePath = "C:\\WASDI\\GIT\\wasdai\\config.json"
oConfig = WasdiConfig(sConfigFilePath)

sToken = oConfig.aiAgent.llm_token
sEndpoint = oConfig.aiAgent.llm_endpoint
sModelName = oConfig.aiAgent.llm_model

def prompt(sUserPrompt: str):
    oClient = OpenAI(
        base_url=sEndpoint,
        api_key=sToken
    )

    oResponse = oClient.chat.completions.create(
        messages=[
            {"role": "system", 
             "content": "You are a sarcastic assistant." # set the tone for replies of the model
            }, 
            {"role": "user", 
             "content": sUserPrompt
            },
        ],
        model=sModelName,
    )

    logging.info(oResponse.choices[0].message.content)


def promptRAG(sQuestion: str):
    with open("wasdi_user_manual.txt", "r", encoding="utf-8") as oFile:
        sWasdiDocsText = oFile.read()

    # text splitter initialization
    oTextSplitter = RecursiveCharacterTextSplitter( # looks like this is the industry standard for text splitting
        chunk_size=1000, 
        chunk_overlap=200, # TODO: what is that
        length_function=len
    )

    # create documents (chunks) from file
    oDocuments = oTextSplitter.create_documents([sWasdiDocsText])

    # get the embeddings model
    sEndpoint = oConfig.aiAgent.llm_endpoint
    oEmbeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=sEndpoint)

    # initialize Chroma db as a vector store
    oVectorStore = Chroma(
        collection_name="embeddings", # "wasdi_docs",
        embedding_function=oEmbeddings,
        persist_directory="C:\\WASDI\\ChromaDB" # "./chroma_db"
    )

    # save the document chunks to the vector store
    logging.info(f"embeddings count: {oVectorStore._collection.count()}")

    """
    if oVectorStore._collection.count() == 0: # if the collection is empty, add the documents   
        asDocumentIds = oVectorStore.add_documents(oDocuments)
    else:
        logging.info("Collection already has documents, skipping adding documents to the vector store.")
    """

    """
    aoResults = oVectorStore.similarity_search(sQuestion, k=3) # get the 3 most similar chunks to the question
    for oResult in aoResults:
        logging.info(f"\n\n* {oResult.page_content} [{oResult.metadata}]\n\n") # print the content of the chunks
    """

    # set the retriever
    oRetriever = oVectorStore.as_retriever()

    # initialize the LLM instance
    sToken = oConfig.aiAgent.llm_token
    sModelName = oConfig.aiAgent.llm_model

    oLLM = ChatOpenAI(
        base_url=sEndpoint + "/v1",
        api_key=sToken,
        model=sModelName
    )

    sPromptTemplate = """Use the context provided to answer the user's question below. If you do not know the answer 
    based on the context provided, tell the user that you do  not know the answer to their question based on the context 
    provided and that you are sorry.
    
    context: {context}
    
    question: {query}
    
    answer: """

    oCustomRAGPrompt = PromptTemplate.from_template(sPromptTemplate)

    # Create the RAG Chain
    rag_chain = (
        {"context": oRetriever | format_docs, "query": RunnablePassthrough()}
        | oCustomRAGPrompt
        | oLLM
        | StrOutputParser()
    )

    # Query the RAG Chain
    sAnswer =rag_chain.invoke(sQuestion)
    logging.info(f"Answer: {sAnswer}")

# with ptr-retrieval query rewriting
def promptRAG_Evolution(sQuestion: str):
    """
    with open("wasdi_user_manual.txt", "r", encoding="utf-8") as oFile:
        sWasdiDocsText = oFile.read()


    # text splitter initialization
    oTextSplitter = RecursiveCharacterTextSplitter( # looks like this is the industry standard for text splitting
        chunk_size=1000, 
        chunk_overlap=200, # TODO: what is that
        length_function=len
    )

    # create documents (chunks) from file
    oDocuments = oTextSplitter.create_documents([sWasdiDocsText])
    """

    # get the embeddings model
    sEndpoint = oConfig.aiAgent.llm_endpoint
    oEmbeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # initialize Chroma db as a vector store
    oVectorStore = Chroma(
        collection_name="embeddings", # "wasdi_docs",
        embedding_function=oEmbeddings,
        persist_directory="C:\\WASDI\\ChromaDB"
    )

    # save the document chunks to the vector store
    logging.info(f"embeddings count: {oVectorStore._collection.count()}")
    """
    if oVectorStore._collection.count() == 0: # if the collection is empty, add the documents   
        asDocumentIds = oVectorStore.add_documents(oDocuments)
    else:
        logging.info("Collection already has documents, skipping adding documents to the vector store.")
    """

    # set the retriever
    oRetriever = oVectorStore.as_retriever()

    # initialize the Flash Rerank Compressor for post-retrieval re-ranking
    oCompressor = FlashrankRerank()
    oCompressionRetriever = ContextualCompressionRetriever(
        base_compressor=oCompressor,
        base_retriever=oRetriever
    )

    # initialize the LLM instance
    sToken = oConfig.aiAgent.llm_token
    sModelName = oConfig.aiAgent.llm_model

    oLLM = ChatOpenAI(
        base_url=sEndpoint + "/v1",
        api_key=sToken,
        model=sModelName
    )


    logging.info(f"Received the rewritten query");


    sPromptTemplate = """Use the context provided to answer the user's question below. If you do not know the answer 
    based on the context provided, tell the user that you do  not know the answer to their question based on the context 
    provided and that you are sorry.
    
    context: {context}
    
    question: {query}
    
    answer: """

    oCustomRAGPrompt = PromptTemplate.from_template(sPromptTemplate)

    # inizialise the custom RAG chain
    oRAGChain = RAGChain(
        oLLM=oLLM,
        oRetriever=oCompressionRetriever,
        oPrompt=oCustomRAGPrompt
    )


    oResponse = oRAGChain.invokeRAGChain(sQuestion)

    logging.info(f"Answer: {oResponse.content}")



def format_docs(aoDocs):
    return "\n\n".join(oDoc.page_content for oDoc in aoDocs)


async def promptMCP(sUserPrompt: str):
    oLLM = ChatOpenAI(
        base_url=sEndpoint + "/v1",
        api_key=sToken,
        model="llama3.1:8b"
    )

    """
    oClient = MultiServerMCPClient({
        "wasdi": {
            "command": "python",
            "args": ["C:\\WASDI\\GIT\\wasdai\\mcp_server\\wasdiMCPServer.py"],
            "transport": "stdio"
        }
    })
    """
    oClient = MultiServerMCPClient({
        "wasdi": {
            "url": "http://localhost:7000/mcp",
            "transport": "http"
        }
    })
    tools = await oClient.get_tools()
    agent = create_agent(model=oLLM, tools=tools)

    oResult = await agent.ainvoke({
        "messages": [{"role": "user", "content": sUserPrompt}]
    })

    sResponse = oResult["messages"][-1].content
    logging.info(f"PROMPT: {sUserPrompt}")
    logging.info(f"Response from MCP agent: {sResponse}")
    return sResponse



if __name__ == "__main__":
    # the date will not be correct, but it is just to test the connection to the model
    # the problem is that the model does not have access to the current date because it is a local model
    # so it does not access internet connection to get the current date
    sQuestion = "How do I search for a product in WASDI?" # does not reply correctly
    sQuestion2 = "How do I share a workspace with one of my colleagues?"
    sQuestion3 = "Which application can I use in WASDI to get floods maps in open areas?"
    sQuestion4 = "What is Automatic AUTOWADE algorithm for floods in WASDI?"
    

    # promptRAG(sQuestion2)
    # promptRAG_Evolution(sQuestion3)
    sMCPPrompt1 = "Call the WASDI hello endpoint"
    sMCPPrompt2 = "Give me the list of my workspaces in WASDI"
    sMCPPrompt3 = "Give me the list of my workspaces' names in WASDI, together with the node id"
    asyncio.run(promptMCP(sMCPPrompt2))