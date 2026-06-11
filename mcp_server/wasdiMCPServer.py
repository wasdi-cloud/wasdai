import httpx
import logging
import uvicorn

from mcp.server.fastmcp import FastMCP
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from utils.WasdiConfig import WasdiConfig
from utils.LoggingConfiguration import setupLogging
from ai_agent.RAGChain import RAGChain

setupLogging()

# -----------------------
# INITIALIZATION
# -----------------------
logging.info("Loading configuration")
sConfigFilePath = "C:\\WASDI\\GIT\\wasdai\\config.json"
if not (oConfig := WasdiConfig(sConfigFilePath)):
    logging.error("Failed to load configuration")
    raise RuntimeError(f"Could not load config from {sConfigFilePath}")

logging.info("Loading Embeddings")
if not (oEmbeddings := HuggingFaceEmbeddings(model_name="BAAI/bge-m3")):
    logging.error("Failed to load embeddings")
    raise RuntimeError("Could not load embeddings")

logging.info("Loading the vector store")
oVectorStore = Chroma(
        collection_name="embeddings", # "wasdi_docs",
        embedding_function=oEmbeddings,
        persist_directory=oConfig.chromaStore.persistDirectory
    )
if not oVectorStore:
    logging.error("Failed to load vector store")
    raise RuntimeError(f"Could not load vector store from {oConfig.chromaStore.persistDirectory}")

logging.info("Initializing the RAG chain")
sEndpoint = oConfig.aiAgent.llm_endpoint
sToken = oConfig.aiAgent.llm_token
sModelName = oConfig.aiAgent.llm_model

oLLM = ChatOpenAI(
    base_url=sEndpoint + "/v1",
    api_key=sToken,
    model=sModelName
)

oRetriever = oVectorStore.as_retriever()

# initialize the Flash Rerank Compressor for post-retrieval re-ranking
oCompressor = FlashrankRerank()
oCompressionRetriever = ContextualCompressionRetriever(
    base_compressor=oCompressor,
    base_retriever=oRetriever
)
sPromptTemplate = """Use the context provided to answer the user's question below. If you do not know the answer 
based on the context provided, tell the user that you do  not know the answer to their question based on the context 
provided and that you are sorry.

context: {context}

question: {query}

answer: """

oCustomRAGPrompt = PromptTemplate.from_template(sPromptTemplate)

oRAGChain = RAGChain(
    oLLM=oLLM,
    oRetriever=oCompressionRetriever,
    oPrompt=oCustomRAGPrompt
)

oMcpServer = FastMCP("wasdi-mcp-server", "0.1.0")

@oMcpServer.tool()
def hello(sName: str) -> str:
    """Says hello to someone, whose name is give as an input parameter."""
    return f"Hello {sName}!"


@oMcpServer.tool()
async def wasdiHello() -> str:
    """Calls the WASDI hello enpoint to check if the service is up and running."""
    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get("https://www.wasdi.net/wasdiwebserver/rest/wasdi/hello")
        oResponse.raise_for_status()
        return oResponse.text
    
@oMcpServer.tool()
async def searchWasdiDocs(sQuery: str) -> str:
    """
    Searches the internal WASDI documentation and knowledge base.
    Use this tool whenever the user asks for explanations about the system,
    how to use features, how to navigate the WASDI platform, or general platform knowledge.
    """
    
    
@oMcpServer.tool()
async def get_workspaces() -> str:
    """
    Returns the list of workspaces for the currently authenticated user.
    The workspaces being retunned are those that the user has access to, either as owner or as collaborator.
    The AI agent should return to the user the total count of workspaces and the name of each workspace.
    """
    session_token  = "xxx"
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://www.wasdi.net/wasdiwebserver/rest/ws/byuser",
            headers={"x-session-token": session_token}
        )
        response.raise_for_status()
        print(response)
        return response.text

if __name__ == "__main__":
    # extract the ASGI web application from the MCP server and run it with Uvicorn
    # uvicorn is the server only responsible for accepting raw HTTP traffic
    # the web aapplication contains the businnes logic
    oApp = oMcpServer.streamable_http_app
    uvicorn.run(oApp, host="0.0.0.0", port=8000)