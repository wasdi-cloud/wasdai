import httpx
import logging
import uvicorn
import os
from starlette.middleware.cors import CORSMiddleware

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp import Context
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
sConfigFilePath = os.getenv(
    "WASDI_CONFIG_PATH", 
    "C:\\WASDI\\GIT\\wasdai\\config.json"
)

if not (s_oConfig := WasdiConfig(sConfigFilePath)):
    logging.error("Failed to load configuration")
    raise RuntimeError(f"Could not load config from {sConfigFilePath}")

logging.info("Loading Embeddings")
if not (s_oEmbeddings := HuggingFaceEmbeddings(model_name="BAAI/bge-m3")):
    logging.error("Failed to load embeddings")
    raise RuntimeError("Could not load embeddings")

logging.info("Loading the vector store")
s_oVectorStore = Chroma(
        collection_name="embeddings", # "wasdi_docs",
        embedding_function=s_oEmbeddings,
        persist_directory=s_oConfig.chromaStore.persistDirectory
    )
if not s_oVectorStore:
    logging.error("Failed to load vector store")
    raise RuntimeError(f"Could not load vector store from {s_oConfig.chromaStore.persistDirectory}")

logging.info("Initializing the RAG chain")
s_sEndpoint = s_oConfig.aiAgent.llm_endpoint
s_sToken = s_oConfig.aiAgent.llm_token
s_sModelName = s_oConfig.aiAgent.llm_model

s_oLLM = ChatOpenAI(
    base_url=s_sEndpoint + "/v1",
    api_key=s_sToken,
    model=s_sModelName
)

s_oRetriever = s_oVectorStore.as_retriever()

# initialize the Flash Rerank Compressor for post-retrieval re-ranking
s_oCompressor = FlashrankRerank()
s_oCompressionRetriever = ContextualCompressionRetriever(
    base_compressor=s_oCompressor,
    base_retriever=s_oRetriever
)

s_sPromptTemplate = """Use the context provided to answer the user's question below. If you do not know the answer 
based on the context provided, tell the user that you do  not know the answer to their question based on the context 
provided and that you are sorry.

context: {context}
question: {query}
answer: """

s_oCustomRAGPrompt = PromptTemplate.from_template(s_sPromptTemplate)

s_oRAGChain = RAGChain(
    oLLM=s_oLLM,
    oRetriever=s_oCompressionRetriever,
    oPrompt=s_oCustomRAGPrompt
)

s_oMcpServer = FastMCP("wasdi-mcp-server", "0.1.0")

@s_oMcpServer.tool()
def hello(sName: str) -> str:
    """Says hello to someone, whose name is give as an input parameter."""
    return f"Hello {sName}!"


@s_oMcpServer.tool()
async def wasdiHello() -> str:
    """WASDI hello endpoint can be used to check if the service is up and running. 
    The call does not need any authentication. If it works, the API returns a json with 'stringValue': 'Hello Wasdi!!'. 
    If it does not work can return not found or not available or any other http error."""
    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get("https://www.wasdi.net/wasdiwebserver/rest/wasdi/hello")
        oResponse.raise_for_status()
        return oResponse.text
    
@s_oMcpServer.tool()
async def searchWasdiDocs(sUserPrompt: str) -> str:
    """
    Searches the internal WASDI documentation and knowledge base.
    Use this tool whenever the user asks for explanations about the system,
    how to use features, how to navigate the WASDI platform, general platform knowledge
    or general Earth Observation (EO) knowledge.
    """
    oResponse = s_oRAGChain.invokeRAGChain(sUserPrompt)
    return oResponse.content

    
    
@s_oMcpServer.tool()
async def get_workspaces(oContext: Context = None) -> str:
    """
    Returns the list of workspaces for the currently authenticated user.
    The workspaces being returned are those that the user has access to, either as owner or as collaborator.
    The AI agent should return to the user the total count of workspaces and the name of each workspace.
    """
    session_token = ""
    if oContext and oContext.request_context and oContext.request_context.request:
        oRequest = oContext.request_context.request
        session_token = oRequest.headers.get("x-session-token") or ""

    if not session_token:
        raise ValueError("Missing x-session-token header")
    
    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"https://www.wasdi.net/wasdiwebserver/rest/ws/byuser",
            headers={"x-session-token": session_token}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI get_workspaces call completed with status %s", oResponse.status_code)
        return oResponse.text
    

if __name__ == "__main__":
    # extract the ASGI web application from the MCP server and run it with Uvicorn
    # uvicorn is the server only responsible for accepting raw HTTP traffic
    # the web aapplication contains the businnes logic
    oApp = s_oMcpServer.streamable_http_app()

    # Comma-separated list of origins. Use "*" to allow all origins.
    sCorsOrigins = os.getenv("WASDI_CORS_ALLOW_ORIGINS", "*")
    aoCorsOrigins = [sOrigin.strip() for sOrigin in sCorsOrigins.split(",") if sOrigin.strip()]
    bAllowAllOrigins = "*" in aoCorsOrigins

    oApp.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if bAllowAllOrigins else aoCorsOrigins,
        allow_methods=["*"],
        allow_headers=["*"],
        # Browsers reject credentialed CORS responses with wildcard origins.
        allow_credentials=not bAllowAllOrigins,
    )

    uvicorn.run(oApp, host="0.0.0.0", port=7000)