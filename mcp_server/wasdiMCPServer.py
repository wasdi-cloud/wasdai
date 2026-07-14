import httpx
import logging
import uvicorn
import os
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
import urllib.parse

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp import Context
from mcp.server.transport_security import TransportSecuritySettings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from utils.WasdiConfig import WasdiConfig
from utils.LoggingConfiguration import setupLogging
from ai_agent.RAGChain import RAGChain
from utils.Utils import *
from mcp_server.modules.workflows import register_workflow_tools
from mcp_server.modules.applications import register_application_tools
from mcp_server.modules.workspaces import register_workspace_tools
from mcp_server.modules.developer import register_developer_tools
from mcp_server.modules.catalog import register_catalog_tools

sConfigFilePath = os.getenv(
    "WASDI_CONFIG_PATH", 
    "C:\\WASDI\\GIT\\wasdai\\config.json"
)

if not (s_oConfig := WasdiConfig(sConfigFilePath)):
    logging.error("Failed to load configuration")
    raise RuntimeError(f"Could not load config from {sConfigFilePath}")

setupLogging(s_oConfig.MCP_server.logLevel)

# INITIALIZATION
logging.info("Loading configuration")
s_sWasdiApiUrl = os.getenv("WASDI_API_URL", "https://www.wasdi.net/wasdiwebserver").rstrip("/")


logging.info("Loading Embeddings")
s_oEmbeddingConfig = getattr(s_oConfig, "embedding", None)
s_sEmbeddingModelName = getattr(s_oEmbeddingConfig, "modelName", "BAAI/bge-m3")
s_sHuggingFaceToken = getattr(s_oEmbeddingConfig, "huggingface_token", "")

aoEmbeddingArgs = {
    "model_name": s_sEmbeddingModelName,
}

if s_sHuggingFaceToken:
    # Keep HF Hub authentication explicit for higher rate limits and stable downloads.
    os.environ["HF_TOKEN"] = s_sHuggingFaceToken
    aoEmbeddingArgs["model_kwargs"] = {"token": s_sHuggingFaceToken}

if not (s_oEmbeddings := HuggingFaceEmbeddings(**aoEmbeddingArgs)):
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
s_oCompressionRetriever = s_oRetriever

s_sPromptTemplate = """Use the context to answer the user's question. You are a WASDI and Earth Observation (EO) expert, you help users to use WASDI including interface, coding new apps, using existing apps. Use searchWasdiDocs to search the documentation.
All functional execution tools require a unique alphanumeric sWorkspaceId. When a user refers to a workspace by its human-readable name, you MUST first invoke get_workspaces_by_user to list all available environments. Map the human-specified name to the correct workspaceId before calling any downstream data or execution tools.
When a user asks to run a WASDI processor or application, do not attempt to guess or synthesize the input parameters. You must first invoke get_processor_ui or get_single_deployed_processor to inspect the sample parameter structures, and read get_processor_help to ensure semantic correctness before generating the execution payload.
If you do not know the answer based on the context provided, tell the user that you do  not know the answer to their question based on the context provided 
and that you are sorry.
context: {context}
question: {query}
answer: """

s_oCustomRAGPrompt = PromptTemplate.from_template(s_sPromptTemplate)

s_oRAGChain = RAGChain(
    oLLM=s_oLLM,
    oRetriever=s_oCompressionRetriever,
    oPrompt=s_oCustomRAGPrompt
)

s_oMcpServer = FastMCP("wasdi-mcp-server", "0.1.0", transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))

register_workspace_tools(s_oMcpServer, s_sWasdiApiUrl)
register_application_tools(s_oMcpServer, s_sWasdiApiUrl)
register_developer_tools(s_oMcpServer, s_sWasdiApiUrl)
register_workflow_tools(s_oMcpServer, s_sWasdiApiUrl)
register_catalog_tools(s_oMcpServer, s_sWasdiApiUrl)

oApp = s_oMcpServer.streamable_http_app()

sCorsOrigins = os.getenv("WASDI_CORS_ALLOW_ORIGINS", "*")
aoCorsOrigins = [sOrigin.strip() for sOrigin in sCorsOrigins.split(",") if sOrigin.strip()]
bAllowAllOrigins = "*" in aoCorsOrigins

oApp.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if bAllowAllOrigins else aoCorsOrigins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=not bAllowAllOrigins,
)

oApp.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "testmcp.wasdi.net", "mcp.wasdi.net", "ai-mcp", "*.wasdi.net"] if bAllowAllOrigins else aoCorsOrigins
)

@s_oMcpServer.tool()
def hello(sName: str) -> str:
    """Says hello to someone, whose name is give as an input parameter. Use this to check if the MCP Server is working. """
    return f"Hello {sName}!"

@s_oMcpServer.tool()
async def wasdi_hello() -> str:
    """WASDI hello can be used to check if the WASDI service is up and running. 
    If it works, the API returns a json with 'stringValue': 'Hello Wasdi!!'. 
    If it does not work can return not found or not available or any other http error."""
    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(f"{s_sWasdiApiUrl}/rest/wasdi/hello")
        oResponse.raise_for_status()
        return oResponse.text
    
@s_oMcpServer.tool()
async def search_wasdi_docs(sUserPrompt: str) -> str:
    """
    Searches the internal WASDI documentation and knowledge base.
    Use this tool whenever the user asks for explanations about the system,
    how to use features, how to navigate the WASDI platform, general platform knowledge
    or general Earth Observation (EO) knowledge. The agent can use it also to understand better the functionalities of the other tools exposed.
    sUserPrompt is the question or query from the user that needs to be answered using the WASDI documentation.
    """
    oResponse = s_oRAGChain.invokeRAGChain(sUserPrompt)
    return oResponse.content


if __name__ == "__main__":
    uvicorn.run(oApp, host="0.0.0.0", port=7000)
