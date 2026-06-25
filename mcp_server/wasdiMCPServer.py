from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
import httpx
import logging
import uvicorn
import os
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

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

setupLogging()

# INITIALIZATION
logging.info("Loading configuration")
sConfigFilePath = os.getenv(
    "WASDI_CONFIG_PATH", 
    "C:\\WASDI\\GIT\\wasdai\\config.json"
)
s_sWasdiApiUrl = os.getenv("WASDI_API_URL", "https://www.wasdi.net/wasdiwebserver").rstrip("/")

if not (s_oConfig := WasdiConfig(sConfigFilePath)):
    logging.error("Failed to load configuration")
    raise RuntimeError(f"Could not load config from {sConfigFilePath}")

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

s_sPromptTemplate = """Use the context to answer the user's question. You are a WASDI and Earth Observation (EO) expert, you help users to use WASDI interface and to code WASDI applications using wasdi libraries. searchWasdiDocs should help to search the documentation where the architecture, the main entities, the APIs and the libraries are documented.
The comment of each method try to describe the purpose of the method, the input parameters and the output. The comment can be used to understand how to use the method and what is the expected result. 
If you do not know the answer based on the context provided, tell the user that you do  not know the answer to their question based on the context provided and that you are sorry.

context: {context}
question: {query}
answer: """

s_oCustomRAGPrompt = PromptTemplate.from_template(s_sPromptTemplate)

s_oRAGChain = RAGChain(
    oLLM=s_oLLM,
    oRetriever=s_oCompressionRetriever,
    oPrompt=s_oCustomRAGPrompt
)

s_oMcpServer = FastMCP("wasdi-mcp-server", 
                       "0.1.0", 
                       transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))
@asynccontextmanager
async def mcp_lifespan(app: FastAPI):
    # This keeps the streamable_http session connections open globally
    async with s_oMcpServer.run_transport_managers():
        yield
        
oApp = s_oMcpServer.streamable_http_app(lifespan=mcp_lifespan)

sCorsOrigins = os.getenv("WASDI_CORS_ALLOW_ORIGINS", "*")
aoCorsOrigins = [sOrigin.strip() for sOrigin in sCorsOrigins.split(",") if sOrigin.strip()]
bAllowAllOrigins = "*" in aoCorsOrigins

allowed_hosts = ["localhost", "127.0.0.1", "testmcp.wasdi.net", "mcp.wasdi.net", "ai-mcp", "*.wasdi.net"]
if not bAllowAllOrigins:
    # Merge CORS domains with your trusted system hostnames
    allowed_hosts = list(set(allowed_hosts + aoCorsOrigins))
else:
     # Allow all during open CORS mode to prevent drops
    allowed_hosts = ["*"]


oApp.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if bAllowAllOrigins else aoCorsOrigins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=not bAllowAllOrigins,
)

oApp.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=allowed_hosts
)

@s_oMcpServer.tool()
def hello(sName: str) -> str:
    """Says hello to someone, whose name is give as an input parameter."""
    return f"Hello {sName}!"


def getSessionToken(oContext=None) -> str:
    """Utility function to get the session token from the current context. 
    This is needed to inject the session token into the headers of the HTTP calls made by the MCP tools."""

    if oContext is None:
        logging.warning("getSessionToken called without context")
        return ""

    if oContext and oContext.request_context and oContext.request_context.request:
        oRequest = oContext.request_context.request
        sSessionToken = oRequest.headers.get("x-session-token") or ""
        return sSessionToken
    return ""


async def getNodeUrlForWorkspace(sWorkspaceId: str, sSessionToken: str) -> str:
    """Utility function to resolve the correct node URL for a workspace.
    This is used for node-based APIs that store data per node, not on the main server.
    
    Args:
        sWorkspaceId: The workspace ID for which to resolve the node URL
        sSessionToken: The session token for authentication
        
    Returns:
        The node base URL (apiUrl from workspace details), or falls back to main server if resolution fails
    """
    import json
    
    try:
        async with httpx.AsyncClient() as oClient:
            oWsResponse = await oClient.get(
                f"{s_sWasdiApiUrl}/rest/ws/getws",
                params={"workspace": sWorkspaceId},
                headers={"x-session-token": sSessionToken}
            )
            oWsResponse.raise_for_status()
            oWsData = json.loads(oWsResponse.text)
            sNodeUrl = oWsData.get("apiUrl", s_sWasdiApiUrl)
            logging.debug("Resolved node URL for workspace %s: %s", sWorkspaceId, sNodeUrl)
            return sNodeUrl
    except Exception as e:
        logging.warning("Failed to resolve node URL for workspace %s, falling back to main server: %s", sWorkspaceId, str(e))
        return s_sWasdiApiUrl


async def getWorkspaceIdForProcessWorkspace(sProcessObjId: str, sSessionToken: str) -> str:
    """Resolve the workspace id associated to a process workspace id.
    Returns an empty string if the process or workspace cannot be resolved.
    """
    if not sProcessObjId:
        return ""

    import json

    try:
        async with httpx.AsyncClient() as oClient:
            oProcessResponse = await oClient.get(
                f"{s_sWasdiApiUrl}/rest/process/byid",
                params={"procws": sProcessObjId},
                headers={"x-session-token": sSessionToken},
            )
            oProcessResponse.raise_for_status()
            oProcessData = json.loads(oProcessResponse.text)
            sWorkspaceId = oProcessData.get("workspaceId") or oProcessData.get("workspace") or ""
            if not sWorkspaceId:
                logging.warning("Workspace id not found in process details for process %s", sProcessObjId)
            return sWorkspaceId
    except Exception as e:
        logging.warning("Failed to resolve workspace id from process %s: %s", sProcessObjId, str(e))
        return ""


async def getNodeUrlForProcessWorkspace(
    sProcessObjId: str,
    sSessionToken: str,
    sWorkspaceId: str = None,
) -> str:
    """Resolve node URL for a process workspace, with optional explicit workspace id.

    Uses the provided workspace id when available, otherwise resolves it from process details.
    Falls back to the main server URL if resolution fails.
    """
    sResolvedWorkspaceId = sWorkspaceId
    if not sResolvedWorkspaceId:
        sResolvedWorkspaceId = await getWorkspaceIdForProcessWorkspace(sProcessObjId, sSessionToken)

    if sResolvedWorkspaceId:
        return await getNodeUrlForWorkspace(sResolvedWorkspaceId, sSessionToken)

    logging.warning("Failed to resolve node URL for process %s, falling back to main server", sProcessObjId)
    return s_sWasdiApiUrl

@s_oMcpServer.tool()
async def wasdiHello() -> str:
    """WASDI hello endpoint can be used to check if the service is up and running. 
    The call does not need any authentication. If it works, the API returns a json with 'stringValue': 'Hello Wasdi!!'. 
    If it does not work can return not found or not available or any other http error."""
    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(f"{s_sWasdiApiUrl}/rest/wasdi/hello")
        oResponse.raise_for_status()
        return oResponse.text
    
@s_oMcpServer.tool()
async def searchWasdiDocs(sUserPrompt: str) -> str:
    """
    Searches the internal WASDI documentation and knowledge base.
    Use this tool whenever the user asks for explanations about the system,
    how to use features, how to navigate the WASDI platform, general platform knowledge
    or general Earth Observation (EO) knowledge. The agent can use it also to understand better the functionalities of the other tools exposed.
    sUserPrompt is the question or query from the user that needs to be answered using the WASDI documentation.
    """
    oResponse = s_oRAGChain.invokeRAGChain(sUserPrompt)
    return oResponse.content

    
    
@s_oMcpServer.tool()
async def get_workspaces_by_user(oContext: Context = None) -> str:
    """
    Returns the list of workspaces for the currently authenticated user.
    The workspaces being returned are those that the user has access to, either as owner or as collaborator.
    Can be used if the agent needs to list the workspaces of a user, or search if a workspace with a specific name exists.
    Return a list of WorkspaceListInfoViewModel in JSON format: properties are 
    workspaceId: unique id of the workspace
    workspaceName: name of the workspace
    ownerUserId: user id of the owner of the workspace,
    sharedUsers: list of strings with user ids of the users that the workspace is shared with
    nodeCode: code of the node where the workspace is located
    creationDate: date of creation of the workspace
    storageSize: storage size of the workspace in bytes 
    isPublic: boolean that indicates if the workspace is public or private
    readOnly: boolean that indicates if the workspace is read only
    activeNode: boolean that indicates if the node where the workspace is located is active or not
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")
    
    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/ws/byuser",
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI get_workspaces call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_workspace_details(sWorkspaceId: str, oContext: Context = None) -> str:
    """
    Returns the workspace editor view model for a specific workspace.
    This mirrors WorkspaceResource.getWorkspaceEditorViewModel and is useful when the agent
    needs the full information about a workspace, including node, permissions, dates, storage size, or sharing details.
    
    The call returns a JSON with the following properties:

    workspaceId: unique id of the workspace
    name: name of the workspace
    userId: user id of the owner of the workspace
    apiUrl: base url of the node where the workspace is located, used for some specific calls that need to target the node directly
    creationDate: date of creation of the workspace
    lastEditDate: date of the last modification of the workspace
    sharedUsers: list of user ids of the users that the workspace is shared with
    nodeCode: code of the node where the workspace is located
    activeNode: boolean that indicates if the node where the workspace is located is active or not
    processesCount: number of processes in the workspace
    cloudProvider: name of the cloud provider where the node that host the workspace is located
    slaLink: link to the SLA of the cloud that host the workspace
    storageSize: storage size of the workspace in bytes
    isPublic: boolean that indicates if the workspace is public or private
    readOnly: boolean that indicates if the workspace is read only
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/ws/getws",
            params={"workspace": sWorkspaceId},
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI get_workspace_details call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_workspace_name_by_id(sWorkspaceId: str, oContext: Context = None) -> str:
    """
    Returns the workspace name for a given workspace id.
    Can be used when the agent needs to get the name of a workspace starting from its id.
    sWorkspaceId: is the unique workspace id that we are searching to get the name.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/ws/wsnamebyid",
            params={"workspace": sWorkspaceId},
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI get_workspace_name_by_id call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def create_new_workspace(sName: str = None, sNodeCode: str = None, oContext: Context = None) -> str:
    """
    Creates a new workspace for the authenticated user. Can be used by the agent to create a new workspace for the user. The workspaceId is the real key. The names are unique: if the name already exists, WASDI will add (1) or (2) etc. 
    
    Inputs:
    sName: is the name of the workspace to be created.
    sNodeCode: is the code of the node where the workspace will be created. If not provided, the workspace will be created in a node selected by WASDI according the access rights.

    the call returns null in case of errors or a JSON object with:

    IntValue: ignored in this API
    StringValue: the id of the newly created workspace, if the call is successful. The id is a string that uniquely identifies the workspace and can be used in other calls to get information about the workspace or to perform actions on it.
    DoubleValue: ignored in this API
    BoolValue: True if the workspace is created successfully.
        
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/ws/create",
            params={"name": sName, "node": sNodeCode},
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI createWorkspace call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def share_workspace_with_user(sWorkspaceId: str, sDestinationUserId: str, sRights: str = None, oContext: Context = None) -> str:
    """
    Shares a workspace with another user. If the target user does not exists or is already included, it will fail.
    The agent can use this API to share a workspace with another user, for example when the user asks to share a workspace with a colleague or with a group of users. 
    The agent can also use this API to change the access rights of a user that already has access to the workspace, for example to give write access to a user that currently has only read access: this must be done before removing the existing share and then creating a new one

    Inputs
    sWorkspaceId: is the unique id of the workspace to be shared.
    sDestinationUserId: is the user id of the user with whom the workspace will be shared.
    sRights: is the level of access that the destination user will have on the workspace. It can be "read" for read-only access or "write" for read and write access. If not provided, the default access level is "read".

    the call returns null in case of errors or a JSON object with:

    IntValue: The http code of the response, 200 if the workspace is shared successfully and different codes in case of errors
    StringValue: a message describing the error or the success of the operation
    DoubleValue: ignored in this API
    BoolValue: True if the workspace is shared successfully, False in case of errors

    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    if not sDestinationUserId:
        raise ValueError("Missing destination user id")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.put(
            f"{s_sWasdiApiUrl}/rest/ws/share/add",
            params={"workspace": sWorkspaceId, "userId": sDestinationUserId, "rights": sRights},
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI shareWorkspace call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_users_with_access_to_workspace(sWorkspaceId: str, oContext: Context = None) -> str:
    """
    Returns the list of users that have access to a workspace.
    The agent can use it to tell to the user or check the users that can access to workspace

    Input
    sWorkspaceId: is the unique id of the workspace for which we want to get the list of users that have access to it.

    Output
    the call returns an empty array in case of errors or an array JSON object with:
        workspaceId: the unique id of the workspace
        userId: the user id of the user that has access to the workspace
        ownerId: the user id of the owner of the workspace
        permissions: the level of access that the user has on the workspace, it can be "read" for read-only access or "write" for read and write access
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/ws/share/byworkspace",
            params={"workspace": sWorkspaceId},
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getEnabledUsersSharedWorksace call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def add_product_to_workspace(sProductName: str, sWorkspaceId: str, oContext: Context = None) -> str:
    """
    Adds a product to a workspace. The API works on database, not on the real file. The file that is going to be added must be present in the workspace folder but this API does not check if it really is there or not.
    In WASDI all files are represented as products, even if they are not EO products, so this API can be used to add any file to the workspace, as long as the file is present in the local node workspace folder and the name of the file is provided as an input parameter.
    The path is always relative to the root of the workspace that host the product.
    A typical use case of this API is when the agent needs to add a file that is generated during the execution of a process to the workspace, in order to make it available for the user in the WASDI interface and for other processes that can be executed after.
    Usually the agent generates a file in the local node workspace folder, then it uses this API to add the file to the workspace, providing the name of the file and the id of the workspace as input parameters.
    This is a node-based API.

    Input
    sProductName: is the name of the product to be added to the workspace, it must be present in the local node workspace folder
    sWorkspaceId: is the unique id of the workspace to which the product will be added

    Output
    the call returns null in case of errors or a JSON object with:

    IntValue: ignored in this API
    StringValue: ignored in this API
    DoubleValue: ignored in this API
    BoolValue: True if the product is added successfully to the workspace, False otherwise.

    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProductName:
        raise ValueError("Missing product name")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    # Resolve the node URL for this workspace
    sNodeUrl = await getNodeUrlForWorkspace(sWorkspaceId, sSessionToken)

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{sNodeUrl}/rest/product/addtows",
            params={"name": sProductName, "workspace": sWorkspaceId},
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI addProductToWorkspace call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_product_details_by_product_name(sProductName: str, sWorkspaceId: str, oContext: Context = None) -> str:
    """
    Returns a product view model by file name. The Product must exists in the workspace. The path of the product is always relative to the root of the workspace.
    This is a node-based API.

    Input
    sProductName: is the name of the product to be searched in the workspace, it must be present in the workspace
    sWorkspaceId: is the unique id of the workspace where the product is located

    Output
    the call returns null in case of errors or a JSON object with:

    bbox: optional property with the bounding box of the product, in case the product is an EO product with georeferenced data
    name: is the file name without extension
    description: optional property with the description of the product, if it is provided by the user when the product is created or edited
    fileName: is the file name with extension
    productFriendlyName: is a name that the user can assign to this product
    metadataFileCreated: metadata are not read by defualt. If the user wants to access it, a file is created in the dedicated wasdi folder. This property is a boolean true if the metadata has  been generated, false otherwise.
    metadataFileReference: if the metadataFileCreated is true, this property contains the path to the metadata file that has been generated. The path is relative to the metadata wasdi folder on the server
    metadata: real metadata if the metadataFileCreated is true. This property is not provided by default because it can be very heavy, especially for products with a lot of metadata, so it is better to read the metadata only when it is needed, using the metadataFileReference property to access the metadata file.
    bandsGroups: optional property with the bands groups of the product, if it is an EO product with multiple bands
    style: optional property with the name of the style of the product. Styles are Geoserver styles the user can upload in wasdi. If the product has a style assigned, it means that the user has uploaded a style in wasdi and assigned it to this product, so the style property contains the name of the style that is assigned to the product. The agent can use this information to suggest to the user to use this style when visualizing the product in wasdi, or to use this style as a reference when generating a new style for this product.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProductName:
        raise ValueError("Missing product name")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    # Resolve the node URL for this workspace
    sNodeUrl = await getNodeUrlForWorkspace(sWorkspaceId, sSessionToken)

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{sNodeUrl}/rest/product/byname",
            params={"name": sProductName, "workspace": sWorkspaceId},
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getByProductName call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_detailed_list_of_products_by_workspace(sWorkspaceId: str, oContext: Context = None) -> str:
    """
    Returns the detailed list of products in a workspace. 
    This mirrors ProductResource.getListByWorkspace and is useful when the agent needs to get the full list of products in a workspace with all the details, including metadata, styles, bands information, and so on.
    There are 2 alternatives get_light_list_of_products_by_workspace and get_names_of_products_by_workspace.
    This API can take a lot of time for workspaces with a lot of products, so it is better to use it only when the agent really needs all the details of all the products in the workspace. If the agent needs only the names of the products, it can use get_names_of_products_by_workspace, if it needs some details but not all, it can use get_light_list_of_products_by_workspace.
    It will always be possible later to get the details of a specific product using the get_product_details_by_product_name API, so the agent can start with a light call to get the list of products with few details and then get the details of the products that are interesting for it using get_product_details_by_product_name.

    Input
    sWorkspaceId: is the unique id of the workspace for which we want to get the list of products.

    Output
    the call returns null in case of errors or an array JSON object with the following properties for each product:

    bbox: optional property with the bounding box of the product, in case the product is an EO product with georeferenced data
    name: is the file name without extension
    description: optional property with the description of the product, if it is provided by the user when the product is created or edited
    fileName: is the file name with extension
    productFriendlyName: is a name that the user can assign to this product
    metadataFileCreated: metadata are not read by defualt. If the user wants to access it, a file is created in the dedicated wasdi folder. This property is a boolean true if the metadata has  been generated, false otherwise.
    metadataFileReference: if the metadataFileCreated is true, this property contains the path to the metadata file that has been generated. The path is relative to the metadata wasdi folder on the server
    metadata: real metadata if the metadataFileCreated is true. This property is not provided by default because it can be very heavy, especially for products with a lot of metadata, so it is better to read the metadata only when it is needed, using the metadataFileReference property to access the metadata file.
    bandsGroups: optional property with the bands groups of the product, if it is an EO product with multiple bands
    style: optional property with the name of the style of the product. Styles are Geoserver styles the user can upload in wasdi. If the product has a style assigned, it means that the user has uploaded a style in wasdi and assigned it to this product, so the style property contains the name of the style that is assigned to the product. The agent can use this information to suggest to the user to use this style when visualizing the product in wasdi, or to use this style as a reference when generating a new style for this product.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/product/byws",
            params={"workspace": sWorkspaceId},
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getListByWorkspace call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_light_list_of_products_by_workspace(sWorkspaceId: str, oContext: Context = None) -> str:
    """
    Returns the light list of products in a workspace.
    This mirrors ProductResource.getLightListByWorkspace and is useful when the agent needs to get a list of products in a workspace with only basic details, without the full one.
    The light list is faster to be returned by the server and to be processed by the agent, so it is better to use it when the agent needs only some details of the products in the workspace, but not all. If the agent needs all the details of the all the products, it can use get_detailed_list_of_products_by_workspace, if it needs only the names of the products it can use get_names_of_products_by_workspace.
    Once a product name is available, it will always be possible to get the full details of the product using the get_product_details_by_product_name API, so the agent can start with a light call to get the list of products with few details and then get the details of the products that are interesting for it using get_product_details_by_product_name.

    Input
    sWorkspaceId: is the unique id of the workspace for which we want to get the light list of products.

    Output
    the call returns an empty array in case of errors or an array JSON object with the following properties for each product:

    name: is the file name without extension
    productFriendlyName: is a name that the user can assign to this product
    bbox: optional property with the bounding box of the product, in case the product is an EO product with georeferenced data.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/product/bywslight",
            params={"workspace": sWorkspaceId},
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getLightListByWorkspace call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_names_of_products_by_workspace(sWorkspaceId: str, oContext: Context = None) -> str:
    """
    Returns the file names of the products in a workspace. 
    This mirrors ProductResource.getNamesByWorkspace. It is useful when the agent needs only the names of the products in the workspace, without any other details, for example to check if a product with a specific name exists in the workspace.
    It is the fastest API to get the list of products in a workspace, so it is better to use it when the agent needs only the names of the products. If the agent needs some details but not all, it can use get_light_list_of_products_by_workspace, if it needs all the details of the products it can use get_detailed_list_of_products_by_workspace.

    Input
    sWorkspaceId: is the unique id of the workspace for which we want to get the names of the products.
    
    Output
    the call returns an empty array in case of errors or an array of strings with the names of the products in the workspace. The names are the file names with extension, for example "image.tif", "data.csv", and so on.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/product/namesbyws",
            params={"workspace": sWorkspaceId},
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getNamesByWorkspace call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_processes_by_workspace(
    sWorkspaceId: str,
    sStatus: str = None,
    sOperationType: str = None,
    sNamePattern: str = None,
    sDateFrom: str = None,
    sDateTo: str = None,
    iStartIndex: int = None,
    iEndIndex: int = None,
    oContext: Context = None,
) -> str:
    """
    Returns a filtered list of process workspaces in a workspace. A process workspace is a process that has been executed in a workspace. Process Workspaces type tells the operation that is stored in operationType.
    Operations are started using run_processor, run_snap_workflow, mosaic, multisubset, regrid, import_product_in_wasdi, ingest_existing_file_in_workspace, share_file_to_workspace, publish_product_band_in_wms, upload_new_processor, redeploy_processor, environmentupdate, killprocesstree. The operationType is the name of the operation that has been executed in the workspace.
    This is a node-based API. ProcessWorkspaces are stored per node, not all in the main node. This tool automatically resolves the node URL from the workspace details.
    The agent can use this API to get the list of processes that have been executed in a workspace, with the possibility to filter by status, operation type, name pattern and date range. This can be useful for example to check if a process with a specific name or a specific operation type has been executed in the workspace, or to get the list of all the processes that are currently running in the workspace.
    This API can be used also to try to understand what went wrong: we can get the list of processes that are in error status and then call get_process_payload and or get_processor_logs to get more details about the error and try to understand what went wrong.
    The API is paginated, so the agent can use the start index and end index parameters to get only a subset of the results, for example to get the first 10 processes or to get the next 10 processes after the first 10.

    Inputs
    sWorkspaceId: is the unique id of the workspace for which we want to get the process workspaces.
    sStatus: is an optional filter for the status of the process workspaces.
    sOperationType: is an optional filter for the operation type of the process workspaces.
    sNamePattern: is an optional filter for the name pattern of the process workspaces.
    sDateFrom: is an optional filter for the start date of the process workspaces.
    sDateTo: is an optional filter for the end date of the process workspaces.
    iStartIndex: is an optional filter for the start index of the process workspaces.
    iEndIndex: is an optional filter for the end index of the process workspaces.

    Output
    A list process workspace; each element has the following properties:

    String productName: name of the product target of this process. The name is historical, but can represent in reality a name of a product, or of an application or of a SNAP workflow
    String operationType: type of the operation performed by this process. Types are INGEST, DOWNLOAD, SHARE, PUBLISHBAND, GRAPH, DEPLOYPROCESSOR, RUNPROCESSOR, MOSAIC, MULTISUBSET, REGRID, DELETEPROCESSOR, INFO, REDEPLOYPROCESSOR, LIBRARYUPDATE, ENVIRONMENTUPDATE, KILLPROCESSTREE,
    String operationSubType: subtype of the operation performed by this process. Each operation can have a subtype in theory. In reality now is used for DOWNALOD operations: subtype is the data provider of the data that is downloaded, for example COPERNICUS, CREODIAS2, LSA etc
	String operationDate: date of the operation creation
    String operationStartDate: start date of the operation
    String operationEndDate: end date of the operation
    String lastChangeDate: date of the last status change
	String userId: id of the user who started the operation
    String fileSize: size of the file ie for a download operation
    String status: status of the process. Status can be CREATED, RUNNING, WAITING, READY, DONE, ERROR, STOPPED. The process is created CREATED. Then is the scheduler that triggers it in start. Applications moves in WAITING when the user calls waitProcess from the lib. When the process is done, it become READY and the scheduler will move in RUNNING again when there is a slot
    int progressPerc: progress percentage of the process
    String processObjId: id of the process object
    int pid: id of the process in the operating system of the node where it is executed
    String payload: json output created by the process when id done. The content of the payload is defined by the process itself, but it can contain useful information for the user
    String workspaceId: id of the workspace where the process is executed

    This mirrors ProcessWorkspaceResource.getProcessByWorkspace.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    # Resolve the node URL for this workspace
    sNodeUrl = await getNodeUrlForWorkspace(sWorkspaceId, sSessionToken)

    aoParams = {
        "workspace": sWorkspaceId,
        "status": sStatus,
        "operationType": sOperationType,
        "namePattern": sNamePattern,
        "dateFrom": sDateFrom,
        "dateTo": sDateTo,
        "startindex": iStartIndex,
        "endindex": iEndIndex,
    }
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{sNodeUrl}/rest/process/byws",
            params=aoParams,
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getProcessByWorkspace call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_last_processes_by_workspace(sWorkspaceId: str, oContext: Context = None) -> str:
    """
    Returns the last five process workspaces for a workspace.
    This is a node-based API. This tool automatically resolves the node URL from the workspace details.
    This mirrors ProcessWorkspaceResource.getLastProcessByWorkspace.

    Input
    sWorkspaceId: is the unique id of the workspace for which we want to get the last five process workspaces.

    Output
    the call returns an empty array in case of errors or an array JSON object with the following properties for each process workspace:
    String productName: name of the product target of this process. The name is historical, but can represent in reality a name of a product, or of an application or of a SNAP workflow
    String operationType: type of the operation performed by this process. Types are INGEST, DOWNLOAD, SHARE, PUBLISHBAND, GRAPH, DEPLOYPROCESSOR, RUNPROCESSOR, MOSAIC, MULTISUBSET, REGRID, DELETEPROCESSOR, INFO, REDEPLOYPROCESSOR, LIBRARYUPDATE, ENVIRONMENTUPDATE, KILLPROCESSTREE,
    String operationSubType: subtype of the operation performed by this process. Each operation can have a subtype in theory. In reality now is used for DOWNALOD operations: subtype is the data provider of the data that is downloaded, for example COPERNICUS, CREODIAS2, LSA etc
	String operationDate: date of the operation creation
    String operationStartDate: start date of the operation
    String operationEndDate: end date of the operation
    String lastChangeDate: date of the last status change
	String userId: id of the user who started the operation
    String fileSize: size of the file ie for a download operation
    String status: status of the process. Status can be CREATED, RUNNING, WAITING, READY, DONE, ERROR, STOPPED. The process is created CREATED. Then is the scheduler that triggers it in start. Applications moves in WAITING when the user calls waitProcess from the lib. When the process is done, it become READY and the scheduler will move in RUNNING again when there is a slot
    int progressPerc: progress percentage of the process
    String processObjId: id of the process object
    int pid: id of the process in the operating system of the node where it is executed
    String payload: json output created by the process when id done. The content of the payload is defined by the process itself, but it can contain useful information for the user
    String workspaceId: id of the workspace where the process is executed
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    # Resolve the node URL for this workspace
    sNodeUrl = await getNodeUrlForWorkspace(sWorkspaceId, sSessionToken)

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{sNodeUrl}/rest/process/lastbyws",
            params={"workspace": sWorkspaceId},
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getLastProcessByWorkspace call completed with status %s", oResponse.status_code)
        return oResponse.text

@s_oMcpServer.tool()
async def get_summary_of_running_processes_for_workspace(sWorkspaceId: str = None, oContext: Context = None) -> str:
    """
    Returns process summary counts for a workspace and user. The agent can use this API to get a quick overview of the processes that are 
    running in a workspace, for example to check how many processes are currently running and how many are waiting, and so on. 
    This mirrors ProcessWorkspaceResource.getSummary.
    This is a node-based API.

    Input
    sWorkspaceId: is the unique id of the workspace for which we want to get the summary of running processes.

    Output:
	int userProcessWaiting: number of processes that are waiting for the user who is calling this API. This is the number of processes that are in WAITING status and that have been started by the user who is calling this API. The agent can use this information to check if the user has any processes that are waiting to be executed, for example to suggest to the user to start a new process or to check if there are any processes that are waiting for the user to take action.
	int userProcessRunning: number of processes that are currently running for the user who is calling this API. This is the number of processes that are in RUNNING status and that have been started by the user who is calling this API. The agent can use this information to check if the user has any processes that are currently running, for example to suggest to the user to wait for the processes to complete before starting a new one.	
	int allProcessWaiting: number of processes that are waiting in the workspace. This is the number of processes that are in WAITING status, regardless of the user who started them. The agent can use this information to check if there are any processes that are waiting to be executed in the workspace.
	int allProcessRunning: number of processes that are currently running in the workspace. This is the number of processes that are in RUNNING status, regardless of the user who started them. The agent can use this information to check if there are any processes that are currently running in the workspace.    

    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    aoParams = {}
    if sWorkspaceId:
        aoParams["workspace"] = sWorkspaceId

    # Resolve the node URL for this workspace
    sNodeUrl = await getNodeUrlForWorkspace(sWorkspaceId, sSessionToken)

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{sNodeUrl}/rest/process/summary",
            params=aoParams,
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getSummary call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def kill_process_in_workspace(
    sProcessObjId: str,
    bKillTheEntireTree: bool = None,
    sWorkspaceId: str = None,
    oContext: Context = None,
) -> str:
    """
    Kills a running process workspace. Each process, when is the first, can trigger new sub-processes. 
    This API can be used to kill a process and all its sub-processes, or only the process itself. The agent can use this API to stop a process that is running in a workspace, 
    for example if the user wants to cancel the execution of a process or if the process is taking too long to complete.
    This mirrors ProcessWorkspaceResource.deleteProcess.
    This is a node-based API.

    Inputs
    sProcessObjId: is the unique id of the process workspace
    bKillTheEntireTree: is an optional parameter that indicates if the entire tree of processes should be killed. If true, the process and all its sub-processes will be killed.
    sWorkspaceId: is the unique id of the workspace in which the process is running

    Output:
    simply Http response. 200 for ok. Note the API is asynch; the kill process will take more time and is a process itself of type KILLPROCESSTREE 
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessObjId:
        raise ValueError("Missing process id")

    aoParams = {"procws": sProcessObjId}
    if bKillTheEntireTree is not None:
        aoParams["treeKill"] = "true" if bKillTheEntireTree else "false"

    sNodeUrl = await getNodeUrlForProcessWorkspace(sProcessObjId, sSessionToken, sWorkspaceId)

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{sNodeUrl}/rest/process/delete",
            params=aoParams,
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI deleteProcess call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_process_by_id(sProcessObjId: str, sWorkspaceId: str = None, oContext: Context = None) -> str:
    """
    Returns a process workspace view model by id.
    This mirrors ProcessWorkspaceResource.getProcessById. This is a node-based API.
    Can be used to get the details of a process workspace.

    Inputs:
    sProcessObjId: is the unique id of the process workspace
    sWorkspaceId: is the unique id of the workspace in which the process is running

    Output:
    return an empty Process Workspace View Model in case of errors or a JSON object with the following properties:
    String productName: name of the product target of this process. The name is historical, but can represent in reality a name of a product, or of an application or of a SNAP workflow
    String operationType: type of the operation performed by this process. Types are INGEST, DOWNLOAD, SHARE, PUBLISHBAND, GRAPH, DEPLOYPROCESSOR, RUNPROCESSOR, MOSAIC, MULTISUBSET, REGRID, DELETEPROCESSOR, INFO, REDEPLOYPROCESSOR, LIBRARYUPDATE, ENVIRONMENTUPDATE, KILLPROCESSTREE,
    String operationSubType: subtype of the operation performed by this process. Each operation can have a subtype in theory. In reality now is used for DOWNALOD operations: subtype is the data provider of the data that is downloaded, for example COPERNICUS, CREODIAS2, LSA etc
	String operationDate: date of the operation creation
    String operationStartDate: start date of the operation
    String operationEndDate: end date of the operation
    String lastChangeDate: date of the last status change
	String userId: id of the user who started the operation
    String fileSize: size of the file ie for a download operation
    String status: status of the process. Status can be CREATED, RUNNING, WAITING, READY, DONE, ERROR, STOPPED. The process is created CREATED. Then is the scheduler that triggers it in start. Applications moves in WAITING when the user calls waitProcess from the lib. When the process is done, it become READY and the scheduler will move in RUNNING again when there is a slot
    int progressPerc: progress percentage of the process
    String processObjId: id of the process object
    int pid: id of the process in the operating system of the node where it is executed
    String payload: json output created by the process when id done. The content of the payload is defined by the process itself, but it can contain useful information for the user
    String workspaceId: id of the workspace where the process is executed

    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessObjId:
        raise ValueError("Missing process id")

    sNodeUrl = await getNodeUrlForProcessWorkspace(sProcessObjId, sSessionToken, sWorkspaceId)

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{sNodeUrl}/rest/process/byid",
            params={"procws": sProcessObjId},
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getProcessById call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_status_processes_by_id(asProcessesWorkspaceId: list[str], oContext: Context = None) -> str:
    """
    Returns the status of multiple process workspaces in a single call. Is faster than calling get_process_status_by_id for each process workspace id. 
    The agent can use this API to get the status of multiple processes in a single call.

    This mirrors ProcessWorkspaceResource.getStatusProcessesById. This is a node-based API.

    Inputs:
    asProcessesWorkspaceId: is a list of unique ids (strings) of the process workspaces

    Output:
    the call returns an empty array in case of errors or an array of Strings: one for each asProcessesWorkspaceId in input, with a value describing the status of the process workspace. Status can be CREATED, RUNNING, WAITING, READY, DONE, ERROR, STOPPED. The process is created CREATED. Then is the scheduler that triggers it in start. Applications moves in WAITING when the user calls waitProcess from the lib. When the process is done, it become READY and the scheduler will move in RUNNING again when there is a slot.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not asProcessesWorkspaceId:
        raise ValueError("Missing process id list")
    
    sNodeUrl = await getNodeUrlForProcessWorkspace(asProcessesWorkspaceId[0], sSessionToken, None)    

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.post(
            f"{sNodeUrl}/rest/process/statusbyid",
            json=asProcessesWorkspaceId,
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getStatusProcessesById call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_process_status_by_id(sProcessObjId: str, sWorkspaceId: str = None, oContext: Context = None) -> str:
    """
    Returns the status of a single process workspace. Is faster than calling get_process_by_id because it returns only the status of the process workspace, without all the other details. 
    The agent can use this API to get the status of a process workspace, for example to check if a process is still running or if it has completed.
    This mirrors ProcessWorkspaceResource.getProcessStatusById. This is a node-based API.

    Inputs:
    sProcessObjId: is the unique id of the process workspace
    sWorkspaceId: is the unique id of the workspace in which the process is running

    Output:
    a string with the status of the process workspace. Status can be CREATED, RUNNING, WAITING, READY, DONE, ERROR, STOPPED. The process is created CREATED. Then is the scheduler that triggers it in start. Applications moves in WAITING when the user calls waitProcess from the lib. When the process is done, it become READY and the scheduler will move in RUNNING again when there is a slot.
    In case of error in the API it return in any case the string ERROR. The agent can use this information to check if the process is still running or if it has completed, and take appropriate action based on the status of the process workspace.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessObjId:
        raise ValueError("Missing process id")

    sNodeUrl = await getNodeUrlForProcessWorkspace(sProcessObjId, sSessionToken, sWorkspaceId)

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{sNodeUrl}/rest/process/getstatusbyid",
            params={"procws": sProcessObjId},
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getProcessStatusById call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def update_process_by_id(
    sProcessObjId: str,
    sNewStatus: str,
    iPerc: int,
    sSendToRabbit: str = None,
    sWorkspaceId: str = None,
    oContext: Context = None,
) -> str:
    """
    Updates a process workspace status and progress. This is a node-based API. 
    It is more fast than updating the full process workspace.
    This mirrors ProcessWorkspaceResource.updateProcessById.

    Inputs:
    sProcessObjId: is the unique id of the process workspace
    sNewStatus: is the new status of the process workspace. Must be one of the following values: CREATED, RUNNING, WAITING, READY, DONE, ERROR, STOPPED. 
    iPerc: is the progress percentage of the process workspace. To not update perc, just pass a number <0 or > 100
    sSendToRabbit: is an optional parameter to send the update to RabbitMQ that will notify the client
    sWorkspaceId: is the unique id of the workspace in which the process is running

    Output:
    The updated Process Workspace View Model in JSON format. The properties are the same as in get_process_by_id.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessObjId:
        raise ValueError("Missing process id")

    if not sNewStatus:
        raise ValueError("Missing process status")

    aoParams = {
        "procws": sProcessObjId,
        "status": sNewStatus,
        "perc": iPerc,
    }
    if sSendToRabbit is not None:
        aoParams["sendrabbit"] = sSendToRabbit

    sNodeUrl = await getNodeUrlForProcessWorkspace(sProcessObjId, sSessionToken, sWorkspaceId)

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{sNodeUrl}/rest/process/updatebyid",
            params=aoParams,
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI updateProcessById call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def set_process_payload(sProcessObjId: str, sPayload: str, sWorkspaceId: str = None, oContext: Context = None) -> str:
    """
    Sets the payload of a process workspace. This is a node-based API. Usually, only processes itself update the payload, so the agent should not use this API.
    This mirrors ProcessWorkspaceResource.setProcessPayloadPOST.

    Inputs:
    sProcessObjId: is the unique id of the process workspace
    sPayload: is the new payload of the process workspace. The payload usually is a JSON string but can be any string. The content of the payload is defined by the process itself, but it can contain useful information for the user.
    sWorkspaceId: is the unique id of the workspace in which the process is running

    Output:
    The updated Process Workspace View Model in JSON format. The properties are the same as in get_process_by_id.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessObjId:
        raise ValueError("Missing process id")

    sNodeUrl = await getNodeUrlForProcessWorkspace(sProcessObjId, sSessionToken, sWorkspaceId)

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.post(
            f"{sNodeUrl}/rest/process/setpayload",
            params={"procws": sProcessObjId},
            content=sPayload,
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI setProcessPayloadPOST call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_process_payload(sProcessObjId: str, sWorkspaceId: str = None, oContext: Context = None) -> str:
    """
    Returns the payload of a process workspace. This is a node-based API. The payload is a text ouput of the process, usually a JSON. 
    This mirrors ProcessWorkspaceResource.getPayload.

    Inputs:
    sProcessObjId: is the unique id of the process workspace
    sWorkspaceId: is the unique id of the workspace in which the process is running

    Outputs:
    A string with the payload of the process workspace. The payload usually is a JSON string but can be any string. The content of the payload is defined by the process itself, but it can contain useful information for the user. 
    Null in case of errors. The agent can use this information to get the output of a process workspace, for example to check the results of a process or to get the details of an error that occurred during the execution of a process.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessObjId:
        raise ValueError("Missing process id")

    sNodeUrl = await getNodeUrlForProcessWorkspace(sProcessObjId, sSessionToken, sWorkspaceId)

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{sNodeUrl}/rest/process/payload",
            params={"procws": sProcessObjId},
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getPayload call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def upload_new_processor(
    sFilePath: str,
    sWorkspaceId: str,
    sName: str,
    sVersion: str = None,
    sDescription: str = None,
    sType: str = None,
    sParamsSample: str = None,
    iPublic: int = None,
    iTimeout: int = None,
    bForce: bool = False,
    oContext: Context = None,
) -> str:
    """
    Create a new processor in WASDI. The processor is uploaded as a zip file. In python, the zip must contain myProcessor.py and can contain pip.txt and or packages.txt. Never upload the config.json or the params.json files.
    pip.txt contains a line for each python package to install (wasdi will do it). Eventually packages.txt contains a line for each system package to install (wasdi will do it). 
    myProcessor.py must have a structure like this:
        import wasdi

        def run():
            wasdi.wasdiLog("Here I can start to code")


        if __name__ == '__main__':
            wasdi.init("./config.json")
            run()    

    For the rest, can be as any other python application, with also other py files and using all the lib needed. 
    The inputs are a JSON string. From the code the user can use wasdi.getParameter("PARAM_NAME", "DEFAULT") to get the value of the parameter. 
    Usually applications reads inputs, search and import images in the workspace, create new files that are added to the workspace and save a payload that is another json.
    Each application can call whatever other application: asynch or waiting for it using wasdi.waitProcess("ID").
    The code should only avoid to use absolute path and, instead, call always wasdi.getPath("FILE.EXT") to get the path: this will work both on local PC and on the cloud.

    The agent can use this API to upload a new processor to WASDI, to add a new processing algorithm.
    The user can ask to create a wasdi app: the agent needs to code the processor, create the zip and call this API to upload it.
    This mirrors ProcessorsResource.uploadProcessor.
    The file is read from the local filesystem and sent as multipart/form-data.

    Inputs:
    sFilePath: is the local path to the zip file containing the processor code and dependencies.
    sWorkspaceId: is the unique id of a workspace: in this case is used only to send notifications to the client. But a workspace is needed to upload. Just open one of the user or create one if none is available.
    sName: is the name of the processor. The name must be unique in WASDI. If a processor with the same name already exists, the upload will fail.
    sVersion: obsolete. Set "1".
    sDescription: is an optional description of the processor. 
    sType: type of the processor. There are different processor types in WASDI. More will come in future. Usually you will use one of these, with the first higher priority PYTHON312_UBUNTU24, PIP_ONESHOT, PYTHON_PIP_2, PYTHON_PIP_2_UBUNTU_20. Other exists use the docs for other types
    sParamsSample: is an optional JSON string that contains a sample of the parameters that the processor expects.
    iPublic: is an optional integer that indicates if the processor should be public (1) or private (0). If not specified, the default is private (0).
    iTimeout: is an optional integer that indicates the timeout in seconds for the processor. If not specified, the default is 3600 seconds (1 hour). Use carefully -1 for no timeout. The agent can use this information to set a timeout for the processor, for example to prevent long-running processes from consuming too many resources.
    bForce: please also use FALSE for this as an agent

    Output:

    JSON object with:

    IntValue: http code equivalent of the result. 200 means ok.
    StringValue: in case of error, a related message. In case of success, the id of the process workspace that will deploy the application in WASDI. Can be used to get its status and verify when the deploy is finished.
    DoubleValue: ignored in this API
    BoolValue: True if the the upload is a success, False otherwise. In case of success, the id of the process workspace that will deploy the application in WASDI. Can be used to get its status and verify when the deploy is finished.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sFilePath:
        raise ValueError("Missing processor zip file path")

    if not os.path.isfile(sFilePath):
        raise ValueError(f"Processor zip file not found: {sFilePath}")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    if not sName:
        raise ValueError("Missing processor name")

    aoParams = {
        "workspace": sWorkspaceId,
        "name": sName,
        "version": sVersion,
        "description": sDescription,
        "type": sType,
        "paramsSample": sParamsSample,
        "public": iPublic,
        "timeout": iTimeout,
        "force": bForce,
    }
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    with open(sFilePath, "rb") as oFile:
        aoFiles = {"file": (os.path.basename(sFilePath), oFile, "application/zip")}

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.post(
                f"{s_sWasdiApiUrl}/rest/processors/uploadprocessor",
                params=aoParams,
                files=aoFiles,
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI uploadProcessor call completed with status %s", oResponse.status_code)
            return oResponse.text


@s_oMcpServer.tool()
async def get_deployed_processors(oContext: Context = None) -> str:
    """
    Returns all deployed processors visible (owned, public, shared with) to the authenticated user.
    This mirrors ProcessorsResource.getDeployedProcessors.
    The agent can use it to get the list of processors that are available to the user, for example to display them in a UI or to select one for execution or answer to a question that is searching for some processor. 
    
    Output:
    A list of DeployedProcessorViewModel JSON objects, each with the following properties (empty may be no processors or an error):

 	String processorId: unique id of the processor
	String processorName: unique name of the processor
	String processorVersion: version of the processor. Starts from 1. Is incremented by wasdi at every update or redeploy
	String processorDescription: description of the processor
	String imgLink: link to the image associated to the processor, as a kind of icon
	String logo: link to the logo associated to the processor
	String publisher: publisher of the processor
	String publisherNickName: nickname of the publisher
	String paramsSample: sample JSON parameters for the processor: these are very important to start it. 
	isPublic: indicates if the processor is public (1) or not (0)
	minuteTimeout: timeout in minutes for the processor
	type: type of the processor (see types in the upload_new_processor API)
	sharedWithMe: indicates if the processor is shared with the user
	readOnly: indicates if the processor is read-only
	isDeploymentOngoing: indicates if the deployment is ongoing
	lastUpdate: timestamp of the last update

    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/processors/getdeployed",
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getDeployedProcessors call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_single_deployed_processor(
    sProcessorId: str = None,
    sProcessorName: str = None,
    oContext: Context = None,
) -> str:
    """
    Returns a DeployedProcessorViewModel for a specific processor by id or name. The agent can use it to get details of a single processor. For example the user can ask information about a processor using the name.
    This mirrors ProcessorsResource.getSingleDeployedProcessor.

    Inputs:
    sProcessorId: is the unique id of the processor. If provided, it will be used to retrieve the processor. If not provided, sProcessorName must be provided.
    sProcessorName: is the name of the processor. If sProcessorId is not provided, this will be used to retrieve the processor.

    Output:
    A DeployedProcessorViewModel JSON object with the following properties: if the view model is empty there was an error.

 	String processorId: unique id of the processor
	String processorName: unique name of the processor
	String processorVersion: version of the processor. Starts from 1. Is incremented by wasdi at every update or redeploy
	String processorDescription: description of the processor
	String imgLink: link to the image associated to the processor, as a kind of icon
	String logo: link to the logo associated to the processor
	String publisher: publisher of the processor
	String publisherNickName: nickname of the publisher
	String paramsSample: sample JSON parameters for the processor: these are very important to start it. 
	isPublic: indicates if the processor is public (1) or not (0)
	minuteTimeout: timeout in minutes for the processor
	type: type of the processor (see types in the upload_new_processor API)
	sharedWithMe: indicates if the processor is shared with the user
	readOnly: indicates if the processor is read-only
	isDeploymentOngoing: indicates if the deployment is ongoing
	lastUpdate: timestamp of the last update    

    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessorId and not sProcessorName:
        raise ValueError("Missing processor id or processor name")

    aoParams = {
        "processorId": sProcessorId,
        "name": sProcessorName,
    }
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/processors/getprocessor",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getSingleDeployedProcessor call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_market_place_app_list(oFilters: dict = None, oContext: Context = None) -> str:
    """
    Returns the marketplace processor list filtered by the given criteria. Not all the processors goes in the marketplace. The user can run processors directly by code or wasdi client using the json. Or instead can go in the marketplace where it will see a list of processors that has been added to the market.
    The main difference is that a processor in the marketplace has also an UI. UI in WASDI is another optional json file, that become mandatory to show in the marketplace. The UI json maps each processor input parameter to a standard user control in the wasdi interface.
    There are controls for input text, bounding boxes, date, select a product in the workspace, integers etc.
    The ui json tells to wasdi the type of each parameter. Wasdi render an automatic ui. The user fills it. Wasdi re-created the parameters json and starts the application.
    This API return all the applications available in the marketplace so with a user interface associated.
    The agent can use this to list search and explore these applications that have an UI.
    The call is paginated.  This mirrors ProcessorsResource.getMarketPlaceAppList.
    Note that this API return the friendlyName of the processor. This can be used to show a more user-friendly name in the UI, instead of the processorName that is unique but not always user-friendly. 
    The friendlyName is set by the publisher of the processor when it is uploaded to the marketplace. The user may refer to some applications using the friendly name instead of the real name, take care!
    
    Inputs:
    oFilters: is a dictionary that can contain the following keys:
	
	List<String> categories = Or filter on Categories
	List<String> publishers = Or filter on Publishers
	String name: filter on Name (contains)	
	Integer itemsPerPage = 12 default. This is the number of items per page.
	Integer page = 0 default. This is the page number to retrieve. The first page is 0.
	String orderBy: filter on Order By. Can be "name", "publisher", "lastUpdate". Default is "lastUpdate".
	int orderDirection: Sorting Direction (1 = ascending, -1 = descending)

    Output:
    A list of AppListViewModel JSON objects, each with the following properties:
	String processorId: unique id of the processor
	String processorName: unique name of the processor
	String processorDescription: description of the processor
	String imgLink: link to the image associated to the processor, as a kind of icon
	String publisher: publisher of the processor
	String publisherNickName: nickname of the publisher
	Float score: score of the processor in the marketplace. The score is a float value between 0 and 5, representing the average rating given by users who have used the processor. A higher score indicates better user satisfaction and performance.
	Integer votes: number of votes received by the processor in the marketplace. This is an integer value representing the total count of users who have rated the processor. A higher number of votes indicates greater user engagement and feedback.
	String friendlyName: friendly name of the processor
	Float price: price of the processor
	Float squareKilometerPrice: price per square kilometer
	boolean isMine: indicates if the processor belongs to the current user
	boolean buyed: indicates if the processor has been purchased
	String logo: link to the logo of the processor
	boolean readOnly: indicates if the processor is read-only
    
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.post(
            f"{s_sWasdiApiUrl}/rest/processors/getmarketlist",
            json=oFilters or {},
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getMarketPlaceAppList call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_market_place_app_detail(sProcessorName: str, oContext: Context = None) -> str:
    """
    Returns the detailed marketplace information for a processor. See the docs of get_market_place_app_list to get a broader overview.
    This mirrors ProcessorsResource.getMarketPlaceAppDetail.

    Inputs:
    sProcessorName: is the unique name of the processor.

    Output:
    AppListViewModel JSON object, with the following properties:
	String processorId: unique id of the processor
	String processorName: unique name of the processor
	String processorDescription: description of the processor
	String imgLink: link to the image associated to the processor, as a kind of icon
	String publisher: publisher of the processor
	String publisherNickName: nickname of the publisher
	Float score: score of the processor in the marketplace. The score is a float value between 0 and 5, representing the average rating given by users who have used the processor. A higher score indicates better user satisfaction and performance.
	Integer votes: number of votes received by the processor in the marketplace. This is an integer value representing the total count of users who have rated the processor. A higher number of votes indicates greater user engagement and feedback.
	String friendlyName: friendly name of the processor
	Float price: price of the processor
	Float squareKilometerPrice: price per square kilometer
	boolean isMine: indicates if the processor belongs to the current user
	boolean buyed: indicates if the processor has been purchased
	String logo: link to the logo of the processor
	boolean readOnly: indicates if the processor is read-only
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessorName:
        raise ValueError("Missing processor name")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/processors/getmarketdetail",
            params={"processorname": sProcessorName},
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getMarketPlaceAppDetail call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def run_processor(
    sName: str,
    sWorkspaceId: str,
    sEncodedJson: str,
    sParentProcessWorkspaceId: str = None,
    bNotify: bool = None,
    oContext: Context = None,
) -> str:
    """
    Runs a processor. This is a very important API.  
    The agent can use this API to run a processor in WASDI, to start a processing algorithm. The user can ask to run a processor using the name and the parameters. 
    The user can also ask the agent to run the processor but asking the Assistant to create the parameters. This is the important step: how to create the JSON parameters? There are different ways.
    Each processor should have a sample parameters JSON. The agent can get it using the get_single_deployed_processor API and then modify it to create the parameters for the run. The agent can also use the get_market_place_app_detail API to get the sample parameters for a processor in the marketplace.
    Often the processors have also an help where the parameters are explained. The agent can use the get_processor_help API to get the help for a processor and then create the parameters for the run.
    So maybe the user can ask generate for me a false color image of the area of this place. We need to find an application that does it. There may be some knowledge in the documentation or we can get the list and 
    search for similar names. Then get the helps and verify what they declare to do. If we find a processor, we can then understand the params from the sample or from the help and compose the json input file that must be passed to the application.
    PLEASE DO NOT ALLOW too big bounding boxes. Assume a max of 2 square degrees. If the user asks for a bigger area, please ask to split it in smaller areas and run the processor multiple times. This is important to avoid running a processor that will consume too many resources. 
    The agent can also use the get_process_status_by_id API to check the status of a process workspace after starting it, to see if it is still running or if it has completed.
    The can get the logs of a process workspace using the get_processor_logs API, to see the output of the processor or to debug any issues that may have occurred during the execution of the processor.
    Often the guide declares also the files generated, that can be checked once the app is DONE using get_names_of_products_by_workspace. The agent can use these APIs to check the output files generated by the processor and to verify if the expected files are present in the workspace.
    All applications must be executed in a Workspace.
    This mirrors ProcessorsResource.runPost.
    

    Inputs:
    sName: is the unique name of the processor to run.
    sWorkspaceId: is the unique id of the workspace in which the processor will run. The agent can use this information to specify the workspace where the processor should be executed.    
    sEncodedJson: is the JSON string containing the parameters for the processor. The agent can use this information to provide the necessary input parameters for the processor execution. It is URL encoded.
    sParentProcessWorkspaceId: is the unique id of the parent process workspace, if any. Usually is any, because is automatically set by the system when a parent trigger a child.
    bNotify: is a boolean indicating if the user should be notified when the processor completes. If True, the user will receive a notification when the processor finishes executing. If False, no notification will be sent. If not specified, the default is False.

    Output:
    a RunningProcessorViewModel:
    String processorId = the unique id of the processor
    String name = the unique name of the processor
    String processingIdentifier = the unique id of the process workspace that is running the processor
    String status = the status of the process workspace. Can be CREATED, RUNNING, WAITING, READY, DONE, ERROR, STOPPED. Should be CREATED at creation time.
    String jsonEncodedResult = the JSON encoded result of the processor execution. Empyt at creation time.
    String message = any message associated with the processor execution

    In case of errors, the state will be ERROR.
    """

        
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sName:
        raise ValueError("Missing processor name")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    if sEncodedJson is None:
        raise ValueError("Missing encoded json payload")

    aoParams = {
        "name": sName,
        "workspace": sWorkspaceId,
        "parent": sParentProcessWorkspaceId,
        "notify": bNotify,
    }
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.post(
            f"{s_sWasdiApiUrl}/rest/processors/run",
            params=aoParams,
            content=sEncodedJson,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI runPost call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_credits_for_run_paid_processor(sProcessorId: str, sEncodedJson: str, oContext: Context = None) -> str:
    """
    Returns the estimated credits needed for a processor run, if the processors is a paid one based on credits.
    This mirrors ProcessorsResource.getCreditsForRun.

    Inputs:
    sProcessorId: is the unique id of the processor to run.
    sEncodedJson: is the JSON string containing the parameters for the processor.

    Output:
    a number that is the total credits needed.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessorId:
        raise ValueError("Missing processor id")

    if sEncodedJson is None:
        raise ValueError("Missing encoded json payload")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.post(
            f"{s_sWasdiApiUrl}/rest/processors/getcredits",
            params={"processorId": sProcessorId},
            content=sEncodedJson,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getCreditsForRun call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_processor_help(sProcessorName: str, oContext: Context = None) -> str:
    """
    Returns the help text for a processor. Is very important to understand what the processor does and how to use it. The help text is usually a markdown text that explains the processor, its inputs, outputs, and any other relevant information
    like the files that will be generated. 
    The agent can use this information to understand how to run the processor and what parameters to provide.
    Can be very useful to be used before running a processor to understand what it does and how to use it.
    This mirrors ProcessorsResource.help.

    Inputs:
    sProcessorName: is the unique name of the processor.

    Output:
    A string with the help text of the processor. The help text is usually a markdown text that explains the processor, its inputs, outputs, and any other relevant information like the files that will be generated.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessorName:
        raise ValueError("Missing processor name")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/processors/help",
            params={"name": sProcessorName},
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI help call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_process_log_count(sProcessWorkspaceId: str, oContext: Context = None) -> str:
    """
    Returns the count of log rows for a processor workspace. This is a node-based API.
    This mirrors ProcessorsResource.countLogs.

    Inputs:
    sProcessWorkspaceId: is the unique id of the process workspace.

    Output:
    An integer with the count of log rows for the processor.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessWorkspaceId:
        raise ValueError("Missing process workspace id")
    
    # Resolve the node URL for this workspace
    sNodeUrl = await getNodeUrlForProcessWorkspace(sProcessWorkspaceId, sSessionToken, None)

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{sNodeUrl}/rest/processors/logs/count",
            params={"processworkspace": sProcessWorkspaceId},
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI countLogs call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_processor_logs(sProcessWorkspaceId: str, iStartRow: int = None, iEndRow: int = None, oContext: Context = None) -> str:
    """
    Returns a paginated list of log rows for a processor workspace. The agent can use this information to get the logs of a processor, for example to check the output of a process or to debug any issues that may have occurred during the execution of the process.
    This mirrors ProcessorsResource.getLogs. This is a node-based API.

    Inputs:
    sProcessWorkspaceId: is the unique id of the process workspace.
    iStartRow: is the starting row for pagination.
    iEndRow: is the ending row for pagination.

    Output:
    A list of ProcessorLogViewModel

	String logDate: Log Date
	String logRow: Log Text
	int rowNumber: Row Number

    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessWorkspaceId:
        raise ValueError("Missing process workspace id")

    aoParams = {"processworkspace": sProcessWorkspaceId, "startrow": iStartRow, "endrow": iEndRow}
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    # Resolve the node URL for this workspace
    sNodeUrl = await getNodeUrlForProcessWorkspace(sProcessWorkspaceId, sSessionToken, None)

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{sNodeUrl}/rest/processors/logs/list",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getLogs call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def redeploy_processor(sProcessorId: str, sWorkspaceId: str, oContext: Context = None) -> str:
    """
    Forces a redeploy of a processor. 
    This mirrors ProcessorsResource.redeployProcessor.

    Inputs:
    sProcessorId: is the unique id of the processor to redeploy.
    sWorkspaceId: is the unique id of the workspace in which the processor will be redeployed.

    Output:
    Standard http codes. Expect 200 for all fine.

    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessorId:
        raise ValueError("Missing processor id")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/processors/redeploy",
            params={"processorId": sProcessorId, "workspace": sWorkspaceId},
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI redeployProcessor call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def update_processor(sProcessorId: str, oUpdatedProcessorVM: dict, oContext: Context = None) -> str:
    """
    Updates the processor metadata.
    This mirrors ProcessorsResource.updateProcessor.

    Inputs:
    sProcessorId: is the unique id of the processor to update.
    oUpdatedProcessorVM: is the updated DeployedProcessorViewModel, see get_deployed_processors

    Output:
    Standard http codes. Expect 200 for all fine.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessorId:
        raise ValueError("Missing processor id")

    if oUpdatedProcessorVM is None:
        raise ValueError("Missing processor update payload")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.post(
            f"{s_sWasdiApiUrl}/rest/processors/update",
            params={"processorId": sProcessorId},
            json=oUpdatedProcessorVM,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI updateProcessor call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def update_processor_details(sProcessorId: str, oUpdatedProcessorVM: dict, oContext: Context = None) -> str:
    """
    Updates processor details and payment fields. This API is used to update a bigger set of properties of the processor.
    This mirrors ProcessorsResource.updateProcessorDetails.

    Inputs:
    sProcessorId: is the unique id of the processor to update.
    oUpdatedProcessorVM: is the updated AppDetailViewModel:
	String processorId: unique id of the processor
	String processorName: unique name of the processor
	String processorDescription: description of the processor
	String imgLink: link to the image associated to the processor, as a kind of icon
	String publisher: publisher of the processor
	String publisherNickName: nickname of the publisher
	Float score: score of the processor in the marketplace. The score is a float value between 0 and 5, representing the average rating given by users who have used the processor. A higher score indicates better user satisfaction and performance.
	String friendlyName: friendly name of the processor
	String link: an optional link provided by the publisher (take care!)
	String email: an optional email provided by the publisher
	Float ondemandPrice: price of the processor for on-demand runs, if it is a paid processor
	Float squareKilometerPrice = 0f: price per square kilometer for the processor, if it is a paid processor per credit
	String areaParameterName: the name of the parameter in the processor that represents the area of interest, if applicable. Is used to estimate the credits needed in case the processor is paid for area
	Float subscriptionPrice: price of the processor for subscription runs, if it is a paid processor
	Double updateDate: timestamp of the last update
	Double publishDate: timestamp of the publish date
	ArrayList<String> categories: list of category ids associated with the processor
	ArrayList<String> images: list of image links associated with the processor
	Boolean isMine: indicates if the processor belongs to the current user
	Boolean buyed: indicates if the processor has been purchased
	String longDescription: a long description of the processor, if provided
	Boolean showInStore = false: indicates if the processor should be shown in the marketplace
	int maxImages = 6: maximum number of images allowed for the processor
	int reviewsCount = 0: number of reviews for the processor
	int purchased = 0: number of times the processor has been purchased
	int totalRuns = 0: total number of times the processor has been run
	int userRuns = 0: number of times the current user has run the processor
	ArrayList<String> categoryNames = new ArrayList<String>(): list of category names associated with the processor
	String logo: link to the logo of the processor
	boolean readOnly: indicates if the processor is read-only

    Output:
    Standard http codes. Expect 200 for all fine.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessorId:
        raise ValueError("Missing processor id")

    if oUpdatedProcessorVM is None:
        raise ValueError("Missing processor detail payload")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.post(
            f"{s_sWasdiApiUrl}/rest/processors/updatedetails",
            params={"processorId": sProcessorId},
            json=oUpdatedProcessorVM,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI updateProcessorDetails call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def update_processor_files(
    sFilePath: str,
    sProcessorId: str,
    sWorkspaceId: str,
    sInputFileName: str = None,
    oContext: Context = None,
) -> str:
    """
    Updates the processor files using a local file path. When updating a processor, this API is used. The normal flow is that the user or the agent updates one or more files, that
    upload the updated code to WASDI that will trigger a redeploy of the application.
    If the file is only one can be directly uploaded. If are more than one we need a zip.
    This mirrors ProcessorsResource.updateProcessorFiles.

    Inputs:
    sFilePath: is the local file path of the file to upload. It can be a single file or a zip file containing multiple files.
    sProcessorId: is the unique id of the processor to update.
    sWorkspaceId: we need a workspace, but only for the notification to the client. The agent can use any valid user workspaceId here.
    sInputFileName: Name of the input file
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sFilePath:
        raise ValueError("Missing processor file path")

    if not os.path.isfile(sFilePath):
        raise ValueError(f"Processor file not found: {sFilePath}")

    if not sProcessorId:
        raise ValueError("Missing processor id")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    aoParams = {"processorId": sProcessorId, "workspace": sWorkspaceId, "file": sInputFileName}
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    with open(sFilePath, "rb") as oFile:
        aoFiles = {"file": (sInputFileName or os.path.basename(sFilePath), oFile, "application/octet-stream")}

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.post(
                f"{s_sWasdiApiUrl}/rest/processors/updatefiles",
                params=aoParams,
                files=aoFiles,
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI updateProcessorFiles call completed with status %s", oResponse.status_code)
            return oResponse.text


@s_oMcpServer.tool()
async def download_processor(sProcessorId: str, oContext: Context = None) -> str:
    """
    Downloads a processor zip. The binary response is returned as base64 text.
    This mirrors ProcessorsResource.downloadProcessor.

    Inputs:
    sProcessorId: is the unique id of the processor to download.
    
    Output:
    A string containing the base64 encoded content of the processor zip file.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessorId:
        raise ValueError("Missing processor id")

    aoParams = {"token": sSessionToken, "processorId": sProcessorId}
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/processors/downloadprocessor",
            params=aoParams
        )
        oResponse.raise_for_status()
        logging.debug("WASDI downloadProcessor call completed with status %s", oResponse.status_code)
        return oResponse.content.hex()


@s_oMcpServer.tool()
async def share_processor(sProcessorId: str, sUserId: str, sRights: str = None, oContext: Context = None) -> str:
    """
    Shares a processor with a user.
    This mirrors ProcessorsResource.shareProcessor.

    Inputs:
    sProcessorId: is the unique id of the processor to share.
    sUserId: is the unique id of the user to share the processor with.
    sRights: is an optional string indicating the rights to grant to the user. It can be "read", "write". If not specified, the default is "read".

    Output:
    IntValue: ignored in this API
    StringValue: Done if all ok. If not a message of the error.
    DoubleValue: ignored in this API
    BoolValue: True if the processor has been shared successfully.

    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessorId:
        raise ValueError("Missing processor id")

    if not sUserId:
        raise ValueError("Missing user id")

    aoParams = {"processorId": sProcessorId, "userId": sUserId, "rights": sRights}
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.put(
            f"{s_sWasdiApiUrl}/rest/processors/share/add",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI shareProcessor call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_users_can_access_processor(sProcessorId: str, oContext: Context = None) -> str:
    """
    Returns the list of users who can access a processor.
    This mirrors ProcessorsResource.getEnabledUsersSharedProcessor.

    Inputs:
    sProcessorId: is the unique id of the processor.

    Output:
    A list of ProcessorSharingViewModel JSON objects, each with the following properties:    
    userId: unique id of the user
    permission: permission granted to the user for the processor. Can be "read" or "write".
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessorId:
        raise ValueError("Missing processor id")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/processors/share/byprocessor",
            params={"processorId": sProcessorId},
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI get_users_can_access_processor call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_processor_ui(sProcessorName: str, oContext: Context = None) -> str:
    """
    Returns the JSON UI definition of a processor. see the docs of get_market_place_app_list to get a broader overview. 
    The UI definition is a JSON that describes the user interface for the processor, including the input fields, types, and any other relevant information needed to render the UI for the processor.
    This mirrors ProcessorsResource.getUI.

    Inputs:
    sProcessorName: is the unique name of the processor.

    Output:
    A string with the JSON UI definition of the processor.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessorName:
        raise ValueError("Missing processor name")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/processors/ui",
            params={"name": sProcessorName},
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getUI call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_processor_build_logs(sProcessorId: str, oContext: Context = None) -> str:
    """
    Returns the build logs for a processor. After the upload or every redeploy or update files, wasdi will rebuild the docker image of the processor. This API return the log of the docker build operation.
    This is very important to help users when they experience problems deploying their application. 
    The agent can use it to get the docker build output and identify the real problem. Usually problems are due to missing or wrong python dependencies that are not correctly listed by the user in the pip.txt file.
    Some problems come from the version of numpy and gdal. Since the images are pre-done templates, almost each has a fixed version of gdal so the compatibility with numpy is contrained. wasdi cleans the pip.txt filtering 
    all the not existing packages, but also numpy and gdal that are installed by default. If you want to avoid this skip, in pip.txt you have to write not only numpy but numpy==VERSION: in this case will be kept.
    This is to make it work, otherwise we have much more problems in builidng the right gdal. It works very often but sometimes create a 
    big problem. Usually is ok to "play" with these versions, as usually these build logs suggests.
    This mirrors ProcessorsResource.getProcessorBuildLogs.

    Inputs:
    sProcessorId: is the unique id of the processor.

    Output:
    A list of strings: each is a full log of a build operation for the processor. The last is the last build log. The log is a string that contains the output of the docker build command, including any errors or warnings that occurred during the build process.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessorId:
        raise ValueError("Missing processor id")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/processors/logs/build",
            params={"processorId": sProcessorId},
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getProcessorBuildLogs call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def download_product_by_name(sFileName: str, sWorkspaceId: str, sProcessObjId: str = None, sDisposition: str = None, oContext: Context = None) -> str:
    """ 
    Downloads a file by name from a workspace. Returns binary as hex text. This is a node-based API.
    This mirrors CatalogResources.downloadEntryByName.

    Inputs:
    sFileName: is the name of the file to download (always relative to workspace path, so usually just the file name)
    sWorkspaceId: is the unique id of the workspace from which to download the file
    sProcessObjId: is an optional process workspace id. If provided, the download will be associated with this process workspace. Usually keep it null
    sDisposition: is an optional string indicating the content disposition. If not provided, the default is "attachment". Can be "inline" or "attachment".
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sFileName:
        raise ValueError("Missing file name")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    aoParams = {"filename": sFileName, "workspace": sWorkspaceId, "token": sSessionToken, "procws": sProcessObjId, "disposition": sDisposition}
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    sNodeUrl = await getNodeUrlForWorkspace(sWorkspaceId, sSessionToken)

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{sNodeUrl}/rest/catalog/downloadbyname",
            params=aoParams
        )
        oResponse.raise_for_status()
        logging.debug("WASDI download_product_by_name call completed with status %s", oResponse.status_code)
        return oResponse.content.hex()


@s_oMcpServer.tool()
async def check_file_exists_in_node(sFileName: str, sWorkspaceId: str, oContext: Context = None) -> str:
    """
    Checks if a file exists on the current node. This is a node-based API. Since add file to workspace does not check the real existence of the file, this API can be used to check if a file is really present on the node before trying to download it or use it in a processor. 
    The file can aslo be not listed in the database as a product of the workspace.
    This mirrors CatalogResources.checkFileByNode.

    Inputs:
    sFileName: is the name of the file to check (always relative to workspace path, so usually just the file name)
    sWorkspaceId: is the unique id of the workspace in which to check the file

    Output:
    a boolean value.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sFileName:
        raise ValueError("Missing file name")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    sNodeUrl = await getNodeUrlForWorkspace(sWorkspaceId, sSessionToken)

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{sNodeUrl}/rest/catalog/fileOnNode",
            params={"token": sSessionToken, "filename": sFileName, "workspace": sWorkspaceId},
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI checkFileByNode call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def check_download_product_availability_by_name(sFileName: str, sWorkspaceId: str, sProcessObjId: str = None, sVolumePath: str = None, oContext: Context = None) -> str:
    """
    Checks if a file is available for download. This is a node-based API. The difference with check_file_exists_in_node 
    is that this API checks if the file is available for download, which means that it checks if the file is present on the node and is declared as a product in the workspace
    This mirrors CatalogResources.checkDownloadEntryAvailabilityByName.

    Inputs:
    sFileName: is the name of the file to check (always relative to workspace path, so usually just the file name)
    sWorkspaceId: is the unique id of the workspace in which to check the file
    sProcessObjId: is an optional process workspace id. If provided, the check will be associated with this process workspace. Usually keep it null
    sVolumePath: is an optional string indicating the volume path. If not provided, the default is null. This is used to check if the file is available in a specific volume path, if the node has multiple volume paths.

    Output:
    a Json with values:
    StringValue: ignored in this API
    IntValue: ignored in this API
    BoolValue: true if the file is available, false if the file is not
    DoubleValue: ignored in this API

    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sFileName:
        raise ValueError("Missing file name")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    aoParams = {"token": sSessionToken, "filename": sFileName, "workspace": sWorkspaceId, "procws": sProcessObjId, "volumepath": sVolumePath}
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    sNodeUrl = await getNodeUrlForWorkspace(sWorkspaceId, sSessionToken)

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{sNodeUrl}/rest/catalog/checkdownloadavaialibitybyname",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI checkDownloadEntryAvailabilityByName call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def ingest_existing_file_in_workspace(sFileName: str, sWorkspaceId: str, sParentProcessWorkspaceId: str = None, sStyle: str = None, sPlatform: str = None, oContext: Context = None) -> str:
    """
    Ingests a file already existing in a workspace. This is a node-based API. Files can exist in the workspace even if are not listed in the database. Can be an app that creates a temp file
    or a manual upload from a user for example. This API allows to verify that the file is there and add it to the db also.
    This is a node-based API. 
    This mirrors CatalogResources.ingestFileInWorkspace.

    Inputs:
    sFileName: is the name of the file to ingest (always relative to workspace path, so usually just the file name)
    sWorkspaceId: is the unique id of the workspace in which to ingest the file
    sParentProcessWorkspaceId: is an optional process workspace id. If provided, the ingest will be associated with this process workspace. Usually keep it null, is used by the library when triggers a child process
    sStyle: is an optional string indicating the name of the style of the file. If not provided, the default is null. This is used to specify the style of the file that can eventually be used to publish in WMS
    sPlatform: is an optional string indicating the platform of the file. If not provided, the default is null. 

    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sFileName:
        raise ValueError("Missing file name")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    aoParams = {"file": sFileName, "workspace": sWorkspaceId, "parent": sParentProcessWorkspaceId, "style": sStyle, "platform": sPlatform}
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    sNodeUrl = await getNodeUrlForWorkspace(sWorkspaceId, sSessionToken)

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{sNodeUrl}/rest/catalog/upload/ingestinws",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI ingest_existing_file_in_workspace call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_product_properties(sFileName: str, sWorkspaceId: str, bGetChecksum: bool = None, oContext: Context = None) -> str:
    """
    Returns the properties of a product/file in a workspace. 
    This mirrors CatalogResources.getProductProperties.

    Inputs:
    sFileName: is the name of the file to get properties for (always relative to workspace path, so usually just the file name)
    sWorkspaceId: is the unique id of the workspace in which to get the file properties
    bGetChecksum: is an optional boolean indicating whether to get the checksum of the file. If not provided, the default is false. If true, the checksum will be included in the properties. Usually use false, the checksum is needed just to check if a file is corrupted usually

    Output:
    A JSON object with the properties of the product/file, including:

	
	String fileName: Name of the file (with extension)
	String friendlyName: Friendly name of the file
	long lastUpdateTimestampMs: Last update timestamp in milliseconds
	long size: Size of the file in bytes
	String checksum: Checksum of the file (if bGetChecksum is true)
	String style: Style of the file (if applicable)
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sFileName:
        raise ValueError("Missing file name")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    aoParams = {"file": sFileName, "workspace": sWorkspaceId, "getchecksum": bGetChecksum}
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/catalog/properties",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getProductProperties call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def eo_data_search_get_count(sQuery: str, sProviders: str = None, oContext: Context = None) -> str:
    """
    Returns the total number of EO search results for a query.
    This mirrors OpenSearchResource.count.

    Inputs:
    sQuery: is the search query string
    sProviders: is an optional comma-separated list of providers to filter the search. use "AUTO" to automatically select providers.

    Output:
    The total count of search results
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sQuery:
        raise ValueError("Missing query")

    aoParams = {"query": sQuery, "providers": sProviders}
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/search/query/count",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI count call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def eo_data_paginated_search(
    sQuery: str,
    sProvider: str = None,
    sOffset: str = None,
    sLimit: str = None,
    sSortedBy: str = None,
    sOrder: str = None,
    oContext: Context = None,
) -> str:
    """
    Executes a paginated EO search query.
    This mirrors OpenSearchResource.search.

    Inputs:
    sQuery: is the search query string
    sProvider: is an optional comma-separated list of providers to filter the search. use "AUTO" to automatically select providers.
    sOffset: is an optional string indicating the offset for pagination. If not provided, the default is 0.
    sLimit: is an optional string indicating the limit for pagination. If not provided, the default is 10.
    sSortedBy: is an optional string indicating the field to sort by. If not provided, the default is "startDate".
    sOrder: is an optional string indicating the order of sorting. Can be "asc" or "desc". If not provided, the default is "desc".

    Output:
    A list of JSON objects (QueryResultViewModel) containing the search results:
	
	String preview:  Encoded Image Preview
	String title: File Name
	String summary: Description. Supports a sort of std like: "Date: 2021-12-25T18:25:03.242Z, Instrument: SAR, Mode: IW, Satellite: S1A, Size: 0.95 GB" but is not mandatory
	String id: Provider Id
	String link: Link (or equivalent) to access the file
	String footprint: WKT Footprint
	provider: Data Provider that found this item
	Map<String, String> properties: Dictionary of additional properties	
	String volumeName: If this is accessible in a Volume, here we have the nameIf this is accessible in a Volume, here we have the name
	String volumePath: If this is accessible in a Volume, here we have the path in the volume
	String platform: Unique code of the platform/mission of this entry 
	
    Basic info are:
        .Title -> Name of the file
        .Summary -> Description. Supports a sort of std like: "Date: 2021-12-25T18:25:03.242Z, Instrument: SAR, Mode: IW, Satellite: S1A, Size: 0.95 GB" but is not mandatory
        .Id -> Provider unique id
        .Link -> Link to download the file
        .Footprint -> Bounding box in WKT ie POLYGON ((-7.087445 31.109682, -4.389633 31.524973, -4.062707 29.77639, -6.712266 29.357685, -7.087445 31.109682))
                    Note: for POLYGON the convention is LON LAT, LON LAT...
        .Provider -> Provider used to get this info.

    Properties is a dictionary filled with all the properties supported by the data provider.
    Can be seen with the "info" button in the client.
            Some Commonly used, and shown in the client, are:
                ."date": reference Date
                ."instrument": used instrument 
                ."sensoroperationalmode": sensing mode
                ."size": image size as string
                ."relativeOrbit": relative orbit of the acquisition
                ."relativeorbitnumber": same of above, used by the client
                ."platformname": Platform Name

    The libs searchs for a property called relativeOrbit
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sQuery:
        raise ValueError("Missing query")

    aoParams = {
        "providers": sProvider,
        "query": sQuery,
        "offset": sOffset,
        "limit": sLimit,
        "sortedby": sSortedBy,
        "order": sOrder,
    }
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/search/query",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI search call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_data_providers(oContext: Context = None) -> str:
    """
    Returns the list of available EO data providers.
    This mirrors OpenSearchResource.getDataProviders.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/search/providers",
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getDataProviders call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def eo_data_search_count_list(asQueries: list[str], sProviders: str = None, oContext: Context = None) -> str:
    """
    Returns the total count of EO results for a list of queries.
    This mirrors OpenSearchResource.countList.

    Inputs:
    asQueries: is a list of search query strings. Usually only one is used
    sProviders: is an optional comma-separated list of providers to filter the search. use "AUTO" to automatically select providers.

    Output:
    The total count of EO results for the provided queries.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not asQueries:
        raise ValueError("Missing query list")

    aoParams = {"providers": sProviders}
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.post(
            f"{s_sWasdiApiUrl}/rest/search/query/countlist",
            params=aoParams,
            json=asQueries,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI countList call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def eo_data_search_list(asQueries: list[str], sProvider: str = None, oContext: Context = None) -> str:
    """
    Executes EO searches for a list of queries.
    This mirrors OpenSearchResource.searchList.

    Inputs:
    asQueries: is a list of search query strings. Usually only one is used
    sProviders: is an optional comma-separated list of providers to filter the search. use "AUTO" to automatically select providers.

    Output:
    A list of JSON objects (QueryResultViewModel) containing the search results:
	
	String preview:  Encoded Image Preview
	String title: File Name
	String summary: Description. Supports a sort of std like: "Date: 2021-12-25T18:25:03.242Z, Instrument: SAR, Mode: IW, Satellite: S1A, Size: 0.95 GB" but is not mandatory
	String id: Provider Id
	String link: Link (or equivalent) to access the file
	String footprint: WKT Footprint
	provider: Data Provider that found this item
	Map<String, String> properties: Dictionary of additional properties	
	String volumeName: If this is accessible in a Volume, here we have the nameIf this is accessible in a Volume, here we have the name
	String volumePath: If this is accessible in a Volume, here we have the path in the volume
	String platform: Unique code of the platform/mission of this entry 
	
    Basic info are:
        .Title -> Name of the file
        .Summary -> Description. Supports a sort of std like: "Date: 2021-12-25T18:25:03.242Z, Instrument: SAR, Mode: IW, Satellite: S1A, Size: 0.95 GB" but is not mandatory
        .Id -> Provider unique id
        .Link -> Link to download the file
        .Footprint -> Bounding box in WKT ie POLYGON ((-7.087445 31.109682, -4.389633 31.524973, -4.062707 29.77639, -6.712266 29.357685, -7.087445 31.109682))
                    Note: for POLYGON the convention is LON LAT, LON LAT...
        .Provider -> Provider used to get this info.

    Properties is a dictionary filled with all the properties supported by the data provider.
    Can be seen with the "info" button in the client.
            Some Commonly used, and shown in the client, are:
                ."date": reference Date
                ."instrument": used instrument 
                ."sensoroperationalmode": sensing mode
                ."size": image size as string
                ."relativeOrbit": relative orbit of the acquisition
                ."relativeorbitnumber": same of above, used by the client
                ."platformname": Platform Name

    The libs searchs for a property called relativeOrbit        
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not asQueries:
        raise ValueError("Missing query list")

    aoParams = {"providers": sProvider}
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.post(
            f"{s_sWasdiApiUrl}/rest/search/querylist",
            params=aoParams,
            json=asQueries,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI searchList call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def share_file_to_workspace(
    sOriginWorkspaceId: str,
    sDestinationWorkspaceId: str,
    sProductName: str,
    sParentProcessWorkspaceId: str = None,
    oContext: Context = None,
) -> str:
    """
    Shares a file from one workspace to another. The file is copied to the destination workspace. The file is not moved, so it will still be available in the origin workspace.
    This mirrors FileBufferResource.share.

    Inputs:
    sOriginWorkspaceId: is the unique id of the origin workspace
    sDestinationWorkspaceId: is the unique id of the destination workspace
    sProductName: is the name of the product/file to share
    sParentProcessWorkspaceId: is an optional process workspace id. If provided, the share will be associated with this process workspace. Usually keep it null, is used by the library when triggers a child process

    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sOriginWorkspaceId:
        raise ValueError("Missing origin workspace id")

    if not sDestinationWorkspaceId:
        raise ValueError("Missing destination workspace id")

    if not sProductName:
        raise ValueError("Missing product name")

    aoParams = {
        "originWorkspaceId": sOriginWorkspaceId,
        "destinationWorkspaceId": sDestinationWorkspaceId,
        "productName": sProductName,
        "parent": sParentProcessWorkspaceId,
    }
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/filebuffer/share",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI share call completed with status %s", oResponse.status_code)
        return oResponse.text

@s_oMcpServer.tool()
async def import_product_in_wasdi(oImageImportViewModel: dict, oContext: Context = None) -> str:
    """
    Triggers import/download of an image in WASDI.
    This mirrors FileBufferResource.imageImport.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not oImageImportViewModel:
        raise ValueError("Missing image import payload")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.post(
            f"{s_sWasdiApiUrl}/rest/filebuffer/download",
            json=oImageImportViewModel,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI imageImport call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def publish_product_band_in_wms(
    sFileUrl: str,
    sWorkspaceId: str,
    sBand: str,
    sStyle: str = None,
    sParentProcessWorkspaceId: str = None,
    oContext: Context = None,
) -> str:
    """
    Publishes a band on GeoServer for a file in a workspace.
    This mirrors FileBufferResource.publishBand.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sFileUrl:
        raise ValueError("Missing file url")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    if not sBand:
        raise ValueError("Missing band")

    aoParams = {
        "fileUrl": sFileUrl,
        "workspace": sWorkspaceId,
        "band": sBand,
        "style": sStyle,
        "parent": sParentProcessWorkspaceId,
    }
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/filebuffer/publishband",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI publishBand call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_application_packages_list(sName: str, oContext: Context = None) -> str:
    """
    Gets the list of packages in an application/processor.
    This mirrors PackageManagerResource.getListPackages.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sName:
        raise ValueError("Missing application name")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/packageManager/listPackages",
            params={"name": sName},
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getListPackages call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_application_environment_actions_list(sName: str, oContext: Context = None) -> str:
    """
    Gets the list of actions executed on an application/processor environment.
    This mirrors PackageManagerResource.getEnvironmentActionsList.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sName:
        raise ValueError("Missing application name")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/packageManager/environmentActions",
            params={"name": sName},
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getEnvironmentActionsList call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_application_package_manager_version(sName: str, oContext: Context = None) -> str:
    """
    Gets the version of the Package Manager of an application/processor.
    This mirrors PackageManagerResource.getManagerVersion.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sName:
        raise ValueError("Missing application name")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/packageManager/managerVersion",
            params={"name": sName},
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getManagerVersion call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def update_application_environment_with_action(
    sProcessorId: str,
    sWorkspaceId: str,
    sUpdateCommand: str = None,
    oContext: Context = None,
) -> str:
    """
    Forces an update of the environment of a processor.
    This mirrors PackageManagerResource.environmentUpdate.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessorId:
        raise ValueError("Missing processor id")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    aoParams = {
        "processorId": sProcessorId,
        "workspace": sWorkspaceId,
        "updateCommand": sUpdateCommand,
    }
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/packageManager/environmentupdate",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI environmentUpdate call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def reset_application_action_list(
    sProcessorId: str,
    sWorkspaceId: str,
    oContext: Context = None,
) -> str:
    """
    Resets the action list for a processor.
    This mirrors PackageManagerResource.resetActionList.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessorId:
        raise ValueError("Missing processor id")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    aoParams = {
        "processorId": sProcessorId,
        "workspace": sWorkspaceId,
    }
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/packageManager/reset",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI resetActionList call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def printer_store_new_map(sPrinterViewModelJson: str, oContext: Context = None) -> str:
    """
    Stores a map configuration and returns a UUID for later retrieval.
    This mirrors PrinterResource.storemap.
    Accepts a JSON string representing a PrinterViewModel with baseMap, center (lat/lng), and format (pdf/png) fields.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sPrinterViewModelJson:
        raise ValueError("Missing printer view model JSON")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.post(
            f"{s_sWasdiApiUrl}/rest/print/storemap",
            content=sPrinterViewModelJson,
            headers={
                "x-session-token": sSessionToken,
                "Content-Type": "application/json",
            },
        )
        oResponse.raise_for_status()
        logging.debug("WASDI storemap call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def printer_print(sUUID: str, oContext: Context = None) -> str:
    """
    Retrieves a map image (PNG) or PDF document by UUID.
    This mirrors PrinterResource.print.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sUUID:
        raise ValueError("Missing UUID parameter")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/print",
            params={"uuid": sUUID},
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI print call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def mosaic(
    sDestinationProductName: str,
    sWorkspaceId: str,
    sMosaicSettingJson: str,
    sParentId: str = None,
    oContext: Context = None,
) -> str:
    """
    Triggers a mosaic operation on products.
    This mirrors ProcessingResources.mosaic.
    Accepts a JSON string representing MosaicSetting.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sDestinationProductName:
        raise ValueError("Missing destination product name")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    if not sMosaicSettingJson:
        raise ValueError("Missing mosaic setting JSON")

    aoParams = {
        "name": sDestinationProductName,
        "workspace": sWorkspaceId,
        "parent": sParentId,
    }
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.post(
            f"{s_sWasdiApiUrl}/rest/processing/mosaic",
            content=sMosaicSettingJson,
            params=aoParams,
            headers={
                "x-session-token": sSessionToken,
                "Content-Type": "application/json",
            },
        )
        oResponse.raise_for_status()
        logging.debug("WASDI mosaic call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def regrid(
    sDestinationProductName: str,
    sWorkspaceId: str,
    sRegridSettingJson: str,
    sParentId: str = None,
    oContext: Context = None,
) -> str:
    """
    Triggers a regrid operation on products.
    This mirrors ProcessingResources.regrid.
    Accepts a JSON string representing RegridSetting.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sDestinationProductName:
        raise ValueError("Missing destination product name")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    if not sRegridSettingJson:
        raise ValueError("Missing regrid setting JSON")

    aoParams = {
        "name": sDestinationProductName,
        "workspace": sWorkspaceId,
        "parent": sParentId,
    }
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.post(
            f"{s_sWasdiApiUrl}/rest/processing/regrid",
            content=sRegridSettingJson,
            params=aoParams,
            headers={
                "x-session-token": sSessionToken,
                "Content-Type": "application/json",
            },
        )
        oResponse.raise_for_status()
        logging.debug("WASDI regrid call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def multiSubset(
    sSourceProductName: str,
    sDestinationProductName: str,
    sWorkspaceId: str,
    sMultiSubsetSettingJson: str,
    sParentId: str = None,
    oContext: Context = None,
) -> str:
    """
    Triggers a multi-subset operation on products.
    This mirrors ProcessingResources.multiSubset.
    Accepts a JSON string representing MultiSubsetSetting.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sSourceProductName:
        raise ValueError("Missing source product name")

    if not sDestinationProductName:
        raise ValueError("Missing destination product name")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    if not sMultiSubsetSettingJson:
        raise ValueError("Missing multi-subset setting JSON")

    aoParams = {
        "source": sSourceProductName,
        "name": sDestinationProductName,
        "workspace": sWorkspaceId,
        "parent": sParentId,
    }
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.post(
            f"{s_sWasdiApiUrl}/rest/processing/multisubset",
            content=sMultiSubsetSettingJson,
            params=aoParams,
            headers={
                "x-session-token": sSessionToken,
                "Content-Type": "application/json",
            },
        )
        oResponse.raise_for_status()
        logging.debug("WASDI multiSubset call completed with status %s", oResponse.status_code)
        return oResponse.text

@s_oMcpServer.tool()
async def upload_snap_workflow_file(
    sWorkspaceId: str,
    sName: str,
    sFilePathOrBase64: str,
    sDescription: str = None,
    bPublic: bool = None,
    oContext: Context = None,
) -> str:
    """
    Uploads a new SNAP Workflow XML file.
    This mirrors WorkflowsResource.uploadFile.
    Accepts either a local file path or base64-encoded file content.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    if not sName:
        raise ValueError("Missing workflow name")

    if not sFilePathOrBase64:
        raise ValueError("Missing file path or base64 content")

    aoParams = {
        "workspace": sWorkspaceId,
        "name": sName,
        "description": sDescription,
        "public": bPublic,
    }
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    # Handle file: try as path first, fall back to base64 decode
    try:
        import os
        if os.path.isfile(sFilePathOrBase64):
            with open(sFilePathOrBase64, "rb") as f:
                oFileContent = f.read()
        else:
            import base64
            oFileContent = base64.b64decode(sFilePathOrBase64)
    except Exception as e:
        raise ValueError(f"Invalid file path or base64 content: {str(e)}")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.post(
            f"{s_sWasdiApiUrl}/rest/workflows/uploadfile",
            files={"file": ("workflow.xml", oFileContent, "application/xml")},
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI uploadFile call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def update_snap_workflow_file(
    sWorkflowId: str,
    sFilePathOrBase64: str,
    oContext: Context = None,
) -> str:
    """
    Updates an existing SNAP Workflow XML file.
    This mirrors WorkflowsResource.updateFile.
    Accepts either a local file path or base64-encoded file content.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sWorkflowId:
        raise ValueError("Missing workflow id")

    if not sFilePathOrBase64:
        raise ValueError("Missing file path or base64 content")

    aoParams = {"workflowid": sWorkflowId}

    # Handle file: try as path first, fall back to base64 decode
    try:
        import os
        if os.path.isfile(sFilePathOrBase64):
            with open(sFilePathOrBase64, "rb") as f:
                oFileContent = f.read()
        else:
            import base64
            oFileContent = base64.b64decode(sFilePathOrBase64)
    except Exception as e:
        raise ValueError(f"Invalid file path or base64 content: {str(e)}")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.post(
            f"{s_sWasdiApiUrl}/rest/workflows/updatefile",
            files={"file": ("workflow.xml", oFileContent, "application/xml")},
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI updateWorkflowFile call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_snap_workflow_xml(sWorkflowId: str, oContext: Context = None) -> str:
    """
    Retrieves the XML content of a workflow.
    This mirrors WorkflowsResource.getXML.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sWorkflowId:
        raise ValueError("Missing workflow id")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/workflows/getxml",
            params={"workflowId": sWorkflowId},
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getWorkflowXML call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def update_snap_workflow_xml(
    sWorkflowId: str,
    sGraphXml: str,
    oContext: Context = None,
) -> str:
    """
    Updates the XML content of a workflow.
    This mirrors WorkflowsResource.updateXML.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sWorkflowId:
        raise ValueError("Missing workflow id")

    if not sGraphXml:
        raise ValueError("Missing graph XML content")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.post(
            f"{s_sWasdiApiUrl}/rest/workflows/updatexml",
            content=sGraphXml,
            params={"workflowId": sWorkflowId},
            headers={
                "x-session-token": sSessionToken,
                "Content-Type": "application/xml",
            },
        )
        oResponse.raise_for_status()
        logging.debug("WASDI updateWorkflowXML call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def update_snap_workflow_params(
    sWorkflowId: str,
    sName: str,
    sDescription: str = None,
    bPublic: bool = None,
    oContext: Context = None,
) -> str:
    """
    Updates the parameters of a workflow (name, description, public flag).
    This mirrors WorkflowsResource.updateParams.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sWorkflowId:
        raise ValueError("Missing workflow id")

    if not sName:
        raise ValueError("Missing workflow name")

    aoParams = {
        "workflowid": sWorkflowId,
        "name": sName,
        "description": sDescription,
        "public": bPublic,
    }
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.post(
            f"{s_sWasdiApiUrl}/rest/workflows/updateparams",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI updateWorkflowParams call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_snap_workflows_by_user(oContext: Context = None) -> str:
    """
    Retrieves all workflows for the current user, including public and shared workflows.
    This mirrors WorkflowsResource.getWorkflowsByUser.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/workflows/getbyuser",
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getWorkflowsByUser call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def share_snap_workflow(
    sWorkflowId: str,
    sUserId: str,
    sRights: str = None,
    oContext: Context = None,
) -> str:
    """
    Shares a workflow with another user.
    This mirrors WorkflowsResource.shareWorkflow.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sWorkflowId:
        raise ValueError("Missing workflow id")

    if not sUserId:
        raise ValueError("Missing user id")

    aoParams = {
        "workflowId": sWorkflowId,
        "userId": sUserId,
        "rights": sRights,
    }
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.put(
            f"{s_sWasdiApiUrl}/rest/workflows/share/add",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI shareWorkflow call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def delete_snap_workflow_sharing(
    sWorkflowId: str,
    sUserId: str,
    oContext: Context = None,
) -> str:
    """
    Removes workflow sharing for a user.
    This mirrors WorkflowsResource.deleteUserSharingWorkflow.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sWorkflowId:
        raise ValueError("Missing workflow id")

    if not sUserId:
        raise ValueError("Missing user id")

    aoParams = {
        "workflowId": sWorkflowId,
        "userId": sUserId,
    }

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.delete(
            f"{s_sWasdiApiUrl}/rest/workflows/share/delete",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI deleteWorkflowSharing call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_snap_workflow_sharings(sWorkflowId: str, oContext: Context = None) -> str:
    """
    Retrieves all users with whom a workflow is shared.
    This mirrors WorkflowsResource.getEnableUsersSharedWorkflow.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sWorkflowId:
        raise ValueError("Missing workflow id")

    aoParams = {"workflowId": sWorkflowId}

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/workflows/share/byworkflow",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getWorkflowSharings call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def run_snap_workflow(
    sWorkflowId: str,
    sWorkspaceId: str,
    sWorkflowViewModelJson: str,
    sParentProcessWorkspaceId: str = None,
    oContext: Context = None,
) -> str:
    """
    Executes a workflow in a workspace.
    This mirrors WorkflowsResource.run.
    Accepts a JSON string representing SnapWorkflowViewModel.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sWorkflowId:
        raise ValueError("Missing workflow id")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    if not sWorkflowViewModelJson:
        raise ValueError("Missing workflow view model JSON")

    aoParams = {
        "workspace": sWorkspaceId,
        "parent": sParentProcessWorkspaceId,
    }
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.post(
            f"{s_sWasdiApiUrl}/rest/workflows/run",
            content=sWorkflowViewModelJson,
            params=aoParams,
            headers={
                "x-session-token": sSessionToken,
                "Content-Type": "application/json",
            },
        )
        oResponse.raise_for_status()
        logging.debug("WASDI runWorkflow call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def download_snap_workflow(
    sWorkflowId: str,
    sTokenSessionId: str = None,
    oContext: Context = None,
) -> str:
    """
    Downloads a workflow XML file.
    This mirrors WorkflowsResource.download.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken and not sTokenSessionId:
        raise ValueError("Missing x-session-token header or token query param")

    if not sWorkflowId:
        raise ValueError("Missing workflow id")

    aoParams = {
        "workflowId": sWorkflowId,
        "token": sTokenSessionId,
    }
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    oHeaders = {}
    if sSessionToken:
        oHeaders["x-session-token"] = sSessionToken

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/workflows/download",
            params=aoParams,
            headers=oHeaders,
        )
        oResponse.raise_for_status()
        logging.debug("WASDI downloadWorkflow call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def get_snap_workflow_by_name(sWorkflowName: str, oContext: Context = None) -> str:
    """
    Retrieves a workflow by its name.
    This mirrors WorkflowsResource.getWorkflowByName.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sWorkflowName:
        raise ValueError("Missing workflow name")

    aoParams = {"name": sWorkflowName}

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{s_sWasdiApiUrl}/rest/workflows/byname",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getWorkflowByName call completed with status %s", oResponse.status_code)
        return oResponse.text

if __name__ == "__main__":
    uvicorn.run(oApp, host="0.0.0.0", port=7000)
