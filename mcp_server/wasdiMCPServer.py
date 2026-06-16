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

s_oMcpServer = FastMCP("wasdi-mcp-server", "0.1.0")

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
                "https://www.wasdi.net/wasdiwebserver/rest/ws/getws",
                params={"workspace": sWorkspaceId},
                headers={"x-session-token": sSessionToken}
            )
            oWsResponse.raise_for_status()
            oWsData = json.loads(oWsResponse.text)
            sNodeUrl = oWsData.get("apiUrl", "https://www.wasdi.net/wasdiwebserver")
            logging.debug("Resolved node URL for workspace %s: %s", sWorkspaceId, sNodeUrl)
            return sNodeUrl
    except Exception as e:
        logging.warning("Failed to resolve node URL for workspace %s, falling back to main server: %s", sWorkspaceId, str(e))
        return "https://www.wasdi.net/wasdiwebserver"


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
                "https://www.wasdi.net/wasdiwebserver/rest/process/byid",
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
    return "https://www.wasdi.net/wasdiwebserver"

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
    sUserPrompt is the question or query from the user that needs to be answered using the WASDI documentation.
    """
    oResponse = s_oRAGChain.invokeRAGChain(sUserPrompt)
    return oResponse.content

    
    
@s_oMcpServer.tool()
async def get_workspaces(oContext: Context = None) -> str:
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
            "https://www.wasdi.net/wasdiwebserver/rest/ws/byuser",
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
            "https://www.wasdi.net/wasdiwebserver/rest/ws/getws",
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
            "https://www.wasdi.net/wasdiwebserver/rest/ws/wsnamebyid",
            params={"workspace": sWorkspaceId},
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI get_workspace_name_by_id call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def createWorkspace(sName: str = None, sNodeCode: str = None, oContext: Context = None) -> str:
    """
    Creates a new workspace for the authenticated user. Can be used by the agent to create a new workspace for the user. The workspaceId is the real key. The names are unique: if the name already exists, WASDI will add (1) or (2) etc. 
    
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
            "https://www.wasdi.net/wasdiwebserver/rest/ws/create",
            params={"name": sName, "node": sNodeCode},
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI createWorkspace call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def shareWorkspace(sWorkspaceId: str, sDestinationUserId: str, sRights: str = None, oContext: Context = None) -> str:
    """
    Shares a workspace with another user.
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
            "https://www.wasdi.net/wasdiwebserver/rest/ws/share/add",
            params={"workspace": sWorkspaceId, "userId": sDestinationUserId, "rights": sRights},
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI shareWorkspace call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def getEnabledUsersSharedWorksace(sWorkspaceId: str, oContext: Context = None) -> str:
    """
    Returns the list of users that have access to a workspace.
    sWorkspaceId: is the unique id of the workspace for which we want to get the list of users that have access to it.

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
            "https://www.wasdi.net/wasdiwebserver/rest/ws/share/byworkspace",
            params={"workspace": sWorkspaceId},
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getEnabledUsersSharedWorksace call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def addProductToWorkspace(sProductName: str, sWorkspaceId: str, oContext: Context = None) -> str:
    """
    Adds a product to a workspace. The file that is going to be added must be present in the local node workspace folder but this API does not check
    In WASDI all files are represented as products, even if they are not EO products, so this API can be used to add any file to the workspace, as long as the file is present in the local node workspace folder and the name of the file is provided as an input parameter.
    The path is always relative to the root of the workspace that host the product.
    A typical use case of this API is when the agent needs to add a file that is generated during the execution of a process to the workspace, in order to make it available for the user in the WASDI interface and for other processes that can be executed after.
    Usually the agent generates a file in the local node workspace folder, then it uses this API to add the file to the workspace, providing the name of the file and the id of the workspace as input parameters.

    sProductName: is the name of the product to be added to the workspace, it must be present in the local node workspace folder
    sWorkspaceId: is the unique id of the workspace to which the product will be added

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
async def getByProductName(sProductName: str, sWorkspaceId: str, oContext: Context = None) -> str:
    """
    Returns a product view model by file name. The Product must exists in the workspace. The path of the product is always relative to the root of the workspace.

    sProductName: is the name of the product to be searched in the workspace, it must be present in the workspace
    sWorkspaceId: is the unique id of the workspace where the product is located

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
async def getListByWorkspace(sWorkspaceId: str, oContext: Context = None) -> str:
    """
    Returns the detailed list of products in a workspace. 
    This mirrors ProductResource.getListByWorkspace and is useful when the agent needs to get the full list of products in a workspace with all the details, including metadata, styles, bands information, and so on.
    There are 2 alternatives getLightListByWorkspace and getNamesByWorkspace.
    This API can take a lot of time for workspaces with a lot of products, so it is better to use it only when the agent really needs all the details of all the products in the workspace. If the agent needs only the names of the products, it can use getNamesByWorkspace, if it needs some details but not all, it can use getLightListByWorkspace.
    It will always be possible later to get the details of a specific product using the getByProductName API, so the agent can start with a light call to get the list of products with few details and then get the details of the products that are interesting for it using getByProductName.
    sWorkspaceId: is the unique id of the workspace for which we want to get the list of products.

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
            "https://www.wasdi.net/wasdiwebserver/rest/product/byws",
            params={"workspace": sWorkspaceId},
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getListByWorkspace call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def getLightListByWorkspace(sWorkspaceId: str, oContext: Context = None) -> str:
    """
    Returns the light list of products in a workspace.
    This mirrors ProductResource.getLightListByWorkspace and is useful when the agent needs to get a list of products in a workspace with only basic details, without the full one.
    The light list is faster to be returned by the server and to be processed by the agent, so it is better to use it when the agent needs only some details of the products in the workspace, but not all. If the agent needs all the details of the all the products, it can use getListByWorkspace, if it needs only the names of the products it can use getNamesByWorkspace.
    Once a product name is available, it will always be possible to get the full details of the product using the getByProductName API, so the agent can start with a light call to get the list of products with few details and then get the details of the products that are interesting for it using getByProductName.
    sWorkspaceId: is the unique id of the workspace for which we want to get the light list of products.

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
            "https://www.wasdi.net/wasdiwebserver/rest/product/bywslight",
            params={"workspace": sWorkspaceId},
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getLightListByWorkspace call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def getNamesByWorkspace(sWorkspaceId: str, oContext: Context = None) -> str:
    """
    Returns the file names of the products in a workspace. 
    sWorkspaceId: is the unique id of the workspace for which we want to get the names of the products.
    This mirrors ProductResource.getNamesByWorkspace. It is useful when the agent needs only the names of the products in the workspace, without any other details, for example to check if a product with a specific name exists in the workspace.
        It is the fastest API to get the list of products in a workspace, so it is better to use it when the agent needs only the names of the products. If the agent needs some details but not all, it can use getLightListByWorkspace, if it needs all the details of the products it can use getListByWorkspace.

        the call returns an empty array in case of errors or an array of strings with the names of the products in the workspace. The names are the file names with extension, for example "image.tif", "data.csv", and so on.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            "https://www.wasdi.net/wasdiwebserver/rest/product/namesbyws",
            params={"workspace": sWorkspaceId},
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getNamesByWorkspace call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def getProcessByWorkspace(
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
    Returns a filtered list of process workspaces in a workspace. A process workspace is a process that has been executed in a workspace. 
    This is a node-based API. ProcessWorkspaces are stored per node, not all in the main node. This tool automatically resolves the node URL from the workspace details.

    Inputs are
    sWorkspaceId: is the unique id of the workspace for which we want to get the process workspaces.
    sStatus: is an optional filter for the status of the process workspaces.
    sOperationType: is an optional filter for the operation type of the process workspaces.
    sNamePattern: is an optional filter for the name pattern of the process workspaces.
    sDateFrom: is an optional filter for the start date of the process workspaces.
    sDateTo: is an optional filter for the end date of the process workspaces.
    iStartIndex: is an optional filter for the start index of the process workspaces.
    iEndIndex: is an optional filter for the end index of the process workspaces.

    A process workspace has the following properties:

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
async def getLastProcessByWorkspace(sWorkspaceId: str, oContext: Context = None) -> str:
    """
    Returns the last five process workspaces for a workspace.
    This is a node-based API. This tool automatically resolves the node URL from the workspace details.
    This mirrors ProcessWorkspaceResource.getLastProcessByWorkspace.
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
async def getLastProcessByUser(oContext: Context = None) -> str:
    """
    Returns the last five process workspaces for the authenticated user.
    This mirrors ProcessWorkspaceResource.getLastProcessByUser.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")
    

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            "https://www.wasdi.net/wasdiwebserver/rest/process/lastbyusr",
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getLastProcessByUser call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def getSummary(sWorkspaceId: str = None, oContext: Context = None) -> str:
    """
    Returns process summary counts for a workspace and user.
    This mirrors ProcessWorkspaceResource.getSummary.
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
async def deleteProcess(
    sProcessObjId: str,
    bKillTheEntireTree: bool = None,
    sWorkspaceId: str = None,
    oContext: Context = None,
) -> str:
    """
    Kills a running process workspace.
    This mirrors ProcessWorkspaceResource.deleteProcess.
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
async def getProcessById(sProcessObjId: str, sWorkspaceId: str = None, oContext: Context = None) -> str:
    """
    Returns a process workspace view model by id.
    This mirrors ProcessWorkspaceResource.getProcessById.
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
async def getStatusProcessesById(asProcessesWorkspaceId: list[str], oContext: Context = None) -> str:
    """
    Returns the status of multiple process workspaces in a single call.
    This mirrors ProcessWorkspaceResource.getStatusProcessesById.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not asProcessesWorkspaceId:
        raise ValueError("Missing process id list")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.post(
            "https://www.wasdi.net/wasdiwebserver/rest/process/statusbyid",
            json=asProcessesWorkspaceId,
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getStatusProcessesById call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def getProcessStatusById(sProcessObjId: str, sWorkspaceId: str = None, oContext: Context = None) -> str:
    """
    Returns the status of a single process workspace.
    This mirrors ProcessWorkspaceResource.getProcessStatusById.
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
async def updateProcessById(
    sProcessObjId: str,
    sNewStatus: str,
    iPerc: int,
    sSendToRabbit: str = None,
    sWorkspaceId: str = None,
    oContext: Context = None,
) -> str:
    """
    Updates a process workspace status and progress.
    This mirrors ProcessWorkspaceResource.updateProcessById.
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
async def setProcessPayloadPOST(sProcessObjId: str, sPayload: str, sWorkspaceId: str = None, oContext: Context = None) -> str:
    """
    Sets the payload of a process workspace using the POST API.
    This mirrors ProcessWorkspaceResource.setProcessPayloadPOST.
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
async def getPayload(sProcessObjId: str, sWorkspaceId: str = None, oContext: Context = None) -> str:
    """
    Returns the payload of a process workspace.
    This mirrors ProcessWorkspaceResource.getPayload.
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
async def getProcessParameters(sProcessObjId: str, sWorkspaceId: str = None, oContext: Context = None) -> str:
    """
    Returns the JSON parameters of a process workspace.
    This mirrors ProcessWorkspaceResource.getProcessParameters.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessObjId:
        raise ValueError("Missing process id")

    sNodeUrl = await getNodeUrlForProcessWorkspace(sProcessObjId, sSessionToken, sWorkspaceId)

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            f"{sNodeUrl}/rest/process/paramsbyid",
            params={"procws": sProcessObjId},
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getProcessParameters call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def uploadProcessor(
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
    Uploads a processor zip file to WASDI.
    This mirrors ProcessorsResource.uploadProcessor.
    The file is read from the local filesystem and sent as multipart/form-data.
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
                "https://www.wasdi.net/wasdiwebserver/rest/processors/uploadprocessor",
                params=aoParams,
                files=aoFiles,
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI uploadProcessor call completed with status %s", oResponse.status_code)
            return oResponse.text


@s_oMcpServer.tool()
async def getDeployedProcessors(oContext: Context = None) -> str:
    """
    Returns all deployed processors visible to the authenticated user.
    This mirrors ProcessorsResource.getDeployedProcessors.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            "https://www.wasdi.net/wasdiwebserver/rest/processors/getdeployed",
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getDeployedProcessors call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def getSingleDeployedProcessor(
    sProcessorId: str = None,
    sProcessorName: str = None,
    oContext: Context = None,
) -> str:
    """
    Returns a single deployed processor by id or name.
    This mirrors ProcessorsResource.getSingleDeployedProcessor.
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
            "https://www.wasdi.net/wasdiwebserver/rest/processors/getprocessor",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getSingleDeployedProcessor call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def getMarketPlaceAppList(oFilters: dict = None, oContext: Context = None) -> str:
    """
    Returns the marketplace processor list filtered by the given criteria.
    This mirrors ProcessorsResource.getMarketPlaceAppList.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.post(
            "https://www.wasdi.net/wasdiwebserver/rest/processors/getmarketlist",
            json=oFilters or {},
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getMarketPlaceAppList call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def getMarketPlaceAppDetail(sProcessorName: str, oContext: Context = None) -> str:
    """
    Returns the detailed marketplace information for a processor.
    This mirrors ProcessorsResource.getMarketPlaceAppDetail.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessorName:
        raise ValueError("Missing processor name")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            "https://www.wasdi.net/wasdiwebserver/rest/processors/getmarketdetail",
            params={"processorname": sProcessorName},
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getMarketPlaceAppDetail call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def runPost(
    sName: str,
    sWorkspaceId: str,
    sEncodedJson: str,
    sParentProcessWorkspaceId: str = None,
    bNotify: bool = None,
    oContext: Context = None,
) -> str:
    """
    Runs a processor using the POST endpoint.
    This mirrors ProcessorsResource.runPost.
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
            "https://www.wasdi.net/wasdiwebserver/rest/processors/run",
            params=aoParams,
            content=sEncodedJson,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI runPost call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def getCreditsForRun(sProcessorId: str, sEncodedJson: str, oContext: Context = None) -> str:
    """
    Returns the estimated credits needed for a processor run.
    This mirrors ProcessorsResource.getCreditsForRun.
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
            "https://www.wasdi.net/wasdiwebserver/rest/processors/getcredits",
            params={"processorId": sProcessorId},
            content=sEncodedJson,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getCreditsForRun call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def help(sProcessorName: str, oContext: Context = None) -> str:
    """
    Returns the help text for a processor.
    This mirrors ProcessorsResource.help.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessorName:
        raise ValueError("Missing processor name")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            "https://www.wasdi.net/wasdiwebserver/rest/processors/help",
            params={"name": sProcessorName},
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI help call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def countLogs(sProcessWorkspaceId: str, oContext: Context = None) -> str:
    """
    Returns the count of log rows for a processor workspace.
    This mirrors ProcessorsResource.countLogs.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessWorkspaceId:
        raise ValueError("Missing process workspace id")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            "https://www.wasdi.net/wasdiwebserver/rest/processors/logs/count",
            params={"processworkspace": sProcessWorkspaceId},
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI countLogs call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def getLogs(sProcessWorkspaceId: str, iStartRow: int = None, iEndRow: int = None, oContext: Context = None) -> str:
    """
    Returns a paginated list of log rows for a processor workspace.
    This mirrors ProcessorsResource.getLogs.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessWorkspaceId:
        raise ValueError("Missing process workspace id")

    aoParams = {"processworkspace": sProcessWorkspaceId, "startrow": iStartRow, "endrow": iEndRow}
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            "https://www.wasdi.net/wasdiwebserver/rest/processors/logs/list",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getLogs call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def redeployProcessor(sProcessorId: str, sWorkspaceId: str, oContext: Context = None) -> str:
    """
    Forces a redeploy of a processor.
    This mirrors ProcessorsResource.redeployProcessor.
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
            "https://www.wasdi.net/wasdiwebserver/rest/processors/redeploy",
            params={"processorId": sProcessorId, "workspace": sWorkspaceId},
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI redeployProcessor call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def updateProcessor(sProcessorId: str, oUpdatedProcessorVM: dict, oContext: Context = None) -> str:
    """
    Updates the processor metadata.
    This mirrors ProcessorsResource.updateProcessor.
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
            "https://www.wasdi.net/wasdiwebserver/rest/processors/update",
            params={"processorId": sProcessorId},
            json=oUpdatedProcessorVM,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI updateProcessor call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def updateProcessorDetails(sProcessorId: str, oUpdatedProcessorVM: dict, oContext: Context = None) -> str:
    """
    Updates processor details and payment fields.
    This mirrors ProcessorsResource.updateProcessorDetails.
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
            "https://www.wasdi.net/wasdiwebserver/rest/processors/updatedetails",
            params={"processorId": sProcessorId},
            json=oUpdatedProcessorVM,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI updateProcessorDetails call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def updateProcessorFiles(
    sFilePath: str,
    sProcessorId: str,
    sWorkspaceId: str,
    sInputFileName: str = None,
    oContext: Context = None,
) -> str:
    """
    Updates the processor files using a local file path.
    This mirrors ProcessorsResource.updateProcessorFiles.
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
                "https://www.wasdi.net/wasdiwebserver/rest/processors/updatefiles",
                params=aoParams,
                files=aoFiles,
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI updateProcessorFiles call completed with status %s", oResponse.status_code)
            return oResponse.text


@s_oMcpServer.tool()
async def downloadProcessor(sProcessorId: str, sTokenSessionId: str = None, oContext: Context = None) -> str:
    """
    Downloads a processor zip. The binary response is returned as base64 text.
    This mirrors ProcessorsResource.downloadProcessor.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken and not sTokenSessionId:
        raise ValueError("Missing x-session-token header")

    if not sProcessorId:
        raise ValueError("Missing processor id")

    aoParams = {"token": sTokenSessionId, "processorId": sProcessorId}
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            "https://www.wasdi.net/wasdiwebserver/rest/processors/downloadprocessor",
            params=aoParams,
            headers={"x-session-token": sSessionToken} if sSessionToken else {},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI downloadProcessor call completed with status %s", oResponse.status_code)
        return oResponse.content.hex()


@s_oMcpServer.tool()
async def shareProcessor(sProcessorId: str, sUserId: str, sRights: str = None, oContext: Context = None) -> str:
    """
    Shares a processor with a user.
    This mirrors ProcessorsResource.shareProcessor.
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
            "https://www.wasdi.net/wasdiwebserver/rest/processors/share/add",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI shareProcessor call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def getEnabledUsersSharedProcessor(sProcessorId: str, oContext: Context = None) -> str:
    """
    Returns the list of users who can access a processor.
    This mirrors ProcessorsResource.getEnabledUsersSharedProcessor.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessorId:
        raise ValueError("Missing processor id")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            "https://www.wasdi.net/wasdiwebserver/rest/processors/share/byprocessor",
            params={"processorId": sProcessorId},
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getEnabledUsersSharedProcessor call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def getUI(sProcessorName: str, oContext: Context = None) -> str:
    """
    Returns the JSON UI definition of a processor.
    This mirrors ProcessorsResource.getUI.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessorName:
        raise ValueError("Missing processor name")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            "https://www.wasdi.net/wasdiwebserver/rest/processors/ui",
            params={"name": sProcessorName},
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getUI call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def getProcessorBuildLogs(sProcessorId: str, oContext: Context = None) -> str:
    """
    Returns the build logs for a processor.
    This mirrors ProcessorsResource.getProcessorBuildLogs.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessorId:
        raise ValueError("Missing processor id")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            "https://www.wasdi.net/wasdiwebserver/rest/processors/logs/build",
            params={"processorId": sProcessorId},
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getProcessorBuildLogs call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def downloadEntryByName(sFileName: str, sWorkspaceId: str, sTokenSessionId: str = None, sProcessObjId: str = None, sDisposition: str = None, oContext: Context = None) -> str:
    """
    Downloads a file by name from a workspace. Returns binary as hex text.
    This mirrors CatalogResources.downloadEntryByName.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken and not sTokenSessionId:
        raise ValueError("Missing x-session-token header")

    if not sFileName:
        raise ValueError("Missing file name")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    aoParams = {"filename": sFileName, "workspace": sWorkspaceId, "token": sTokenSessionId, "procws": sProcessObjId, "disposition": sDisposition}
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            "https://www.wasdi.net/wasdiwebserver/rest/catalog/downloadbyname",
            params=aoParams,
            headers={"x-session-token": sSessionToken} if sSessionToken else {},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI downloadEntryByName call completed with status %s", oResponse.status_code)
        return oResponse.content.hex()


@s_oMcpServer.tool()
async def checkFileByNode(sFileName: str, sWorkspaceId: str, oContext: Context = None) -> str:
    """
    Checks if a file exists on the current node.
    This mirrors CatalogResources.checkFileByNode.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sFileName:
        raise ValueError("Missing file name")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            "https://www.wasdi.net/wasdiwebserver/rest/catalog/fileOnNode",
            params={"token": sSessionToken, "filename": sFileName, "workspace": sWorkspaceId},
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI checkFileByNode call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def checkDownloadEntryAvailabilityByName(sFileName: str, sWorkspaceId: str, sProcessObjId: str = None, sVolumePath: str = None, oContext: Context = None) -> str:
    """
    Checks if a file is available for download.
    This mirrors CatalogResources.checkDownloadEntryAvailabilityByName.
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

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            "https://www.wasdi.net/wasdiwebserver/rest/catalog/checkdownloadavaialibitybyname",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI checkDownloadEntryAvailabilityByName call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def ingestFileInWorkspace(sFileName: str, sWorkspaceId: str, sParentProcessWorkspaceId: str = None, sStyle: str = None, sPlatform: str = None, oContext: Context = None) -> str:
    """
    Ingests a file already existing in a workspace.
    This mirrors CatalogResources.ingestFileInWorkspace.
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

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            "https://www.wasdi.net/wasdiwebserver/rest/catalog/upload/ingestinws",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI ingestFileInWorkspace call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def getProductProperties(sFileName: str, sWorkspaceId: str, bGetChecksum: bool = None, oContext: Context = None) -> str:
    """
    Returns the properties of a product/file.
    This mirrors CatalogResources.getProductProperties.
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
            "https://www.wasdi.net/wasdiwebserver/rest/catalog/properties",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getProductProperties call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def count(sQuery: str, sProviders: str = None, oContext: Context = None) -> str:
    """
    Returns the total number of EO search results for a query.
    This mirrors OpenSearchResource.count.
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
            "https://www.wasdi.net/wasdiwebserver/rest/search/query/count",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI count call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def search(
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
            "https://www.wasdi.net/wasdiwebserver/rest/search/query",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI search call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def getDataProviders(oContext: Context = None) -> str:
    """
    Returns the list of available EO data providers.
    This mirrors OpenSearchResource.getDataProviders.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            "https://www.wasdi.net/wasdiwebserver/rest/search/providers",
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getDataProviders call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def countList(asQueries: list[str], sProviders: str = None, oContext: Context = None) -> str:
    """
    Returns the total count of EO results for a list of queries.
    This mirrors OpenSearchResource.countList.
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
            "https://www.wasdi.net/wasdiwebserver/rest/search/query/countlist",
            params=aoParams,
            json=asQueries,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI countList call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def searchList(asQueries: list[str], sProvider: str = None, oContext: Context = None) -> str:
    """
    Executes EO searches for a list of queries.
    This mirrors OpenSearchResource.searchList.
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
            "https://www.wasdi.net/wasdiwebserver/rest/search/querylist",
            params=aoParams,
            json=asQueries,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI searchList call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def share(
    sOriginWorkspaceId: str,
    sDestinationWorkspaceId: str,
    sProductName: str,
    sParentProcessWorkspaceId: str = None,
    oContext: Context = None,
) -> str:
    """
    Shares a file from one workspace to another.
    This mirrors FileBufferResource.share.
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
            "https://www.wasdi.net/wasdiwebserver/rest/filebuffer/share",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI share call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def download(
    sFileUrl: str,
    sFileName: str,
    sProvider: str,
    sWorkspaceId: str,
    sBoundingBox: str = None,
    sParentProcessWorkspaceId: str = None,
    sPlatform: str = None,
    oContext: Context = None,
) -> str:
    """
    Triggers import/download of an image in WASDI (GET compatibility endpoint).
    This mirrors FileBufferResource.download.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sFileName:
        raise ValueError("Missing file name")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    aoParams = {
        "fileUrl": sFileUrl,
        "name": sFileName,
        "provider": sProvider,
        "workspace": sWorkspaceId,
        "bbox": sBoundingBox,
        "parent": sParentProcessWorkspaceId,
        "platform": sPlatform,
    }
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            "https://www.wasdi.net/wasdiwebserver/rest/filebuffer/download",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI download call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def imageImport(oImageImportViewModel: dict, oContext: Context = None) -> str:
    """
    Triggers import/download of an image in WASDI (POST endpoint).
    This mirrors FileBufferResource.imageImport.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not oImageImportViewModel:
        raise ValueError("Missing image import payload")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.post(
            "https://www.wasdi.net/wasdiwebserver/rest/filebuffer/download",
            json=oImageImportViewModel,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI imageImport call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def publishBand(
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
            "https://www.wasdi.net/wasdiwebserver/rest/filebuffer/publishband",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI publishBand call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def getListPackages(sName: str, oContext: Context = None) -> str:
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
            "https://www.wasdi.net/wasdiwebserver/rest/packageManager/listPackages",
            params={"name": sName},
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getListPackages call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def getEnvironmentActionsList(sName: str, oContext: Context = None) -> str:
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
            "https://www.wasdi.net/wasdiwebserver/rest/packageManager/environmentActions",
            params={"name": sName},
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getEnvironmentActionsList call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def getManagerVersion(sName: str, oContext: Context = None) -> str:
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
            "https://www.wasdi.net/wasdiwebserver/rest/packageManager/managerVersion",
            params={"name": sName},
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getManagerVersion call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def environmentUpdate(
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
            "https://www.wasdi.net/wasdiwebserver/rest/packageManager/environmentupdate",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI environmentUpdate call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def resetActionList(
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
            "https://www.wasdi.net/wasdiwebserver/rest/packageManager/reset",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI resetActionList call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def storemap(sPrinterViewModelJson: str, oContext: Context = None) -> str:
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
            "https://www.wasdi.net/wasdiwebserver/rest/print/storemap",
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
async def print(sUUID: str, oContext: Context = None) -> str:
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
            "https://www.wasdi.net/wasdiwebserver/rest/print",
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
            "https://www.wasdi.net/wasdiwebserver/rest/processing/mosaic",
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
            "https://www.wasdi.net/wasdiwebserver/rest/processing/regrid",
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
            "https://www.wasdi.net/wasdiwebserver/rest/processing/multisubset",
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
async def runProcess(
    sOperationType: str,
    sProductName: str,
    sParameterXml: str,
    sParentProcessWorkspaceId: str = None,
    sOperationSubType: str = None,
    oContext: Context = None,
) -> str:
    """
    Runs a generic processing operation with provided parameters.
    This mirrors ProcessingResources.runProcess.
    Accepts an XML string representing the operation parameter.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sOperationType:
        raise ValueError("Missing operation type")

    if not sProductName:
        raise ValueError("Missing product name")

    if not sParameterXml:
        raise ValueError("Missing parameter XML")

    aoParams = {
        "operation": sOperationType,
        "name": sProductName,
        "parent": sParentProcessWorkspaceId,
        "subtype": sOperationSubType,
    }
    aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.post(
            "https://www.wasdi.net/wasdiwebserver/rest/processing/run",
            content=sParameterXml,
            params=aoParams,
            headers={
                "x-session-token": sSessionToken,
                "Content-Type": "application/xml",
            },
        )
        oResponse.raise_for_status()
        logging.debug("WASDI runProcess call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def uploadFile(
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
            "https://www.wasdi.net/wasdiwebserver/rest/workflows/uploadfile",
            files={"file": ("workflow.xml", oFileContent, "application/xml")},
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI uploadFile call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def updateWorkflowFile(
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
            "https://www.wasdi.net/wasdiwebserver/rest/workflows/updatefile",
            files={"file": ("workflow.xml", oFileContent, "application/xml")},
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI updateWorkflowFile call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def getWorkflowXML(sWorkflowId: str, oContext: Context = None) -> str:
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
            "https://www.wasdi.net/wasdiwebserver/rest/workflows/getxml",
            params={"workflowId": sWorkflowId},
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getWorkflowXML call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def updateWorkflowXML(
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
            "https://www.wasdi.net/wasdiwebserver/rest/workflows/updatexml",
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
async def updateWorkflowParams(
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
            "https://www.wasdi.net/wasdiwebserver/rest/workflows/updateparams",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI updateWorkflowParams call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def getWorkflowsByUser(oContext: Context = None) -> str:
    """
    Retrieves all workflows for the current user, including public and shared workflows.
    This mirrors WorkflowsResource.getWorkflowsByUser.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            "https://www.wasdi.net/wasdiwebserver/rest/workflows/getbyuser",
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getWorkflowsByUser call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def shareWorkflow(
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
            "https://www.wasdi.net/wasdiwebserver/rest/workflows/share/add",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI shareWorkflow call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def deleteWorkflowSharing(
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
            "https://www.wasdi.net/wasdiwebserver/rest/workflows/share/delete",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI deleteWorkflowSharing call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def getWorkflowSharings(sWorkflowId: str, oContext: Context = None) -> str:
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
            "https://www.wasdi.net/wasdiwebserver/rest/workflows/share/byworkflow",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getWorkflowSharings call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def runWorkflow(
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
            "https://www.wasdi.net/wasdiwebserver/rest/workflows/run",
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
async def downloadWorkflow(
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
            "https://www.wasdi.net/wasdiwebserver/rest/workflows/download",
            params=aoParams,
            headers=oHeaders,
        )
        oResponse.raise_for_status()
        logging.debug("WASDI downloadWorkflow call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def getWorkflowByName(sWorkflowName: str, oContext: Context = None) -> str:
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
            "https://www.wasdi.net/wasdiwebserver/rest/workflows/byname",
            params=aoParams,
            headers={"x-session-token": sSessionToken},
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getWorkflowByName call completed with status %s", oResponse.status_code)
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