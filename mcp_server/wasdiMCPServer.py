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

s_sPromptTemplate = """Use the context to answer the user's question. You are a WASDI and Earth Observation (EO) expert, you help users to use WASDI interface and to code WASDI applications using wasdi libraries. 
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
    Creates a new workspace for the authenticated user.
    
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

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            "https://www.wasdi.net/wasdiwebserver/rest/product/addtows",
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
        metadataFileReference:
        metadataFileCreated:
        metadata: 
        bandsGroups: 
        style:
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProductName:
        raise ValueError("Missing product name")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            "https://www.wasdi.net/wasdiwebserver/rest/product/byname",
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
    This mirrors ProductResource.getListByWorkspace.
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
    This mirrors ProductResource.getLightListByWorkspace.
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
    This mirrors ProductResource.getNamesByWorkspace.
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
    Returns a filtered list of process workspaces in a workspace.
    This mirrors ProcessWorkspaceResource.getProcessByWorkspace.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

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
            "https://www.wasdi.net/wasdiwebserver/rest/process/byws",
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
    This mirrors ProcessWorkspaceResource.getLastProcessByWorkspace.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sWorkspaceId:
        raise ValueError("Missing workspace id")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            "https://www.wasdi.net/wasdiwebserver/rest/process/lastbyws",
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

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            "https://www.wasdi.net/wasdiwebserver/rest/process/summary",
            params=aoParams,
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getSummary call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def deleteProcess(sProcessObjId: str, bKillTheEntireTree: bool = None, oContext: Context = None) -> str:
    """
    Deletes a running process workspace.
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

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            "https://www.wasdi.net/wasdiwebserver/rest/process/delete",
            params=aoParams,
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI deleteProcess call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def getProcessById(sProcessObjId: str, oContext: Context = None) -> str:
    """
    Returns a process workspace view model by id.
    This mirrors ProcessWorkspaceResource.getProcessById.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessObjId:
        raise ValueError("Missing process id")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            "https://www.wasdi.net/wasdiwebserver/rest/process/byid",
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
async def getProcessStatusById(sProcessObjId: str, oContext: Context = None) -> str:
    """
    Returns the status of a single process workspace.
    This mirrors ProcessWorkspaceResource.getProcessStatusById.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessObjId:
        raise ValueError("Missing process id")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            "https://www.wasdi.net/wasdiwebserver/rest/process/getstatusbyid",
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

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            "https://www.wasdi.net/wasdiwebserver/rest/process/updatebyid",
            params=aoParams,
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI updateProcessById call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def setProcessPayloadPOST(sProcessObjId: str, sPayload: str, oContext: Context = None) -> str:
    """
    Sets the payload of a process workspace using the POST API.
    This mirrors ProcessWorkspaceResource.setProcessPayloadPOST.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessObjId:
        raise ValueError("Missing process id")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.post(
            "https://www.wasdi.net/wasdiwebserver/rest/process/setpayload",
            params={"procws": sProcessObjId},
            content=sPayload,
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI setProcessPayloadPOST call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def getPayload(sProcessObjId: str, oContext: Context = None) -> str:
    """
    Returns the payload of a process workspace.
    This mirrors ProcessWorkspaceResource.getPayload.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessObjId:
        raise ValueError("Missing process id")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            "https://www.wasdi.net/wasdiwebserver/rest/process/payload",
            params={"procws": sProcessObjId},
            headers={"x-session-token": sSessionToken}
        )
        oResponse.raise_for_status()
        logging.debug("WASDI getPayload call completed with status %s", oResponse.status_code)
        return oResponse.text


@s_oMcpServer.tool()
async def getProcessParameters(sProcessObjId: str, oContext: Context = None) -> str:
    """
    Returns the JSON parameters of a process workspace.
    This mirrors ProcessWorkspaceResource.getProcessParameters.
    """
    sSessionToken = getSessionToken(oContext)

    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    if not sProcessObjId:
        raise ValueError("Missing process id")

    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get(
            "https://www.wasdi.net/wasdiwebserver/rest/process/paramsbyid",
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