import httpx
import logging

def getClass(sClassName):
    asParts = sClassName.split('.')
    oModule = ".".join(asParts[:-1])
    oType = __import__(oModule)
    for sComponent in asParts[1:]:
        oType = getattr(oType, sComponent)
    return oType


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


async def getNodeUrlForWorkspace(sWorkspaceId: str, sSessionToken: str, sWasdiApiUrl: str) -> str:
    """Utility function to resolve the correct node URL for a workspace.
    This is used for node-based APIs that store data per node, not on the main server.
    
    Args:
        sWorkspaceId: The workspace ID for which to resolve the node URL
        sSessionToken: The session token for authentication
        s_sWasdiApiUrl: The base URL of the main WASDI API
        
    Returns:
        The node base URL (apiUrl from workspace details), or falls back to main server if resolution fails
    """
    import json
    
    try:
        async with httpx.AsyncClient() as oClient:
            oWsResponse = await oClient.get(
                f"{sWasdiApiUrl}/rest/ws/getws",
                params={"workspace": sWorkspaceId},
                headers={"x-session-token": sSessionToken}
            )
            oWsResponse.raise_for_status()
            oWsData = json.loads(oWsResponse.text)
            sNodeUrl = oWsData.get("apiUrl", sWasdiApiUrl)
            logging.debug("Resolved node URL for workspace %s: %s", sWorkspaceId, sNodeUrl)
            return sNodeUrl
    except Exception as e:
        logging.warning("Failed to resolve node URL for workspace %s, falling back to main server: %s", sWorkspaceId, str(e))
        return sWasdiApiUrl


async def getWorkspaceIdForProcessWorkspace(sProcessObjId: str, sSessionToken: str, sWasdiApiUrl: str) -> str:
    """Resolve the workspace id associated to a process workspace id.
    Returns an empty string if the process or workspace cannot be resolved.
    """
    if not sProcessObjId:
        return ""

    import json

    try:
        async with httpx.AsyncClient() as oClient:
            oProcessResponse = await oClient.get(
                f"{sWasdiApiUrl}/rest/process/byid",
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
    sWasdiApiUrl: str = ""
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
    return sWasdiApiUrl
