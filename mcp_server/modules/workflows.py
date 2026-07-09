import logging
import httpx
import os
import base64
import json
from mcp.server.fastmcp import FastMCP, Context
from utils.Utils import *

def register_workflow_tools(mcp_server: FastMCP, wasdi_api_url: str):
    """Registers all tools related to managing, sharing, and executing SNAP Engine XML workflow graphs."""

    @mcp_server.tool(name="get_snap_workflows_by_user")
    async def get_snap_workflows_by_user(oContext: Context = None) -> str:
        """Retrieves all SNAP workflows available to the current user, including owned, shared, and public graphs.
        
        Use this tool to discover workflow IDs or names before attempting edits or execution.
        """
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{wasdi_api_url}/rest/workflows/getbyuser",
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI get_snap_workflows_by_user completed.")
            return oResponse.text


    @mcp_server.tool(name="get_snap_workflow_by_name")
    async def get_snap_workflow_by_name(sWorkflowName: str, oContext: Context = None) -> str:
        """Retrieves detailed metadata specifications for a single SNAP workflow filtered by its unique name."""
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")
        if not sWorkflowName:
            raise ValueError("Missing workflow name")

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{wasdi_api_url}/rest/workflows/byname",
                params={"name": sWorkflowName},
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI get_snap_workflow_by_name completed.")
            return oResponse.text


    @mcp_server.tool(name="update_snap_workflow_params")
    async def update_snap_workflow_params(
        sWorkflowId: str,
        sName: str,
        sDescription: str = None,
        bPublic: bool = None,
        oContext: Context = None,
    ) -> str:
        """Updates high-level indexing parameters (name, description, visibility) for an existing SNAP workflow.

        Args:
            sWorkflowId: The unique system identifier of the target workflow.
            sName: The new or existing name code to assign.
            sDescription: Optional summary explaining what the graph graph chains together.
            bPublic: Set True to expose this graph publicly to all WASDI users.
        """
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")
        if not sWorkflowId or not sName:
            raise ValueError("Missing required operational inputs: sWorkflowId and sName are mandatory.")

        aoParams = {
            "workflowid": sWorkflowId,
            "name": sName,
            "description": sDescription,
            "public": bPublic,
        }
        aoParams = {k: v for k, v in aoParams.items() if v is not None}

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.post(
                f"{wasdi_api_url}/rest/workflows/updateparams",
                params=aoParams,
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI update_snap_workflow_params completed.")
            return oResponse.text


    @mcp_server.tool(name="get_snap_workflow_xml")
    async def get_snap_workflow_xml(sWorkflowId: str, oContext: Context = None) -> str:
        """Retrieves the raw internal SNAP graph XML code for inspection or parameter extraction.
        
        Use this tool when you need to read the specific SNAP processing steps, node operators, 
        or band bands configurations configured inside the graph file.
        """
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")
        if not sWorkflowId:
            raise ValueError("Missing workflow id")

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{wasdi_api_url}/rest/workflows/getxml",
                params={"workflowId": sWorkflowId},
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI get_snap_workflow_xml completed.")
            return oResponse.text


    @mcp_server.tool(name="update_snap_workflow_xml")
    async def update_snap_workflow_xml(sWorkflowId: str, sGraphXml: str, oContext: Context = None) -> str:
        """Overwrites the raw internal SNAP processing XML code graph configuration for a workflow.

        Args:
            sWorkflowId: The unique system identifier of the workflow being edited.
            sGraphXml: The modified, valid SNAP Engine compliant XML code string.
        """
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")
        if not sWorkflowId or not sGraphXml:
            raise ValueError("Missing target sWorkflowId or raw sGraphXml payload configuration entries.")

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.post(
                f"{wasdi_api_url}/rest/workflows/updatexml",
                content=sGraphXml,
                params={"workflowId": sWorkflowId},
                headers={
                    "x-session-token": sSessionToken,
                    "Content-Type": "application/xml",
                },
            )
            oResponse.raise_for_status()
            logging.debug("WASDI update_snap_workflow_xml completed.")
            return oResponse.text


    @mcp_server.tool(name="upload_snap_workflow_file")
    async def upload_snap_workflow_file(
        sWorkspaceId: str,
        sName: str,
        sFilePathOrBase64: str,
        sDescription: str = None,
        bPublic: bool = None,
        oContext: Context = None,
    ) -> str:
        """Deploys a brand-new SNAP workflow by pushing an XML graph file to the WASDI cluster.

        Args:
            sWorkspaceId: Active workspace ID context scope.
            sName: Unique name assignment tracking identifier for the workflow.
            sFilePathOrBase64: Pass an absolute local filesystem file path pointing to an XML file, 
                               OR pass a direct raw base64-encoded file string representing the graph.
            sDescription: Optional operational guide summary text.
            bPublic: Set True to publish to all system cluster accounts.
        """
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")
        if not sWorkspaceId or not sName or not sFilePathOrBase64:
            raise ValueError("Missing tracking parameters. Verify sWorkspaceId, sName, and sFilePathOrBase64.")

        aoParams = {
            "workspace": sWorkspaceId,
            "name": sName,
            "description": sDescription,
            "public": bPublic,
        }
        aoParams = {k: v for k, v in aoParams.items() if v is not None}

        # Dynamic Content Extraction Engine
        try:
            if os.path.isfile(sFilePathOrBase64):
                with open(sFilePathOrBase64, "rb") as f:
                    oFileContent = f.read()
            else:
                oFileContent = base64.b64decode(sFilePathOrBase64)
        except Exception as e:
            raise ValueError(f"Failed parsing target schema input content stream. Verify path or structural base64 integrity: {e}")

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.post(
                f"{wasdi_api_url}/rest/workflows/uploadfile",
                files={"file": ("workflow.xml", oFileContent, "application/xml")},
                params=aoParams,
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI upload_snap_workflow_file completed.")
            return oResponse.text


    @mcp_server.tool(name="run_snap_workflow")
    async def run_snap_workflow(sWorkflowId: str, sWorkspaceId: str, oWorkflowViewModel: dict, oContext: Context = None) -> str:
        """Asynchronously executes a deployed SNAP graph workflow operation within a targeted workspace scope.

        CRITICAL Optimization: The 'oWorkflowViewModel' parameter is typed natively as a dictionary/JSON object. 
        The agent can pass natural nested structures directly without applying manual string escaped slashes.

        Args:
            sWorkflowId: The targeted workflow ID tracker.
            sWorkspaceId: The execution workspace bucket context housing your geospatial inputs.
            oWorkflowViewModel: The structured configurations dict representing the SnapWorkflowViewModel execution contract.
        """
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")
        if not sWorkflowId or not sWorkspaceId or oWorkflowViewModel is None:
            raise ValueError("Missing primary input properties required for starting workflow threads.")

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.post(
                f"{wasdi_api_url}/rest/workflows/run",
                json=oWorkflowViewModel,
                params={"workspace": sWorkspaceId},
                headers={
                    "x-session-token": sSessionToken,
                    "Content-Type": "application/json",
                },
            )
            oResponse.raise_for_status()
            logging.debug("WASDI run_snap_workflow successfully started.")
            return oResponse.text
        


# @s_oMcpServer.tool()
# async def update_snap_workflow_file(
#     sWorkflowId: str,
#     sFilePathOrBase64: str,
#     oContext: Context = None,
# ) -> str:
#     """
#     Updates an existing SNAP Workflow XML file.
#     This mirrors WorkflowsResource.updateFile.
#     Accepts either a local file path or base64-encoded file content.
#     """
#     sSessionToken = getSessionToken(oContext)

#     if not sSessionToken:
#         raise ValueError("Missing x-session-token header")

#     if not sWorkflowId:
#         raise ValueError("Missing workflow id")

#     if not sFilePathOrBase64:
#         raise ValueError("Missing file path or base64 content")

#     aoParams = {"workflowid": sWorkflowId}

#     # Handle file: try as path first, fall back to base64 decode
#     try:
#         import os
#         if os.path.isfile(sFilePathOrBase64):
#             with open(sFilePathOrBase64, "rb") as f:
#                 oFileContent = f.read()
#         else:
#             import base64
#             oFileContent = base64.b64decode(sFilePathOrBase64)
#     except Exception as e:
#         raise ValueError(f"Invalid file path or base64 content: {str(e)}")

#     async with httpx.AsyncClient() as oClient:
#         oResponse = await oClient.post(
#             f"{s_sWasdiApiUrl}/rest/workflows/updatefile",
#             files={"file": ("workflow.xml", oFileContent, "application/xml")},
#             params=aoParams,
#             headers={"x-session-token": sSessionToken},
#         )
#         oResponse.raise_for_status()
#         logging.debug("WASDI updateWorkflowFile call completed with status %s", oResponse.status_code)
#         return oResponse.text
        
# @s_oMcpServer.tool()
# async def share_snap_workflow(
#     sWorkflowId: str,
#     sUserId: str,
#     sRights: str = None,
#     oContext: Context = None,
# ) -> str:
#     """
#     Shares a workflow with another user.
#     """
#     sSessionToken = getSessionToken(oContext)

#     if not sSessionToken:
#         raise ValueError("Missing x-session-token header")

#     if not sWorkflowId:
#         raise ValueError("Missing workflow id")

#     if not sUserId:
#         raise ValueError("Missing user id")

#     aoParams = {
#         "workflowId": sWorkflowId,
#         "userId": sUserId,
#         "rights": sRights,
#     }
#     aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

#     async with httpx.AsyncClient() as oClient:
#         oResponse = await oClient.put(
#             f"{s_sWasdiApiUrl}/rest/workflows/share/add",
#             params=aoParams,
#             headers={"x-session-token": sSessionToken},
#         )
#         oResponse.raise_for_status()
#         logging.debug("WASDI shareWorkflow call completed with status %s", oResponse.status_code)
#         return oResponse.text


# @s_oMcpServer.tool()
# async def delete_snap_workflow_sharing(
#     sWorkflowId: str,
#     sUserId: str,
#     oContext: Context = None,
# ) -> str:
#     """
#     Removes workflow sharing for a user.
#     """
#     sSessionToken = getSessionToken(oContext)

#     if not sSessionToken:
#         raise ValueError("Missing x-session-token header")

#     if not sWorkflowId:
#         raise ValueError("Missing workflow id")

#     if not sUserId:
#         raise ValueError("Missing user id")

#     aoParams = {
#         "workflowId": sWorkflowId,
#         "userId": sUserId,
#     }

#     async with httpx.AsyncClient() as oClient:
#         oResponse = await oClient.delete(
#             f"{s_sWasdiApiUrl}/rest/workflows/share/delete",
#             params=aoParams,
#             headers={"x-session-token": sSessionToken},
#         )
#         oResponse.raise_for_status()
#         logging.debug("WASDI deleteWorkflowSharing call completed with status %s", oResponse.status_code)
#         return oResponse.text


# @s_oMcpServer.tool()
# async def get_snap_workflow_sharings(sWorkflowId: str, oContext: Context = None) -> str:
#     """
#     Retrieves all users with whom a workflow is shared.
#     """
#     sSessionToken = getSessionToken(oContext)

#     if not sSessionToken:
#         raise ValueError("Missing x-session-token header")

#     if not sWorkflowId:
#         raise ValueError("Missing workflow id")

#     aoParams = {"workflowId": sWorkflowId}

#     async with httpx.AsyncClient() as oClient:
#         oResponse = await oClient.get(
#             f"{s_sWasdiApiUrl}/rest/workflows/share/byworkflow",
#             params=aoParams,
#             headers={"x-session-token": sSessionToken},
#         )
#         oResponse.raise_for_status()
#         logging.debug("WASDI getWorkflowSharings call completed with status %s", oResponse.status_code)
#         return oResponse.text


# @s_oMcpServer.tool()
# async def download_snap_workflow(
#     sWorkflowId: str,
#     sTokenSessionId: str = None,
#     oContext: Context = None,
# ) -> str:
#     """
#     Downloads a workflow XML file.
#     """
#     sSessionToken = getSessionToken(oContext)

#     if not sSessionToken and not sTokenSessionId:
#         raise ValueError("Missing x-session-token header or token query param")

#     if not sWorkflowId:
#         raise ValueError("Missing workflow id")

#     aoParams = {
#         "workflowId": sWorkflowId,
#         "token": sTokenSessionId,
#     }
#     aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

#     oHeaders = {}
#     if sSessionToken:
#         oHeaders["x-session-token"] = sSessionToken

#     async with httpx.AsyncClient() as oClient:
#         oResponse = await oClient.get(
#             f"{s_sWasdiApiUrl}/rest/workflows/download",
#             params=aoParams,
#             headers=oHeaders,
#         )
#         oResponse.raise_for_status()
#         logging.debug("WASDI downloadWorkflow call completed with status %s", oResponse.status_code)
#         return oResponse.text
