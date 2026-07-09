import logging
import os
import json
import httpx
from mcp.server.fastmcp import FastMCP, Context
from utils.Utils import *


def register_developer_tools(mcp_server: FastMCP, wasdi_api_url: str):
    """Registers all tools for WASDI application development: upload, redeploy, update, and environment management."""

    @mcp_server.tool(name="upload_new_processor")
    async def upload_new_processor(
        sFilePath: str,
        sWorkspaceId: str,
        sName: str,
        sDescription: str = None,
        sType: str = None,
        oParamsSample: dict = None, 
        iPublic: int = 0,
        iTimeout: int = 3600,
        oContext: Context = None,
    ) -> str:
        """Uploads a local ZIP archive to deploy a brand-new application/processor in WASDI.

        CRITICAL for Developer Agents: The target ZIP must contain 'myProcessor.py' at its root. 
        It can optionally include 'pip.txt' for Python dependencies and 'packages.txt' for system libraries. 
        Never let the user bundle 'config.json' or 'params.json'.

        Args:
            sFilePath: Absolute local filesystem path to the processor's ZIP archive.
            sWorkspaceId: A valid workspace ID used to route backend client notifications.
            sName: The unique name for this application. Fails if the name is already taken.
            sDescription: A brief summary explaining the purpose and scope of the processor.
            sType: The runtime environment platform. Highly recommended values: 'PYTHON312_UBUNTU24' or 'PIP_ONESHOT'.
            oParamsSample: A structured dictionary representing a sample JSON payload execution contract.
            iPublic: Visibility flag. Use 1 to make it public to all WASDI users, or 0 for private (default 0).
            iTimeout: Execution safety timeout in seconds (default 3600). Use -1 for infinite runtime.
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

        # Convert the dictionary back to a JSON string for the WASDI API
        sParamsSampleStr = json.dumps(oParamsSample) if oParamsSample is not None else None

        aoParams = {
            "workspace": sWorkspaceId,
            "name": sName,
            "version": "1",
            "description": sDescription,
            "type": sType,
            "paramsSample": sParamsSampleStr,
            "public": iPublic,
            "timeout": iTimeout,
            "force": False,
        }
        aoParams = {k: v for k, v in aoParams.items() if v is not None}

        with open(sFilePath, "rb") as oFile:
            aoFiles = {"file": (os.path.basename(sFilePath), oFile, "application/zip")}
            async with httpx.AsyncClient() as oClient:
                oResponse = await oClient.post(
                    f"{wasdi_api_url}/rest/processors/uploadprocessor",
                    params=aoParams,
                    files=aoFiles,
                    headers={"x-session-token": sSessionToken},
                )
                oResponse.raise_for_status()
                logging.debug("WASDI upload_new_processor completed.")
                return oResponse.text

    @mcp_server.tool(name="redeploy_processor")
    async def redeploy_processor(sProcessorId: str, sWorkspaceId: str, oContext: Context = None) -> str:
        """Forces the WASDI infrastructure to rebuild and restart an existing application's Docker container.
        
        Use this tool when a developer updates an application environment, or if the application 
        reaches an inconsistent/broken deployment state and requires a clean container rebuild 
        without changing the primary source code.

        Args:
            sProcessorId: The unique system identifier of the application to rebuild.
            sWorkspaceId: A valid workspace ID used to stream build status alerts back to the user interface.
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
                f"{wasdi_api_url}/rest/processors/redeploy",
                params={"processorId": sProcessorId, "workspace": sWorkspaceId},
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI redeploy_processor completed.")
            return oResponse.text

    @mcp_server.tool(name="update_processor")
    async def update_processor(sProcessorId: str, oUpdatedProcessorVM: dict, oContext: Context = None) -> str:
        """Updates the core operational metadata settings for a specific WASDI application.

        CRITICAL Workflow Rule: The agent MUST call 'get_deployed_processors' or 'get_single_deployed_processor' 
        first to fetch the active 'DeployedProcessorViewModel' dictionary. Modify only the required 
        target fields inside that dictionary, then pass the full object back into this tool.

        Args:
            sProcessorId: The unique system identifier of the processor being modified.
            oUpdatedProcessorVM: The updated DeployedProcessorViewModel structure as a native JSON/dict object.
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
                f"{wasdi_api_url}/rest/processors/update",
                params={"processorId": sProcessorId},
                json=oUpdatedProcessorVM,
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI update_processor completed.")
            return oResponse.text

    @mcp_server.tool(name="update_processor_details")
    async def update_processor_details(sProcessorId: str, oUpdatedProcessorVM: dict, oContext: Context = None) -> str:
        """Updates the extended marketplace storefront configuration, categorization, and pricing for an application.

        CRITICAL Workflow Rule: The agent MUST execute 'get_market_place_app_detail' first to fetch the active 
        'AppDetailViewModel' dictionary. Modify only the required targeting properties (such as 'ondemandPrice', 
        'categories', 'showInStore', or 'friendlyName') inside that dictionary, then pass the full object back.

        Args:
            sProcessorId: The unique system identifier of the target application.
            oUpdatedProcessorVM: The modified AppDetailViewModel dictionary matching storefront rules.
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
                f"{wasdi_api_url}/rest/processors/updatedetails",
                params={"processorId": sProcessorId},
                json=oUpdatedProcessorVM,
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI update_processor_details completed.")
            return oResponse.text
    
    @mcp_server.tool(name="update_processor_files")
    async def update_processor_files(
        sFilePath: str,
        sProcessorId: str,
        sWorkspaceId: str,
        sInputFileName: str = None,
        oContext: Context = None,
    ) -> str:
        """Uploads updated source code files to a processor, automatically triggering a Docker image rebuild cycle.

        Use this tool when a developer wants to push bugs fixes, code edits, or dependency updates to their app.
        If the update involves multiple files or folder structures, the agent MUST package them into a single 
        ZIP archive before passing the path to this tool.

        Args:
            sFilePath: Local filesystem path to the updated script file or bundled ZIP archive.
            sProcessorId: The unique system identifier of the application receiving the update.
            sWorkspaceId: A valid workspace ID used to route build status alerts back to the user interface.
            sInputFileName: Optional filename override string to save as on the destination server layer.
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
        aoParams = {k: v for k, v in aoParams.items() if v is not None}

        with open(sFilePath, "rb") as oFile:
            aoFiles = {"file": (sInputFileName or os.path.basename(sFilePath), oFile, "application/octet-stream")}
            async with httpx.AsyncClient() as oClient:
                oResponse = await oClient.post(
                    f"{wasdi_api_url}/rest/processors/updatefiles",
                    params=aoParams,
                    files=aoFiles,
                    headers={"x-session-token": sSessionToken},
                )
                oResponse.raise_for_status()
                logging.debug("WASDI update_processor_files completed.")
                return oResponse.text

    @mcp_server.tool(name="download_processor")
    async def download_processor(sProcessorId: str, oContext: Context = None) -> str:
        """Downloads the complete application source code archive from the WASDI repository layer.
        
        Use this tool when you need to inspect an existing application's code files, verify 
        its dependency lists (pip.txt), or patch bugs within a processor you do not have locally.

        Args:
            sProcessorId: The unique system identifier of the application package to download.
        Returns:
            The raw binary content of the processor's source ZIP file converted into a hexadecimal string.
        """
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")
        if not sProcessorId:
            raise ValueError("Missing processor id")

        aoParams = {"token": sSessionToken, "processorId": sProcessorId}

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{wasdi_api_url}/rest/processors/downloadprocessor",
                params=aoParams,
            )
            oResponse.raise_for_status()
            logging.debug("WASDI download_processor completed.")
            return oResponse.content.hex()

    @mcp_server.tool(name="get_processor_build_logs")
    async def get_processor_build_logs(sProcessorId: str, oContext: Context = None) -> str:
        """Retrieves historical Docker container image build outputs compiled by the cluster builders.

        CRITICAL Diagnostic Rule: The agent MUST execute this tool immediately whenever a processor deployment, 
        redeploy, or code update operation fails or yields an internal state of 'ERROR'.
        
        Common failure paths to cross-reference in the output text:
          - Missing dependencies inside 'pip.txt'.
          - Version conflicts with 'gdal' and 'numpy' (WASDI strips these out unless explicitly pinned as 'numpy==VERSION').

        Args:
            sProcessorId: The unique system identifier of the target application.
        Returns:
            A JSON array of strings containing the Docker builder logs. The last entry contains the most recent log.
        """
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")
        if not sProcessorId:
            raise ValueError("Missing processor id")

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{wasdi_api_url}/rest/processors/logs/build",
                params={"processorId": sProcessorId},
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI get_processor_build_logs completed.")
            return oResponse.text
        
    @mcp_server.tool(name="get_application_packages_list")
    async def get_application_packages_list(sName: str, oContext: Context = None) -> str:
        """Returns a list of all currently installed third-party library packages inside an app's container.
        
        Use this tool to cross-reference exactly what dependencies are live in the production container 
        when troubleshooting runtime import errors or verifying a successful pip setup script cycle.

        Args:
            sName: The unique name identifier of the target application.
        """
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")
        if not sName:
            raise ValueError("Missing application name")

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{wasdi_api_url}/rest/packageManager/listPackages",
                params={"name": sName},
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI get_application_packages_list completed.")
            return oResponse.text

    @mcp_server.tool(name="get_application_environment_actions_list")
    async def get_application_environment_actions_list(sName: str, oContext: Context = None) -> str:
        """Returns the complete chronological history of system package modifications applied to an app's environment.
        
        Use this tool to audit installation attempts, tracking whether explicit manual updates or system 
        dependency injection commands succeeded or faulted.

        Args:
            sName: The unique name identifier of the target application.
        """
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")
        if not sName:
            raise ValueError("Missing application name")

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{wasdi_api_url}/rest/packageManager/environmentActions",
                params={"name": sName},
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI get_application_environment_actions_list completed.")
            return oResponse.text


    @mcp_server.tool(name="get_application_package_manager_version")
    async def get_application_package_manager_version(sName: str, oContext: Context = None) -> str:
        """Returns the foundational package manager core system version tracking the application's runtime.

        Args:
            sName: The unique name identifier of the target application.
        """
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")
        if not sName:
            raise ValueError("Missing application name")

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{wasdi_api_url}/rest/packageManager/managerVersion",
                params={"name": sName},
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI get_application_package_manager_version completed.")
            return oResponse.text


    @mcp_server.tool(name="update_application_environment_with_action")
    async def update_application_environment_with_action(
        sProcessorId: str,
        sWorkspaceId: str,
        sUpdateCommand: str = None,
        oContext: Context = None,
    ) -> str:
        """Forces an in-place hot-reload update of an app's environment, optionally running a specific shell command.
        
        Use this tool to inject hotfixes, apply configuration variables, or run isolated package 
        install routines directly into the container engine without executing a slow, complete 
        redeploy cycle.

        Args:
            sProcessorId: The unique system identifier of the application package.
            sWorkspaceId: A valid workspace ID used to stream execution status notifications.
            sUpdateCommand: Optional raw command line string to run inside the update environment sequence.
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
        aoParams = {k: v for k, v in aoParams.items() if v is not None}

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{wasdi_api_url}/rest/packageManager/environmentupdate",
                params=aoParams,
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI update_application_environment_with_action completed.")
            return oResponse.text


    @mcp_server.tool(name="reset_application_action_list")
    async def reset_application_action_list(sProcessorId: str, sWorkspaceId: str, oContext: Context = None) -> str:
        """Purges and resets the internal execution queue of pending environment modification actions for an app.
        
        Use this tool as an administrative override if an environment setup sequence hangs, 
        or if conflicting update triggers lock up the app package manager queue.

        Args:
            sProcessorId: The unique system identifier of the stuck application.
            sWorkspaceId: A valid workspace ID used to pipe operation telemetry alerts.
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

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{wasdi_api_url}/rest/packageManager/reset",
                params=aoParams,
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI reset_application_action_list completed.")
            return oResponse.text
