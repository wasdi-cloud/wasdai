import logging
import httpx
import json
import urllib.parse
from mcp.server.fastmcp import FastMCP, Context
from utils.Utils import *

def register_application_tools(mcp_server: FastMCP, wasdi_api_url: str):
    """Registers all tools related to WASDI apps, process execution, task monitoring, and processing operations."""

    @mcp_server.tool(name="get_deployed_processors")
    async def get_deployed_processors(oContext: Context = None) -> str:
        """Returns all deployed processors/apps visible (owned, public, or shared) to the current user.
        
        Use this tool to discover app versions, custom timeout boundaries, unique IDs, and sample parameter sets.
        """
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{wasdi_api_url}/rest/processors/getdeployed",
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI get_deployed_processors completed.")
            return oResponse.text


    @mcp_server.tool(name="get_single_deployed_processor")
    async def get_single_deployed_processor(sProcessorId: str = None, sProcessorName: str = None, oContext: Context = None) -> str:
        """Retrieves targeted specification details for a single application filtered by ID or unique name.
        
        Provide at least one of the lookup filters. Use this tool to extract 'paramsSample' payload structures.
        """
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")
        if not sProcessorId and not sProcessorName:
            raise ValueError("Missing look-up parameter: define either sProcessorId or sProcessorName.")

        aoParams = {"processorId": sProcessorId, "name": sProcessorName}
        aoParams = {k: v for k, v in aoParams.items() if v is not None}

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{wasdi_api_url}/rest/processors/getprocessor",
                params=aoParams,
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI get_single_deployed_processor completed.")
            return oResponse.text


    @mcp_server.tool(name="get_market_place_app_list")
    async def get_market_place_app_list(oFilters: dict = None, oContext: Context = None) -> str:
        """Queries and returns available user-facing applications published inside the WASDI App Store Marketplace.
        
        Args:
            oFilters: Optional search parameters dictionary. Supports fields:
                      'name' (string search context), 'page' (integer index starting at 0),
                      'itemsPerPage' (default 12), 'orderBy' ('name', 'lastUpdate').
        """
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.post(
                f"{wasdi_api_url}/rest/processors/getmarketlist",
                json=oFilters or {},
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI get_market_place_app_list completed.")
            return oResponse.text


    @mcp_server.tool(name="get_processor_ui")
    async def get_processor_ui(sProcessorName: str, oContext: Context = None) -> str:
        """Returns the specialized UI layout schema dictionary mapping inputs to standard user control parameters."""
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")
        if not sProcessorName:
            raise ValueError("Missing processor name identifier.")

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{wasdi_api_url}/rest/processors/ui",
                params={"name": sProcessorName},
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI get_processor_ui completed.")
            return oResponse.text
    
    @mcp_server.tool(name="run_processor")
    async def run_processor(sProcessorName: str, sExecutionWorkspaceId: str, oProcessorsInput: dict, oContext: Context = None) -> str:
        """Triggers asynchronous processing job runs for an application within a designated workspace environment.
        
        Args:
            sProcessorName: Unique engine name code of the app.
            sExecutionWorkspaceId: Workspace ID environment target for data scope.
            oProcessorsInput: The structured operational execution configurations dictionary matching input criteria specs.
        """
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")
        if not sProcessorName or not sExecutionWorkspaceId or oProcessorsInput is None:
            raise ValueError("Missing required arguments for tool processing execution pipeline invocation.")

        # Convert the native object safely to string and apply local URL encoding
        sJsonString = json.dumps(oProcessorsInput)
        sEncodedJson = urllib.parse.quote(sJsonString)

        aoParams = {"name": sProcessorName, "workspace": sExecutionWorkspaceId, "parent": "", "notify": False}

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.post(
                f"{wasdi_api_url}/rest/processors/run",
                params=aoParams,
                content=sEncodedJson,
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI run_processor invocation successful.")
            return oResponse.text

    @mcp_server.tool(name="get_processor_help")
    async def get_processor_help(sProcessorName: str, oContext: Context = None) -> str:
        """Returns the Markdown user manual, parameter guides, and documentation for a processor.
        
        CRITICAL: The agent MUST execute this tool before calling 'run_processor' if the exact 
        JSON structure or acceptable value bounds for the 'oProcessorsInput' parameter are unknown.
        Use this to discover what specific bounding box, date formats, or file dependencies the app requires.
        """
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")
        if not sProcessorName:
            raise ValueError("Missing processor name")

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{wasdi_api_url}/rest/processors/help",
                params={"name": sProcessorName},
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI get_processor_help completed.")
            return oResponse.text

    @mcp_server.tool(name="mosaic")
    async def mosaic(sDestinationProductName: str, sWorkspaceId: str, oMosaicSetting: dict, oContext: Context = None) -> str:
        """Triggers a native asynchronous array image mosaicing task combining multi-source dataset layers.
        
        Args:
            sDestinationProductName: Result file naming string to yield.
            sWorkspaceId: Active workspace holding target image inputs.
            oMosaicSetting: Custom operation mapping details matching the platform MosaicSetting layout.
        """
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")

        sNodeUrl = await getNodeUrlForWorkspace(sWorkspaceId, sSessionToken, wasdi_api_url)
        aoParams = {"name": sDestinationProductName, "workspace": sWorkspaceId}

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.post(
                f"{sNodeUrl}/rest/processing/mosaic",
                json=oMosaicSetting,
                params=aoParams,
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            return oResponse.text


    @mcp_server.tool(name="regrid")
    async def regrid(sDestinationProductName: str, sWorkspaceId: str, oRegridSetting: dict, oContext: Context = None) -> str:
        """Triggers an asynchronous regridding transformation process modifying resolution maps layout grids."""
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")

        sNodeUrl = await getNodeUrlForWorkspace(sWorkspaceId, sSessionToken, wasdi_api_url)
        aoParams = {"name": sDestinationProductName, "workspace": sWorkspaceId}

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.post(
                f"{sNodeUrl}/rest/processing/regrid",
                json=oRegridSetting,
                params=aoParams,
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            return oResponse.text


    @mcp_server.tool()
    async def multiSubset(
        sSourceProductName: str,
        sDestinationProductName: str,
        sWorkspaceId: str,
        sMultiSubsetSettingJson: str,
        oContext: Context = None,
    ) -> str:
        """
        Triggers a multi-subset operation on products.
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
        }
        aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.post(
                f"{wasdi_api_url}/rest/processing/multisubset",
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


    @mcp_server.tool(name="get_processes_by_workspace")
    async def get_processes_by_workspace(
        sWorkspaceId: str,
        sStatus: str = None,
        sOperationType: str = None,
        sNamePattern: str = None,
        iStartIndex: int = 0,
        iEndIndex: int = 20,
        oContext: Context = None,
    ) -> str:
        """Returns historical processing jobs (tasks executed or currently running) scoped inside a workspace.
        
        Use this tool to track pending schedules, find running tasks, or find jobs stuck in status ERROR.
        Pagination fields are filled by default.
        """
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")

        sNodeUrl = await getNodeUrlForWorkspace(sWorkspaceId, sSessionToken, wasdi_api_url)
        aoParams = {
            "workspace": sWorkspaceId, "status": sStatus, "operationType": sOperationType,
            "namePattern": sNamePattern, "startindex": iStartIndex, "endindex": iEndIndex,
        }
        aoParams = {k: v for k, v in aoParams.items() if v is not None}

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{sNodeUrl}/rest/process/byws",
                params=aoParams,
                headers={"x-session-token": sSessionToken}
            )
            oResponse.raise_for_status()
            return oResponse.text


    @mcp_server.tool(name="get_process_status_by_id")
    async def get_process_status_by_id(sProcessObjId: str, oContext: Context = None) -> str:
        """Returns the quick status keyword string for a targeted background process task.
        
        Fast alternative to get full details when polling task updates.
        Returns: One of CREATED, RUNNING, WAITING, READY, DONE, ERROR, STOPPED.
        """
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")

        # Fallback tracking safely handled inside dynamic target resolution routine if workspace context isn't exposed
        async with httpx.AsyncClient() as oClient:
            # Check main router status endpoint
            oResponse = await oClient.get(
                f"{wasdi_api_url}/rest/process/byid",
                params={"procws": sProcessObjId},
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            oData = json.loads(oResponse.text)
            sWorkspaceId = oData.get("workspaceId") or oData.get("workspace")
            sNodeUrl = await getNodeUrlForWorkspace(sWorkspaceId, sSessionToken, wasdi_api_url)
            
            oStatusResponse = await oClient.get(
                f"{sNodeUrl}/rest/process/getstatusbyid",
                params={"procws": sProcessObjId},
                headers={"x-session-token": sSessionToken},
            )
            oStatusResponse.raise_for_status()
            return oStatusResponse.text


    @mcp_server.tool(name="get_processor_logs")
    async def get_processor_logs(sProcessWorkspaceId: str, iStartRow: int = 0, iEndRow: int = 100, oContext: Context = None) -> str:
        """Fetches the logging stdout/stderr streams matrix from the compute cluster for active or terminated tasks.
        
        Crucial tool for debugging workflow processing failures or inspecting step tracking markers.
        """
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{wasdi_api_url}/rest/process/byid",
                params={"procws": sProcessWorkspaceId},
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            sWorkspaceId = json.loads(oResponse.text).get("workspaceId")
            sNodeUrl = await getNodeUrlForWorkspace(sWorkspaceId, sSessionToken, wasdi_api_url)

            aoParams = {"processworkspace": sProcessWorkspaceId, "startrow": iStartRow, "endrow": iEndRow}
            oLogResponse = await oClient.get(
                f"{sNodeUrl}/rest/processors/logs/list",
                params=aoParams,
                headers={"x-session-token": sSessionToken},
            )
            oLogResponse.raise_for_status()
            return oLogResponse.text


    @mcp_server.tool(name="kill_process_in_workspace")
    async def kill_process_in_workspace(sProcessObjId: str, oContext: Context = None) -> str:
        """Terminates a running task execution process tree inside the target active node compute cluster."""
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{wasdi_api_url}/rest/process/byid",
                params={"procws": sProcessObjId},
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            sWorkspaceId = json.loads(oResponse.text).get("workspaceId")
            sNodeUrl = await getNodeUrlForWorkspace(sWorkspaceId, sSessionToken, wasdi_api_url)

            oKillResponse = await oClient.get(
                f"{sNodeUrl}/rest/process/delete",
                params={"procws": sProcessObjId, "treeKill": "true"},
                headers={"x-session-token": sSessionToken}
            )
            oKillResponse.raise_for_status()
            return oKillResponse.text
            
    @mcp_server.tool()
    async def get_processor_logs(sProcessWorkspaceId: str, iStartRow: int = None, iEndRow: int = None, oContext: Context = None) -> str:
        """
        Returns a paginated list of log rows for a processor workspace. 
        The agent can use this information to get the logs of a processor, for example to check the output of a process or to debug any issues that may have occurred during the execution of the process.

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
        sNodeUrl = await getNodeUrlForWorkspace(sProcessWorkspaceId, sSessionToken, None)

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{sNodeUrl}/rest/processors/logs/list",
                params=aoParams,
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI getLogs call completed with status %s", oResponse.status_code)
            return oResponse.text
        
        @mcp_server.tool(name="get_last_processes_by_workspace")
        async def get_last_processes_by_workspace(sWorkspaceId: str, oContext: Context = None) -> str:
            """Returns the five most recent processes executed inside a workspace.
            
            Use this to quickly check the user's recent history or see what task they were just working on.
            """
            sSessionToken = getSessionToken(oContext)
            if not sSessionToken:
                raise ValueError("Missing x-session-token header")

            sNodeUrl = await getNodeUrlForWorkspace(sWorkspaceId, sSessionToken, wasdi_api_url)
            async with httpx.AsyncClient() as oClient:
                oResponse = await oClient.get(
                    f"{sNodeUrl}/rest/process/lastbyws",
                    params={"workspace": sWorkspaceId},
                    headers={"x-session-token": sSessionToken}
                )
                oResponse.raise_for_status()
                return oResponse.text
            
    @mcp_server.tool(name="get_process_payload")
    async def get_process_payload(sProcessObjId: str, oContext: Context = None) -> str:
        """Returns the final text result payload (usually a structured JSON string) created by a successful process task.
        
        Use this tool to extract output metrics, validation data, or filenames produced by a completed run.
        """
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{wasdi_api_url}/rest/process/byid",
                params={"procws": sProcessObjId},
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            sWorkspaceId = json.loads(oResponse.text).get("workspaceId")
            sNodeUrl = await getNodeUrlForWorkspace(sWorkspaceId, sSessionToken, wasdi_api_url)

            oPayloadResponse = await oClient.get(
                f"{sNodeUrl}/rest/process/payload",
                params={"procws": sProcessObjId},
                headers={"x-session-token": sSessionToken}
            )
            oPayloadResponse.raise_for_status()
            return oPayloadResponse.text

    @mcp_server.tool()
    async def share_processor(sProcessorId: str, sUserId: str, sRights: str = None, oContext: Context = None) -> str:
        """
        Shares a processor with a user.

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
                f"{wasdi_api_url}/rest/processors/share/add",
                params=aoParams,
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI shareProcessor call completed with status %s", oResponse.status_code)
            return oResponse.text


    @mcp_server.tool()
    async def get_market_place_app_detail(sProcessorName: str, oContext: Context = None) -> str:
        """
        Returns the detailed marketplace information for a processor. 
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
                f"{wasdi_api_url}/rest/processors/getmarketdetail",
                params={"processorname": sProcessorName},
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI getMarketPlaceAppDetail call completed with status %s", oResponse.status_code)
            return oResponse.text


    # @mcp_server.tool()
    # async def get_credits_for_run_paid_processor(sProcessorId: str, sEncodedJson: str, oContext: Context = None) -> str:
    #     """
    #     Returns the estimated credits needed for a processor run, if the processors is a paid one based on credits.

    #     Inputs:
    #     sProcessorId: is the unique id of the processor to run.
    #     sEncodedJson: is the JSON string containing the parameters for the processor.

    #     Output:
    #     a number that is the total credits needed.
    #     """
    #     sSessionToken = getSessionToken(oContext)

    #     if not sSessionToken:
    #         raise ValueError("Missing x-session-token header")

    #     if not sProcessorId:
    #         raise ValueError("Missing processor id")

    #     if sEncodedJson is None:
    #         raise ValueError("Missing encoded json payload")

    #     async with httpx.AsyncClient() as oClient:
    #         oResponse = await oClient.post(
    #             f"{wasdi_api_url}/rest/processors/getcredits",
    #             params={"processorId": sProcessorId},
    #             content=sEncodedJson,
    #             headers={"x-session-token": sSessionToken},
    #         )
    #         oResponse.raise_for_status()
    #         logging.debug("WASDI getCreditsForRun call completed with status %s", oResponse.status_code)
    #         return oResponse.text

    # @mcp_server.tool()
    # async def get_summary_of_running_processes_for_workspace(sWorkspaceId: str = None, oContext: Context = None) -> str:
    #     """
    #     Returns process summary counts for a workspace and user. The agent can use this API to get a quick overview of the processes that are 
    #     running in a workspace, for example to check how many processes are currently running and how many are waiting, and so on. 

    #     Input
    #     sWorkspaceId: is the unique id of the workspace for which we want to get the summary of running processes.

    #     Output:
    #     int userProcessWaiting: number of processes that are waiting for the user who is calling this API. This is the number of processes that are in WAITING status and that have been started by the user who is calling this API. The agent can use this information to check if the user has any processes that are waiting to be executed, for example to suggest to the user to start a new process or to check if there are any processes that are waiting for the user to take action.
    #     int userProcessRunning: number of processes that are currently running for the user who is calling this API. This is the number of processes that are in RUNNING status and that have been started by the user who is calling this API. The agent can use this information to check if the user has any processes that are currently running, for example to suggest to the user to wait for the processes to complete before starting a new one.	
    #     int allProcessWaiting: number of processes that are waiting in the workspace. This is the number of processes that are in WAITING status, regardless of the user who started them. The agent can use this information to check if there are any processes that are waiting to be executed in the workspace.
    #     int allProcessRunning: number of processes that are currently running in the workspace. This is the number of processes that are in RUNNING status, regardless of the user who started them. The agent can use this information to check if there are any processes that are currently running in the workspace.    

    #     """
    #     sSessionToken = getSessionToken(oContext)

    #     if not sSessionToken:
    #         raise ValueError("Missing x-session-token header")

    #     aoParams = {}
    #     if sWorkspaceId:
    #         aoParams["workspace"] = sWorkspaceId

    #     # Resolve the node URL for this workspace
    #     sNodeUrl = await getNodeUrlForWorkspace(sWorkspaceId, sSessionToken)

    #     async with httpx.AsyncClient() as oClient:
    #         oResponse = await oClient.get(
    #             f"{sNodeUrl}/rest/process/summary",
    #             params=aoParams,
    #             headers={"x-session-token": sSessionToken}
    #         )
    #         oResponse.raise_for_status()
    #         logging.debug("WASDI getSummary call completed with status %s", oResponse.status_code)
    #         return oResponse.text        



    # @mcp_server.tool()
    # async def get_process_by_id(sProcessObjId: str,  oContext: Context = None) -> str:
    #     """
    #     Returns a process workspace view model by id.
    #     The agent can use this to get the details of a process workspace so of any operation executed in WASDI.

    #     Inputs:
    #     sProcessObjId: is the unique id of the process workspace

    #     Output:
    #     return an empty Process Workspace View Model in case of errors or a JSON object with the following properties:
    #     String productName: name of the product target of this process. The name is historical, but can represent in reality a name of a product, or of an application or of a SNAP workflow
    #     String operationType: type of the operation performed by this process. Types are INGEST, DOWNLOAD, SHARE, PUBLISHBAND, GRAPH, DEPLOYPROCESSOR, RUNPROCESSOR, MOSAIC, MULTISUBSET, REGRID, DELETEPROCESSOR, INFO, REDEPLOYPROCESSOR, LIBRARYUPDATE, ENVIRONMENTUPDATE, KILLPROCESSTREE,
    #     String operationSubType: subtype of the operation performed by this process. Each operation can have a subtype in theory. In reality now is used for DOWNALOD operations: subtype is the data provider of the data that is downloaded, for example COPERNICUS, CREODIAS2, LSA etc
    #     String operationDate: date of the operation creation
    #     String operationStartDate: start date of the operation
    #     String operationEndDate: end date of the operation
    #     String lastChangeDate: date of the last status change
    #     String userId: id of the user who started the operation
    #     String fileSize: size of the file ie for a download operation
    #     String status: status of the process. Status can be CREATED, RUNNING, WAITING, READY, DONE, ERROR, STOPPED. The process is created CREATED. Then is the scheduler that triggers it in start. Applications moves in WAITING when the user calls waitProcess from the lib. When the process is done, it become READY and the scheduler will move in RUNNING again when there is a slot
    #     int progressPerc: progress percentage of the process
    #     String processObjId: id of the process object
    #     int pid: id of the process in the operating system of the node where it is executed
    #     String payload: json output created by the process when id done. The content of the payload is defined by the process itself, but it can contain useful information for the user
    #     String workspaceId: id of the workspace where the process is executed

    #     """
    #     sSessionToken = getSessionToken(oContext)

    #     if not sSessionToken:
    #         raise ValueError("Missing x-session-token header")

    #     if not sProcessObjId:
    #         raise ValueError("Missing process id")
        
    #     sWorkspaceId = get_workspace_id_for_process_workspace(sProcessObjId, sSessionToken)
    #     sNodeUrl = await get_node_url_for_process_workspace(sProcessObjId, sSessionToken, sWorkspaceId)

    #     async with httpx.AsyncClient() as oClient:
    #         oResponse = await oClient.get(
    #             f"{sNodeUrl}/rest/process/byid",
    #             params={"procws": sProcessObjId},
    #             headers={"x-session-token": sSessionToken}
    #         )
    #         oResponse.raise_for_status()
    #         logging.debug("WASDI getProcessById call completed with status %s", oResponse.status_code)
    #         return oResponse.text


    # @mcp_server.tool()
    # async def get_status_processes_by_id(asProcessesWorkspaceId: list[str], oContext: Context = None) -> str:
    #     """
    #     Returns the status of multiple process workspaces in a single call. 
    #     Is faster than calling get_process_status_by_id for each process workspace id. 
    #     The agent can use this API to get the status of multiple processes in a single call.

    #     Inputs:
    #     asProcessesWorkspaceId: is a list of unique ids (strings) of the process workspaces

    #     Output:
    #     An array of Strings: one for each asProcessesWorkspaceId in input, with a value describing the status of the process workspace. Status can be CREATED, RUNNING, WAITING, READY, DONE, ERROR, STOPPED. 
    #     the call returns an empty array in case of errors or 
    #     """
    #     sSessionToken = getSessionToken(oContext)

    #     if not sSessionToken:
    #         raise ValueError("Missing x-session-token header")

    #     if not asProcessesWorkspaceId:
    #         raise ValueError("Missing process id list")
        
    #     sNodeUrl = await get_node_url_for_process_workspace(asProcessesWorkspaceId[0], sSessionToken, None)    

    #     async with httpx.AsyncClient() as oClient:
    #         oResponse = await oClient.post(
    #             f"{sNodeUrl}/rest/process/statusbyid",
    #             json=asProcessesWorkspaceId,
    #             headers={"x-session-token": sSessionToken}
    #         )
    #         oResponse.raise_for_status()
    #         logging.debug("WASDI getStatusProcessesById call completed with status %s", oResponse.status_code)
    #         return oResponse.text


    # @mcp_server.tool()
    # async def get_process_log_count(sProcessWorkspaceId: str, oContext: Context = None) -> str:
    #     """
    #     Returns the count of log rows for a processor executed in WASDI.

    #     Inputs:
    #     sProcessWorkspaceId: is the unique id of the process workspace.

    #     Output:
    #     An integer with the count of log rows for the processor.
    #     """
    #     sSessionToken = getSessionToken(oContext)

    #     if not sSessionToken:
    #         raise ValueError("Missing x-session-token header")

    #     if not sProcessWorkspaceId:
    #         raise ValueError("Missing process workspace id")
        
    #     # Resolve the node URL for this workspace
    #     sNodeUrl = await get_node_url_for_process_workspace(sProcessWorkspaceId, sSessionToken, None)

    #     async with httpx.AsyncClient() as oClient:
    #         oResponse = await oClient.get(
    #             f"{sNodeUrl}/rest/processors/logs/count",
    #             params={"processworkspace": sProcessWorkspaceId},
    #             headers={"x-session-token": sSessionToken},
    #         )
    #         oResponse.raise_for_status()
    #         logging.debug("WASDI countLogs call completed with status %s", oResponse.status_code)
    #         return oResponse.text


    # @mcp_server.tool()
    # async def get_users_can_access_processor(sProcessorId: str, oContext: Context = None) -> str:
    #     """
    #     Returns the list of users who can access a processor.

    #     Inputs:
    #     sProcessorId: is the unique id of the processor.

    #     Output:
    #     A list of ProcessorSharingViewModel JSON objects, each with the following properties:    
    #     userId: unique id of the user
    #     permission: permission granted to the user for the processor. Can be "read" or "write".
    #     """
    #     sSessionToken = getSessionToken(oContext)

    #     if not sSessionToken:
    #         raise ValueError("Missing x-session-token header")

    #     if not sProcessorId:
    #         raise ValueError("Missing processor id")

    #     async with httpx.AsyncClient() as oClient:
    #         oResponse = await oClient.get(
    #             f"{wasdi_api_url}/rest/processors/share/byprocessor",
    #             params={"processorId": sProcessorId},
    #             headers={"x-session-token": sSessionToken},
    #         )
    #         oResponse.raise_for_status()
    #         logging.debug("WASDI get_users_can_access_processor call completed with status %s", oResponse.status_code)
    #         return oResponse.text

