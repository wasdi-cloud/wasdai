import logging
import httpx
from mcp.server.fastmcp import FastMCP, Context
from utils.Utils import *

def register_catalog_tools(mcp_server: FastMCP, wasdi_api_url: str):
    """Registers all tools handling external EO catalog data searching, dataset ingestion, and local node file verification."""

    # =====================================================================
    # 1. EXTERNAL EARTH OBSERVATION (EO) CATALOG SEARCH
    # =====================================================================

    @mcp_server.tool(name="get_data_providers")
    async def get_data_providers(oContext: Context = None) -> str:
        """Returns a list of all active external Earth Observation (EO) data providers configured on WASDI.
        
        Use this tool to see which external imagery catalogs (e.g., COPERNICUS, CREODIAS2) are currently available.
        """
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{wasdi_api_url}/rest/search/providers",
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI get_data_providers completed.")
            return oResponse.text


    @mcp_server.tool(name="eo_data_search_get_count")
    async def eo_data_search_get_count(sQuery: str, oContext: Context = None) -> str:
        """Returns the total number of satellite imagery results that match an OpenSearch query string.
        
        Use this to quickly assess the size of a search before pulling the full paginated list.
        """
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")
        if not sQuery:
            raise ValueError("Missing query parameter")

        aoParams = {"query": sQuery, "providers": "AUTO"}

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{wasdi_api_url}/rest/search/query/count",
                params=aoParams,
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI eo_data_search_get_count completed.")
            return oResponse.text


    @mcp_server.tool(name="eo_data_paginated_search")
    async def eo_data_paginated_search(
        sQuery: str,
        sOffset: str = "0",
        sLimit: str = "10",
        sSortedBy: str = "startDate",
        sOrder: str = "desc",
        oContext: Context = None,
    ) -> str:
        """Executes a paginated OpenSearch query across external EO satellite catalogs.
        
        Use this tool to search for satellite orbits, instrument sensing modes, and track titles or footprint polygons.
        
        Args:
            sQuery: The target OpenSearch query string.
            sOffset: Result array starting index offset for pagination (default '0').
            sLimit: Maximum result items to return per page slice (default '10').
            sSortedBy: Metadatum tracking field to sort the results array by (default 'startDate').
            sOrder: Sorting direction constraint. Allowed: 'asc' or 'desc' (default 'desc').
        Returns:
            A JSON array of QueryResultViewModel containing provider identifiers, footprints, and titles.
        """
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")
        if not sQuery:
            raise ValueError("Missing query criteria string.")

        aoParams = {
            "providers": "AUTO", "query": sQuery, "offset": sOffset,
            "limit": sLimit, "sortedby": sSortedBy, "order": sOrder,
        }

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{wasdi_api_url}/rest/search/query",
                params=aoParams,
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI eo_data_paginated_search completed.")
            return oResponse.text


    # =====================================================================
    # 2. FILE INGESTION & DATA TRANSFER PIPELINES
    # =====================================================================

    @mcp_server.tool(name="import_product_in_wasdi")
    async def import_product_in_wasdi(oImageImportViewModel: dict, oContext: Context = None) -> str:
        """Triggers an asynchronous task to download and import an external satellite image into WASDI.

        Args:
            oImageImportViewModel: One of the structured JSON item dictionaries returned by the 
                                   'eo_data_paginated_search' tool. Do not modify its keys.
        Returns:
            An execution process workspace ID tracking string to monitor ingestion status.
        """
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")
        if not oImageImportViewModel:
            raise ValueError("Missing image import dictionary payload.")

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.post(
                f"{wasdi_api_url}/rest/filebuffer/download",
                json=oImageImportViewModel,
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI import_product_in_wasdi background pipeline initialized.")
            return oResponse.text


    @mcp_server.tool(name="ingest_existing_file_in_workspace")
    async def ingest_existing_file_in_workspace(sFileName: str, sWorkspaceId: str, sStyle: str = None, oContext: Context = None) -> str:
        """Verifies and adds an untracked local file present inside the workspace node storage bucket into the database.
        
        Use this tool if a custom app generates an intermediate output file on disk but fails to 
        register it automatically via standard catalog addition endpoints.
        """
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")
        if not sFileName or not sWorkspaceId:
            raise ValueError("Missing required parameter definitions: sFileName and sWorkspaceId are mandatory.")

        sNodeUrl = await getNodeUrlForWorkspace(sWorkspaceId, sSessionToken, wasdi_api_url)
        aoParams = {"file": sFileName, "workspace": sWorkspaceId, "style": sStyle}
        aoParams = {k: v for k, v in aoParams.items() if v is not None}

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{sNodeUrl}/rest/catalog/upload/ingestinws",
                params=aoParams,
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI ingest_existing_file_in_workspace completed.")
            return oResponse.text
        
    
    @mcp_server.tool(name="check_file_exists_in_node")
    async def check_file_exists_in_node(sFileName: str, sWorkspaceId: str, oContext: Context = None) -> str:
        """Checks the physical filesystem on the active cluster computing node to see if a file is present.
        
        Returns:
            A boolean string ('true' or 'false').
        """
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")

        sNodeUrl = await getNodeUrlForWorkspace(sWorkspaceId, sSessionToken, wasdi_api_url)
        aoParams = {"token": sSessionToken, "filename": sFileName, "workspace": sWorkspaceId}

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{sNodeUrl}/rest/catalog/fileOnNode",
                params=aoParams,
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI check_file_exists_in_node completed.")
            return oResponse.text


    @mcp_server.tool(name="get_product_properties")
    async def get_product_properties(sFileName: str, sWorkspaceId: str, bGetChecksum: bool = False, oContext: Context = None) -> str:
        """Returns storage metrics, file sizes, and modification timestamps for a product file.
        
        Args:
            sFileName: Target file name relative to the active workspace folder directory root.
            sWorkspaceId: Unique system workspace identification context.
            bGetChecksum: Set True to trigger hashing checksum validation (warning: heavy file IO operations).
        """
        sSessionToken = getSessionToken(oContext)
        if not sSessionToken:
            raise ValueError("Missing x-session-token header")

        aoParams = {"file": sFileName, "workspace": sWorkspaceId, "getchecksum": bGetChecksum}

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{wasdi_api_url}/rest/catalog/properties",
                params=aoParams,
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI get_product_properties completed.")
            return oResponse.text
        



# @s_oMcpServer.tool()
# async def eo_data_search_count_list(asQueries: list[str], oContext: Context = None) -> str:
#     """
#     Returns the total count of EO results for a list of queries.
#     This mirrors OpenSearchResource.countList.

#     Inputs:
#     asQueries: is a list of search query strings. Usually only one is used

#     Output:
#     The total count of EO results for the provided queries.
#     """
#     sSessionToken = getSessionToken(oContext)

#     if not sSessionToken:
#         raise ValueError("Missing x-session-token header")

#     if not asQueries:
#         raise ValueError("Missing query list")

#     aoParams = {"providers": "AUTO"}
#     aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

#     async with httpx.AsyncClient() as oClient:
#         oResponse = await oClient.post(
#             f"{s_sWasdiApiUrl}/rest/search/query/countlist",
#             params=aoParams,
#             json=asQueries,
#             headers={"x-session-token": sSessionToken},
#         )
#         oResponse.raise_for_status()
#         logging.debug("WASDI countList call completed with status %s", oResponse.status_code)
#         return oResponse.text

# @s_oMcpServer.tool()
# async def eo_data_search_list(asQueries: list[str], oContext: Context = None) -> str:
#     """
#     Executes EO searches for a list of queries.
#     This mirrors OpenSearchResource.searchList.

#     Inputs:
#     asQueries: is a list of search query strings. Usually only one is used

#     Output:
#     A list of JSON objects (QueryResultViewModel) containing the search results:
	
# 	String preview:  Encoded Image Preview
# 	String title: File Name
# 	String summary: Description. Supports a sort of std like: "Date: 2021-12-25T18:25:03.242Z, Instrument: SAR, Mode: IW, Satellite: S1A, Size: 0.95 GB" but is not mandatory
# 	String id: Provider Id
# 	String link: Link (or equivalent) to access the file
# 	String footprint: WKT Footprint
# 	provider: Data Provider that found this item
# 	Map<String, String> properties: Dictionary of additional properties	
# 	String volumeName: If this is accessible in a Volume, here we have the nameIf this is accessible in a Volume, here we have the name
# 	String volumePath: If this is accessible in a Volume, here we have the path in the volume
# 	String platform: Unique code of the platform/mission of this entry 
	
#     Basic info are:
#         .Title -> Name of the file
#         .Summary -> Description. Supports a sort of std like: "Date: 2021-12-25T18:25:03.242Z, Instrument: SAR, Mode: IW, Satellite: S1A, Size: 0.95 GB" but is not mandatory
#         .Id -> Provider unique id
#         .Link -> Link to download the file
#         .Footprint -> Bounding box in WKT ie POLYGON ((-7.087445 31.109682, -4.389633 31.524973, -4.062707 29.77639, -6.712266 29.357685, -7.087445 31.109682))
#                     Note: for POLYGON the convention is LON LAT, LON LAT...
#         .Provider -> Provider used to get this info.

#     Properties is a dictionary filled with all the properties supported by the data provider.
#     Can be seen with the "info" button in the client.
#             Some Commonly used, and shown in the client, are:
#                 ."date": reference Date
#                 ."instrument": used instrument 
#                 ."sensoroperationalmode": sensing mode
#                 ."size": image size as string
#                 ."relativeOrbit": relative orbit of the acquisition
#                 ."relativeorbitnumber": same of above, used by the client
#                 ."platformname": Platform Name

#     The libs searchs for a property called relativeOrbit        
#     """
#     sSessionToken = getSessionToken(oContext)

#     if not sSessionToken:
#         raise ValueError("Missing x-session-token header")

#     if not asQueries:
#         raise ValueError("Missing query list")

#     aoParams = {"providers": "AUTO"}
#     aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

#     async with httpx.AsyncClient() as oClient:
#         oResponse = await oClient.post(
#             f"{s_sWasdiApiUrl}/rest/search/querylist",
#             params=aoParams,
#             json=asQueries,
#             headers={"x-session-token": sSessionToken},
#         )
#         oResponse.raise_for_status()
#         logging.debug("WASDI searchList call completed with status %s", oResponse.status_code)
#         return oResponse.text


# @s_oMcpServer.tool()
# async def check_download_product_availability_by_name(sFileName: str, sWorkspaceId: str, oContext: Context = None) -> str:
#     """
#     Checks if a file is available for download. 
#     The difference with check_file_exists_in_node  is that this API checks if the file is present on the node and is declared as a product in the workspace

#     Inputs:
#     sFileName: is the name of the file to check (always relative to workspace path, so usually just the file name)
#     sWorkspaceId: is the unique id of the workspace in which to check the file
#     sProcessObjId: is an optional process workspace id. If provided, the check will be associated with this process workspace. Usually keep it null
#     sVolumePath: is an optional string indicating the volume path. If not provided, the default is null. This is used to check if the file is available in a specific volume path, if the node has multiple volume paths.

#     Output:
#     a Json with values:
#     StringValue: ignored in this API
#     IntValue: ignored in this API
#     BoolValue: true if the file is available, false if the file is not
#     DoubleValue: ignored in this API

#     """
#     sSessionToken = getSessionToken(oContext)

#     if not sSessionToken:
#         raise ValueError("Missing x-session-token header")

#     if not sFileName:
#         raise ValueError("Missing file name")

#     if not sWorkspaceId:
#         raise ValueError("Missing workspace id")

#     aoParams = {"token": sSessionToken, "filename": sFileName, "workspace": sWorkspaceId}
#     aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

#     sNodeUrl = await getNodeUrlForWorkspace(sWorkspaceId, sSessionToken)

#     async with httpx.AsyncClient() as oClient:
#         oResponse = await oClient.get(
#             f"{sNodeUrl}/rest/catalog/checkdownloadavaialibitybyname",
#             params=aoParams,
#             headers={"x-session-token": sSessionToken},
#         )
#         oResponse.raise_for_status()
#         logging.debug("WASDI checkDownloadEntryAvailabilityByName call completed with status %s", oResponse.status_code)
#         return oResponse.text


# @s_oMcpServer.tool()
# async def publish_product_band_in_wms(
#     sFileUrl: str,
#     sWorkspaceId: str,
#     sBand: str,
#     sStyle: str = None,
#     oContext: Context = None,
# ) -> str:
#     """
#     Publishes a band on GeoServer for a file in a workspace.
#     This mirrors FileBufferResource.publishBand.
#     """
#     sSessionToken = getSessionToken(oContext)

#     if not sSessionToken:
#         raise ValueError("Missing x-session-token header")

#     if not sFileUrl:
#         raise ValueError("Missing file url")

#     if not sWorkspaceId:
#         raise ValueError("Missing workspace id")

#     if not sBand:
#         raise ValueError("Missing band")

#     aoParams = {
#         "fileUrl": sFileUrl,
#         "workspace": sWorkspaceId,
#         "band": sBand,
#         "style": sStyle
#     }
#     aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

#     async with httpx.AsyncClient() as oClient:
#         oResponse = await oClient.get(
#             f"{s_sWasdiApiUrl}/rest/filebuffer/publishband",
#             params=aoParams,
#             headers={"x-session-token": sSessionToken},
#         )
#         oResponse.raise_for_status()
#         logging.debug("WASDI publishBand call completed with status %s", oResponse.status_code)
#         return oResponse.text
