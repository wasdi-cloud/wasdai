import logging
import httpx
from mcp.server.fastmcp import FastMCP, Context
from utils.Utils import *

def register_workspace_tools(mcp_server: FastMCP, wasdi_api_url: str):
    """Registers all tools related to WASDI workspaces and product management to the FastMCP instance."""
    
    @mcp_server.tool()
    async def get_workspaces_by_user(oContext: Context = None) -> str:
        """
        Returns a list of workspaces accessible by the current user.
        
        The returned workspaces can be owned by the user, shared with them, or public.
        Use this tool to find available workspace IDs, names, or storage sizes.
        
        Returns:
            A JSON array of WorkspaceListInfoViewModel containing workspaceId, workspaceName, ownerUserId, 
            storageSize (bytes), isPublic, and nodeCode.        
        """
        sSessionToken = getSessionToken(oContext)

        if not sSessionToken:
            raise ValueError("Missing x-session-token header")
        
        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{wasdi_api_url}/rest/ws/byuser",
                headers={"x-session-token": sSessionToken}
            )
            oResponse.raise_for_status()
            logging.debug("WASDI get_workspaces call completed with status %s", oResponse.status_code)
            return oResponse.text


    @mcp_server.tool()
    async def get_workspace_details(sWorkspaceId: str, oContext: Context = None) -> str:
        """Returns deep metadata for a specific workspace using its unique ID.
        
        Use this tool when you need specialized node URLs, precise creation/edit dates, cloud providers, 
        or user permissions for a specific workspace.
        
        Args:
            sWorkspaceId: The unique ID string of the target workspace.
        """
        sSessionToken = getSessionToken(oContext)

        if not sSessionToken:
            raise ValueError("Missing x-session-token header")

        if not sWorkspaceId:
            raise ValueError("Missing workspace id")

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{wasdi_api_url}/rest/ws/getws",
                params={"workspace": sWorkspaceId},
                headers={"x-session-token": sSessionToken}
            )
            oResponse.raise_for_status()
            logging.debug("WASDI get_workspace_details call completed with status %s", oResponse.status_code)
            return oResponse.text


    @mcp_server.tool()
    async def get_workspace_name_by_id(sWorkspaceId: str, oContext: Context = None) -> str:
        """Resolves a unique workspace alphanumeric ID into its human-readable workspace name.
        
        Use this tool when system payloads or process tasks provide an ID string, and you 
        need to present the human-readable workspace name back to the user.
        """
        sSessionToken = getSessionToken(oContext)

        if not sSessionToken:
            raise ValueError("Missing x-session-token header")

        if not sWorkspaceId:
            raise ValueError("Missing workspace id")

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{wasdi_api_url}/rest/ws/wsnamebyid",
                params={"workspace": sWorkspaceId},
                headers={"x-session-token": sSessionToken}
            )
            oResponse.raise_for_status()
            logging.debug("WASDI get_workspace_name_by_id call completed with status %s", oResponse.status_code)
            return oResponse.text


    @mcp_server.tool()
    async def create_new_workspace(sName: str = None, oContext: Context = None) -> str:
        """
        Creates a brand new workspace for the current user.
        
        Workspace names are unique per user. If the requested name already exists, 
        WASDI will append incrementing numbers automatically (e.g., 'MyWorkspace(1)').
        
        Args:
            sName: The intended name for the new workspace.
        Returns:
            A JSON response object where 'StringValue' contains the unique ID of the newly created workspace.
        """
        sSessionToken = getSessionToken(oContext)

        if not sSessionToken:
            raise ValueError("Missing x-session-token header")

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{wasdi_api_url}/rest/ws/create",
                params={"name": sName, "node": ""},
                headers={"x-session-token": sSessionToken}
            )
            oResponse.raise_for_status()
            logging.debug("WASDI createWorkspace call completed with status %s", oResponse.status_code)
            return oResponse.text


    @mcp_server.tool()
    async def share_workspace_with_user(sWorkspaceId: str, sDestinationUserId: str, sRights: str = None, oContext: Context = None) -> str:
        """Shares a specified workspace with another user or updates their access permissions.
        
        Args:
            sWorkspaceId: Unique ID of the workspace to share.
            sDestinationUserId: Target user ID to receive access.
            sRights: Access level to grant. Allowed values: 'read' (read-only) or 'write' (read/write). Defaults to 'read'.
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
                f"{wasdi_api_url}/rest/ws/share/add",
                params={"workspace": sWorkspaceId, "userId": sDestinationUserId, "rights": sRights},
                headers={"x-session-token": sSessionToken}
            )
            oResponse.raise_for_status()
            logging.debug("WASDI shareWorkspace call completed with status %s", oResponse.status_code)
            return oResponse.text


    # @mcp_server.tool()
    # async def get_users_with_access_to_workspace(sWorkspaceId: str, oContext: Context = None) -> str:
    #     """
    #     Returns the list of users that have access to a workspace.

    #     Input
    #     sWorkspaceId: is the unique id of the workspace for which we want to get the list of users that have access to it.

    #     Output
    #     an array JSON object with:
    #         workspaceId: the unique id of the workspace
    #         userId: the user id of the user that has access to the workspace
    #         ownerId: the user id of the owner of the workspace
    #         permissions: the level of access that the user has on the workspace, it can be "read" for read-only access or "write" for read and write access

    #     the call returns an empty array in case of errors
    #     """
    #     sSessionToken = getSessionToken(oContext)

    #     if not sSessionToken:
    #         raise ValueError("Missing x-session-token header")

    #     if not sWorkspaceId:
    #         raise ValueError("Missing workspace id")

    #     async with httpx.AsyncClient() as oClient:
    #         oResponse = await oClient.get(
    #             f"{wasdi_api_url}/rest/ws/share/byworkspace",
    #             params={"workspace": sWorkspaceId},
    #             headers={"x-session-token": sSessionToken}
    #         )
    #         oResponse.raise_for_status()
    #         logging.debug("WASDI getEnabledUsersSharedWorksace call completed with status %s", oResponse.status_code)
    #         return oResponse.text


    @mcp_server.tool()
    async def get_names_of_products_by_workspace(sWorkspaceId: str, oContext: Context = None) -> str:
        """Returns the quick list of file names (products) contained inside a workspace.
        
        This is the fastest API to inspect what files are present inside a workspace or to verify if a file exists.
        
        Args:
            sWorkspaceId: The unique ID of the target workspace.
        Returns:
            A JSON array of strings containing the files names with extensions (e.g., ['data.tif', 'report.csv']).
        """
        sSessionToken = getSessionToken(oContext)

        if not sSessionToken:
            raise ValueError("Missing x-session-token header")

        if not sWorkspaceId:
            raise ValueError("Missing workspace id")

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{wasdi_api_url}/rest/product/namesbyws",
                params={"workspace": sWorkspaceId},
                headers={"x-session-token": sSessionToken}
            )
            oResponse.raise_for_status()
            logging.debug("WASDI getNamesByWorkspace call completed with status %s", oResponse.status_code)
            return oResponse.text


    @mcp_server.tool()
    async def get_light_list_of_products_by_workspace(sWorkspaceId: str, oContext: Context = None) -> str:
        """Returns basic summary details for all products inside a workspace.
        
        Provides more spatial awareness than raw names while keeping payload transmission lightweight.
        
        Args:
            sWorkspaceId: The unique ID of the workspace.
        Returns:
            A JSON array containing objects with basic fields: 'name', 'productFriendlyName', and spatial 'bbox'.
        """
        sSessionToken = getSessionToken(oContext)

        if not sSessionToken:
            raise ValueError("Missing x-session-token header")

        if not sWorkspaceId:
            raise ValueError("Missing workspace id")

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{wasdi_api_url}/rest/product/bywslight",
                params={"workspace": sWorkspaceId},
                headers={"x-session-token": sSessionToken}
            )
            oResponse.raise_for_status()
            logging.debug("WASDI getLightListByWorkspace call completed with status %s", oResponse.status_code)
            return oResponse.text


    @mcp_server.tool()
    async def get_product_details_by_product_name(sProductName: str, sWorkspaceId: str, oContext: Context = None) -> str:
        """Returns complete metadata details for a single target product file in a workspace.
        
        Use this tool to read advanced metadata flags, GeoServer map styling, spatial bounding boxes, and band groupings.
        
        Args:
            sProductName: The file name of the product relative to the workspace path (e.g., 'satellite_image.tif').
            sWorkspaceId: The workspace ID housing the file.
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
            
    @mcp_server.tool()
    async def download_product_by_name(sFileName: str, sWorkspaceId: str, oContext: Context = None) -> str:
        """Downloads a file directly from the workspace.
        
        Args:
            sFileName: The name of the file relative to the workspace root path.
            sWorkspaceId: The unique ID of the hosting workspace.
        Returns:
            The raw binary file contents converted and returned as a hexadecimal string.
        """
        sSessionToken = getSessionToken(oContext)

        if not sSessionToken:
            raise ValueError("Missing x-session-token header")

        if not sFileName:
            raise ValueError("Missing file name")

        if not sWorkspaceId:
            raise ValueError("Missing workspace id")

        aoParams = {"filename": sFileName, "workspace": sWorkspaceId, "token": sSessionToken, "procws": "", "disposition": "attachment"}
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


    @mcp_server.tool()
    async def add_product_to_workspace(sProductName: str, sWorkspaceId: str, oContext: Context = None) -> str:
        """Registers an untracked local disk file into the WASDI workspace database catalog.
        
        Use this tool when an execution task generates a product file on the node directory, 
        and you want to make it visible and interactive on the main user interface.
        
        Args:
            sProductName: The relative path or name of the file sitting on the local workspace node folder.
            sWorkspaceId: Target workspace identification string.
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


    @mcp_server.tool()
    async def share_file_to_workspace(
        sOriginWorkspaceId: str,
        sDestinationWorkspaceId: str,
        sProductName: str,
        oContext: Context = None,
    ) -> str:
        """
        Sends a file from one workspace to another. 
        The file is copied to the destination workspace. 
        The file is not moved, so it will still be available in the origin workspace.

        Inputs:
        sOriginWorkspaceId: is the unique id of the origin workspace
        sDestinationWorkspaceId: is the unique id of the destination workspace
        sProductName: is the name of the product/file to share
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
            "productName": sProductName
        }
        aoParams = {sKey: sValue for sKey, sValue in aoParams.items() if sValue is not None}

        async with httpx.AsyncClient() as oClient:
            oResponse = await oClient.get(
                f"{wasdi_api_url}/rest/filebuffer/share",
                params=aoParams,
                headers={"x-session-token": sSessionToken},
            )
            oResponse.raise_for_status()
            logging.debug("WASDI share call completed with status %s", oResponse.status_code)
            return oResponse.text

    # @mcp_server.tool()
    # async def get_detailed_list_of_products_by_workspace(sWorkspaceId: str, oContext: Context = None) -> str:
    #     """
    #     Returns the detailed list of products in a workspace. 
    #     The agent can use this API to get the full list of products in a workspace with all the details, including metadata, styles, bands information.
    #     This API can take a lot of time for workspaces with many products, use it only when the agent really needs all the details of all the products in the workspace. 
    #     There are 2 alternatives get_light_list_of_products_by_workspace and get_names_of_products_by_workspace that can be used to get a list of products with less details.    
    #     Using the alternative APIs, the agent can get the details of a specific product using the get_product_details_by_product_name API.

    #     Input
    #     sWorkspaceId: unique id of the workspace for which we want to get the list of products.

    #     Output
    #     An array JSON object with the following properties for each product:

    #     bbox: optional property with the bounding box of the product, in case the product is an EO product with georeferenced data
    #     name: is the file name without extension
    #     description: optional property with the description of the product, if it is provided by the user when the product is created or edited
    #     fileName: is the file name with extension
    #     productFriendlyName: is a name that the user can assign to this product
    #     metadataFileCreated: A boolean true if the metadata has been generated, false otherwise.
    #     metadataFileReference: if the metadataFileCreated is true, this property contains the path to the metadata file that has been generated. The path is relative to the metadata wasdi folder on the server
    #     metadata: real metadata if the metadataFileCreated is true. This property is not provided by default because it can be very heavy, especially for products with a lot of metadata, so it is better to read the metadata only when it is needed, using the metadataFileReference property to access the metadata file.
    #     bandsGroups: optional property with the bands groups of the product, if it is an EO product with multiple bands
    #     style: optional property with the name of the style of the product. Styles are Geoserver styles the user can upload in wasdi. If the product has a style assigned, it means that the user has uploaded a style in wasdi and assigned it to this product, so the style property contains the name of the style that is assigned to the product. The agent can use this information to suggest to the user to use this style when visualizing the product in wasdi, or to use this style as a reference when generating a new style for this product.

    #     the call returns null in case of errors or a
    #     """
    #     sSessionToken = getSessionToken(oContext)

    #     if not sSessionToken:
    #         raise ValueError("Missing x-session-token header")

    #     if not sWorkspaceId:
    #         raise ValueError("Missing workspace id")

    #     async with httpx.AsyncClient() as oClient:
    #         oResponse = await oClient.get(
    #             f"{wasdi_api_url}/rest/product/byws",
    #             params={"workspace": sWorkspaceId},
    #             headers={"x-session-token": sSessionToken}
    #         )
    #         oResponse.raise_for_status()
    #         logging.debug("WASDI getListByWorkspace call completed with status %s", oResponse.status_code)
    #         return oResponse.text



