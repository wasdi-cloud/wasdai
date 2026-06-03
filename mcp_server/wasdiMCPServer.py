from mcp.server.fastmcp import FastMCP
import httpx

oMcpServer = FastMCP("wasdi-mcp-server", "0.1.0")

@oMcpServer.tool()
def hello(sName: str) -> str:
    """Says hello to someone, whose name is give as an input parameter."""
    return f"Hello {sName}!"


@oMcpServer.tool()
async def wasdiHello() -> str:
    """Calls the WASDI hello enpoint to check if the service is up and running."""
    async with httpx.AsyncClient() as oClient:
        oResponse = await oClient.get("https://www.wasdi.net/wasdiwebserver/rest/wasdi/hello")
        oResponse.raise_for_status()
        return oResponse.text
    
@oMcpServer.tool()
async def get_workspaces() -> str:
    """
    Returns the list of workspaces for the currently authenticated user.
    The workspaces being retunned are those that the user has access to, either as owner or as collaborator.
    The AI agent should return to the user the total count of workspaces and the name of each workspace.
    """
    session_token  = "xxx"
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://www.wasdi.net/wasdiwebserver/rest/ws/byuser",
            headers={"x-session-token": session_token}
        )
        response.raise_for_status()
        print(response)
        return response.text

if __name__ == "__main__":
    oMcpServer.run()