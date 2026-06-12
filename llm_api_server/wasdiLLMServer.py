import logging
import os
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Annotated, Any

from fastapi import Body, FastAPI, Header
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import (
    MCPToolCallRequest,
)
from langchain_openai import ChatOpenAI
from utils.LoggingConfiguration import setupLogging

from utils.WasdiConfig import WasdiConfig

setupLogging()

oApp = FastAPI(root_path="/api")

X_SESSION_TOKEN_CTX: ContextVar[str] = ContextVar("x_session_token", default="")

logging.info("Loading configuration")
sConfigFilePath = os.getenv(
    "WASDI_CONFIG_PATH", 
    "C:\\WASDI\\GIT\\wasdai\\config.json"
)

if not (oConfig := WasdiConfig(sConfigFilePath)):
    logging.error("Failed to load configuration")
    raise RuntimeError(f"Could not load config from {sConfigFilePath}")

sEndpoint = oConfig.aiAgent.llm_endpoint
sToken = oConfig.aiAgent.llm_token
sModel = oConfig.aiAgent.llm_model

logging.info("Initializing LLM client")
logging.info(f"LLM Endpoint: {sEndpoint}")
logging.info(f"LLM Model: {sModel}")

s_oLLM = ChatOpenAI(
    base_url=sEndpoint + "/v1",
    api_key=sToken,
    model=sModel
)

if s_oLLM:
    logging.info("LLM client initialized successfully")
else:
    logging.error("Failed to initialize LLM client")
    # raise RuntimeError("LLM client initialization failed")

logging.info("Initializing the MCP client")


async def _inject_session_header(
    oRequest: MCPToolCallRequest,
    oHandler: Callable[[MCPToolCallRequest], Awaitable[Any]],
) -> Any:
    """Inject user session token into MCP HTTP calls for each tool invocation."""
    sToken = X_SESSION_TOKEN_CTX.get().strip()
    if sToken:
        oRequest = oRequest.override(
            headers={
                "x-session-token": sToken,
            }
        )
    return await oHandler(oRequest)


s_oMCPClient = MultiServerMCPClient({
    "wasdi": {
        "url": "http://localhost:7000/mcp",
        "transport": "http",
    }
}, tool_interceptors=[_inject_session_header])

@oApp.get("/hello")
async def hello():
    """Endpoint to test if the server is up and running."""
    return "Hello from the WASDI LLM Server!"


@oApp.post("/chat")
async def chat(
    sPrompt: Annotated[str, Body()],
    x_session_token: Annotated[str | None, Header()] = None,
):
    sSessionToken = (x_session_token or "").strip()
    if not sSessionToken:
        raise ValueError("Missing x-session-token header")

    logging.debug(f"Received request with token: {sSessionToken} and prompt: {sPrompt}")

    oTokenReset = X_SESSION_TOKEN_CTX.set(sSessionToken)
    try:
        # Get tools from MCP server
        s_oTools = await s_oMCPClient.get_tools()

        # RUN THE AGENT
        oAgent = create_agent(model=s_oLLM, tools=s_oTools)

        oResult = await oAgent.ainvoke(
            {"messages": [{"role": "user", "content": sPrompt}]}
        )
    finally:
        X_SESSION_TOKEN_CTX.reset(oTokenReset)

    sResponse = oResult["messages"][-1].content
    logging.info(f"PROMPT: {sPrompt}")
    logging.info(f"Response from MCP agent: {sResponse}")
    return sResponse
    """
    # Return available tools info for now
    return {
        "prompt": sPrompt,
        "tools_available": len(tools),
        "tool_names": [tool.name for tool in tools]
    }
    """



if __name__ == "__main__":
    import uvicorn
    logging.info("Starting WASDI LLM API Server...")
    uvicorn.run("wasdiLLMServer:oApp", host="127.0.0.1", port=8000)