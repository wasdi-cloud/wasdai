import logging
import os
import uuid

import re
import time
from typing import Annotated, Any
from collections.abc import Awaitable, Callable
from contextvars import ContextVar

from fastapi import Body, FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import (
    MCPToolCallRequest,
)
from langchain_openai import ChatOpenAI
from utils.LoggingConfiguration import setupLogging
from utils.WasdiConfig import WasdiConfig
from llm_api_server.MongoDBClient import MongoDBClient
from llm_api_server.data.SessionRepository import SessionRepository
from llm_api_server.data.ChatsRepository import ChatRepository
from llm_api_server.data.UserRepository import UserRepository
from llm_api_server.business.Chat import Chat


setupLogging()

oApp = FastAPI(root_path="/assistant")

X_SESSION_TOKEN_CTX: ContextVar[str] = ContextVar("x_session_token", default="")

logging.info("Loading configuration")
s_sConfigFilePath = os.getenv(
    "WASDI_CONFIG_PATH", 
    "C:\\WASDI\\GIT\\wasdai\\config.json"
)

if not (s_oConfig := WasdiConfig(s_sConfigFilePath)):
    logging.error("Failed to load configuration")
    raise RuntimeError(f"Could not load config from {s_sConfigFilePath}")


logging.info("Adding CORS middleware")
s_asAllowedOrigins = s_oConfig.LLM_server.allowed_origins
oApp.add_middleware(
    CORSMiddleware,
    allow_origins=s_asAllowedOrigins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

sEndpoint = s_oConfig.aiAgent.llm_endpoint
sToken = s_oConfig.aiAgent.llm_token
sModel = s_oConfig.aiAgent.llm_model

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


MongoDBClient._s_oConfig = s_oConfig

@oApp.get("/hello")
async def hello():
    """Endpoint to test if the server is up and running."""
    return "Hello from the WASDI LLM Server!"

@oApp.get("/newChat")
async def new_chat(x_session_token: Annotated[str | None, Header()] = None):
    """Endpoint to initialize a new chat session."""
    logging.info("Initializing new chat session")

    sSessionToken = (x_session_token or "").strip()

    if not isTokenSecure(sSessionToken):
        logging.warning(f"newChat. Invalid or missing session token: {sSessionToken}")
        raise ValueError("Invalid or missing session token")

    sUserId = getUserFromSession(sSessionToken)

    if not sUserId:
        logging.warning(f"newChat. No user associated with session token: {sSessionToken}")
        raise ValueError("No user associated with this session token")

    logging.info(f"newChat. Session found for token: {sSessionToken}, userId: {sUserId}")

    oChatRepository = ChatRepository()

    sUUID = str(uuid.uuid4())
    aoChats = oChatRepository.getEntitiesByField({"chatId": sUUID})
    while aoChats is not None and len(aoChats) > 0:
        sUUID = str(uuid.uuid4())
        aoChats = oChatRepository.getEntitiesByField({"chatId": sUUID})
    
    oNewChat = Chat()
    oNewChat.chatId = sUUID
    oNewChat.userId = sUserId
    oNewChat.startDate = time.time() * 1000

    bResult = oChatRepository.addEntity(oNewChat)

    if not bResult:
        logging.warning(f"newChat. Failed to create a new chat for user {sUserId}")
        raise ValueError("Failed to create a new chat")
    
    return sUUID


def getUserFromSession(sSessionToken: str):
    """Utility function to get user information from session token."""

    if not sSessionToken:
        return NotImplementedError
            
    # check if the token is associated with a user
    oSessionRepository = SessionRepository()
    aoSession = oSessionRepository.getEntitiesByField({"sessionId": sSessionToken})

    if not aoSession:
        logging.warning(f"getUserFromSession. No session found for token: {sSessionToken}")
        return None
    
    oSession = aoSession[0]
    
    # check if the session is not expired
    lNow = int(time.time() * 1000)  # cast to millis
    lTimeSpan = s_oConfig.LLM_server.sessionExpireHours * 60 * 60 * 1000
    lLimit = lNow - lTimeSpan

    if oSession.lastTouch < lLimit:
        return None
    
    sUserId = oSession.userId

    # check if the user id is present
    oUserRepository = UserRepository()
    aoUsers = oUserRepository.getEntitiesByField({"userId": sUserId})

    if not aoUsers:
        logging.warning(f"getUserFromSession. No user found for ID: {sUserId}")
        return None

    oUser = aoUsers[0]
    return oUser.userId


def isTokenSecure(sSessionToken: str) -> bool:
    """Utility function to check if the session token is secure and valid."""
    if not sSessionToken:
        return False

    # ensure the parameter is strictly a string
    if not isinstance(sSessionToken, str):
        logging.warning("isTokenSecure. Security Alert: session token is not a string")
        return False

    # length Check
    if not (10 <= len(sSessionToken) <= 64):
        logging.warning("isTokenSecure. Security Alert: session token has invalid length")
        return False

    # format Check: Only allow alphanumeric characters and hyphens
    if not re.match(r"^[a-zA-Z0-9\-]+$", sSessionToken):
        logging.warning("isTokenSecure. Security Alert: session token contains invalid characters")
        return False

    return True


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