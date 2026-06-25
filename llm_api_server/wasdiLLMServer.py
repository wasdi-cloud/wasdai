import logging
import os
import traceback
import uuid
import re
import time

from typing import Annotated, Any
from itertools import zip_longest
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from fastapi import FastAPI, Body, Header, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import create_agent
from fastapi.responses import StreamingResponse
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import (MCPToolCallRequest,)
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
    model=sModel,
    streaming=True
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
        "url": s_oConfig.MCP_server.url,
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
    
    logging.info("newChat. Initializing new chat session")

    sSessionToken = (x_session_token or "").strip()

    if not isTokenSecure(sSessionToken):
        logging.warning(f"newChat. Invalid or missing session token: {sSessionToken}")
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing session token"
        )

    sUserId = getUserFromSession(sSessionToken)

    if not sUserId:
        logging.warning(f"newChat. No user associated with session token: {sSessionToken}")
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail="No user associated with this session token"
        )


    logging.info(f"newChat. Session found for token: {sSessionToken}, userId: {sUserId}")
    
    oChatRepository = ChatRepository()

    sUUID = str(uuid.uuid4())
    oChat = oChatRepository.getEntityById(sUUID)
    while oChat is not None:
        sUUID = str(uuid.uuid4())
        oChat = oChatRepository.getEntityById(sUUID)
    
    oNewChat = Chat()
    oNewChat.chatId = sUUID
    oNewChat.userId = sUserId
    oNewChat.startDate = time.time() * 1000

    bResult = oChatRepository.addEntity(oNewChat)

    if not bResult:
        logging.warning(f"newChat. Failed to create a new chat for user {sUserId}")
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create a new chat"
        )
    
    return sUUID


def getUserFromSession(sSessionToken: str):
    """Utility function to get user information from session token."""

    if not sSessionToken:
        return NotImplementedError
            
    # check if the token is associated with a user
    oSessionRepository = SessionRepository()
    oSession = oSessionRepository.getEntityById(sSessionToken)

    if not oSession:
        logging.warning(f"getUserFromSession. No session found for token: {sSessionToken}")
        return None
        
    # check if the session is not expired
    lNow = int(time.time() * 1000)  # cast to millis
    lTimeSpan = s_oConfig.LLM_server.sessionExpireHours * 60 * 60 * 1000
    lLimit = lNow - lTimeSpan

    if oSession.lastTouch < lLimit:
        return None
    
    sUserId = oSession.userId

    # check if the user id is present
    oUserRepository = UserRepository()
    oUser = oUserRepository.getEntityById(sUserId)

    if not oUser:
        logging.warning(f"getUserFromSession. No user found for ID: {sUserId}")
        return None

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
    sChatId: Annotated[str, Query(alias="chatId")], 
    x_session_token: Annotated[str | None, Header()] = None,
):
    """
    Implements an interaction between a user and the ai agent. 
    :param sPrompt: the user's prompt
    :param sChatId: the unique identifier of the chat
    """
    sSessionToken = (x_session_token or "").strip()

    logging.debug(f"chat. Received request with token: {sSessionToken} and prompt: {sPrompt}")

    if not isTokenSecure(sSessionToken):
        logging.warning(f"chat. Invalid or missing session token: {sSessionToken}")
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing session token"
        )

    sUserId = getUserFromSession(sSessionToken)

    if not sUserId:
        logging.warning(f"chat. No user associated with session token: {sSessionToken}")
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail="No user associated with this session token"
        )

    logging.info(f"chat. Session found for token: {sSessionToken}, userId: {sUserId}")


    if not sChatId:
        logging.warning(f"chat. Chat id not specified")
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail="Missing chat id"
        )

    oChatRepository = ChatRepository()

    aoChats = oChatRepository.getEntitiesByField({"chatId": sChatId, "userId": sUserId})

    if not aoChats:
        logging.warning(f"getChat. No chat corresponding to the id {sChatId}")
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail="Chat not found"
        )
    
    if len(aoChats) == 0:
        logging.warning(f"getChat. Found zero chats for the id {sChatId}")
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail="Chat not found"
        )
    
    oChat = aoChats[0]    

    try:
        # implement chat history
        aoMessages = [] 
        aoPastPairs = list(zip_longest(oChat.prompts, oChat.answers))[-10:] # Take only the last 10 exchanges to avoid growing the context window
        for sPastPrompt, sPastAnswer in aoPastPairs:
            if sPastPrompt:
                aoMessages.append({"role": "user", "content": sPastPrompt})
            if sPastAnswer:
                aoMessages.append({"role": "assistant", "content": sPastAnswer })
        # at last, append the most recent prompt
        aoMessages.append(
            { "role": "user", "content": sPrompt})

        try:
            # Get tools from MCP server
            s_oTools = await s_oMCPClient.get_tools()

            # RUN THE AGENT
            oAgent = create_agent(model=s_oLLM, tools=s_oTools)

            # stream the response chunks
            async def event_generator():

                oTokenReset = X_SESSION_TOKEN_CTX.set(sSessionToken)

                sFullResponse = ""

                try: 
                    async for oEvent in oAgent.astream_events({"messages": aoMessages}, version="v2"):
                        # select only the messages where the LLM is actually typing text
                        sType = oEvent.get("event")
                        if sType == "on_chat_model_stream":
                            oChunk = oEvent.get("data", {}).get("chunk")
                            if oChunk and hasattr(oChunk, "content") and oChunk.content:
                                sToken = oChunk.content
                                sFullResponse += sToken
                                yield sToken    # yield the text chunk directly to the client
                except Exception as oE:
                    logging.error(f"chat. Agent streaming faild. {oE}")
                    if hasattr(oE, "exceptions"):
                        for i, sub_exc in enumerate(oE.exceptions):
                            logging.error(f"Sub-Exception #{i}: {type(sub_exc).__name__} - {sub_exc}")
                    sError = "\n[The WASDI AI agent encountered an error while streaming]" # TODO: translation
                    sFullResponse += sError
                    yield sError
                finally:
                    # Fallback context reset in case generator setup fails before yielding
                    X_SESSION_TOKEN_CTX.reset(oTokenReset)
                                        
                    if sFullResponse:
                        # now it is the moment to store the chat into the db
                        aoPrompts = oChat.prompts + [sPrompt]
                        aoAnswers = oChat.answers + [sFullResponse]

                        oChat.prompts = aoPrompts
                        oChat.answers = aoAnswers

                        if oChatRepository.updateEntity(oChat) < 0:
                            logging.warning("chat. Chat was not updated")
                        
            """
            oResult = await oAgent.ainvoke(
                {
                    "messages": aoMessages
                }
            )
            
            class MockMessage:
                content = f"This is a hardcoded mock response from the WASDI AI agent to the prompt: {sPrompt} "
            oResult = {"messages": [MockMessage()]}
            """
            

            return StreamingResponse(
                event_generator(), 
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"  # This is the magic bullet for Nginx/Proxies
                })
        except Exception as oE:
            logging.error(f"chat. Setup failed: {oE}")
            raise HTTPException(status_code=500, detail="Internal Server Error")
    except Exception as oE:
        logging.error(f"chat. Exception: {oE}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@oApp.get("/getChat")
async def getChat(
    sChatId: Annotated[str, Query(alias="chatId")], 
    x_session_token: Annotated[str | None, Header()] = None,
):
    """
    Get all the messages exchanged in a chat between the user and the  AI assistant
    :param sChatId: the unique identifier of the chat
    """
    sSessionToken = (x_session_token or "").strip()

    logging.debug(f"getChat. Received request with token: {sSessionToken}")

    if not isTokenSecure(sSessionToken):
        logging.warning(f"getChat. Invalid or missing session token: {sSessionToken}")
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing session token"
        )

    sUserId = getUserFromSession(sSessionToken)

    if not sUserId:
        logging.warning(f"getChat. No user associated with session token: {sSessionToken}")
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail="No user associated with this session token"
        )

    logging.info(f"getChat. Session found for token: {sSessionToken}, userId: {sUserId}")


    if not sChatId:
        logging.warning(f"getChat. Chat id not specified")
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail="Missing chat id"
        )

    oChatRepository = ChatRepository()
    aoChats = oChatRepository.getEntitiesByField({"chatId": sChatId, "userId": sUserId})

    if not aoChats:
        logging.warning(f"getChat. No chat corresponding to the id {sChatId}")
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail="Chat not found"
        )
    
    if len(aoChats) == 0:
        logging.warning(f"getChat. Found zero chats for the id {sChatId}")
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail="Chat not found"
        )
    
    oChat = aoChats[0]
    
    try:
        sTitle = getTitle(oChat.prompts) 

        return {
            "chatId": oChat.chatId,
            "timestamp": oChat.startDate,
            "title": sTitle,
            "prompts": oChat.prompts,
            "answers": oChat.answers
        }
    
    except Exception as oE:
        logging.warning(f"getChat. Exception creating the chat structure to send to the client {oE}")
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat not found"
        )

@oApp.get("/listChat")
async def listChat(
    x_session_token: Annotated[str | None, Header()] = None,
):
    """
    Get the list of all the chats of a user
    """
    sSessionToken = (x_session_token or "").strip()
    logging.debug(f"listChat. Received request with token: {sSessionToken}")
    if not isTokenSecure(sSessionToken):
        logging.warning(f"listChat. Invalid or missing session token: {sSessionToken}")
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing session token"
        )
    
    sUserId = getUserFromSession(sSessionToken)

    if not sUserId:
        logging.warning(f"listChat. No user associated with session token: {sSessionToken}")
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail="No user associated with this session token"
        )

    logging.info(f"listChat. Session found for token: {sSessionToken}, userId: {sUserId}")


    oChatRepository = ChatRepository()
    try: 
        aoChatList = oChatRepository.getEntitiesByField({"userId": sUserId})

        if not aoChatList:
            return []
        
        aoChatList.sort(key=lambda oChat : oChat.startDate, reverse=True)
        
        aoChats = []

        for oChat in aoChatList:
                sTitle = getTitle(oChat.prompts)
                aoChats.append(
                    {
                        "title": sTitle,
                        "chatId": oChat.chatId,
                        "timestamp": oChat.startDate
                    }
                )

        
        return aoChats
                
    except Exception as oE:
        logging.warning(f"getChat. Exception creating the chat structure to send to the client {oE}")
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat not found"
        )

def getTitle(aoPrompts) -> str:
        sTitle = "No title" #TODO: translation
        if aoPrompts: 
            sFirstPrompt = aoPrompts[0]
            # Check if it needs truncation
            if len(sFirstPrompt) > 30:
                sTitle = f"{sFirstPrompt[:30]}..."
            else:
                sTitle = sFirstPrompt
        return sTitle

if __name__ == "__main__":
    import uvicorn
    logging.info("Starting WASDI LLM API Server...")
    uvicorn.run("wasdiLLMServer:oApp", host="127.0.0.1", port=8000)