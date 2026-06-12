import logging
from typing import Annotated

from fastapi import Body, FastAPI, Header
from utils.LoggingConfiguration import setupLogging
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from utils.WasdiConfig import WasdiConfig

setupLogging()

oApp = FastAPI(root_path="/api")

logging.info("Loading configuration")
sConfigFilePath = "C:\\WASDI\\GIT\\wasdai\\config.json"
if not (oConfig := WasdiConfig(sConfigFilePath)):
    logging.error("Failed to load configuration")
    raise RuntimeError(f"Could not load config from {sConfigFilePath}")

sEndpoint = oConfig.aiAgent.llm_endpoint
sToken = oConfig.aiAgent.llm_token

logging.info("Initializing LLM client")
logging.info(f"LLM Endpoint: {sEndpoint}")
logging.info(f"LLM Model: {oConfig.aiAgent.llm_model}")

s_oLLM = ChatOpenAI(
    base_url=sEndpoint + "/v1",
    api_key=sToken,
    model="llama3.1:8b"
)

if s_oLLM:
    logging.info("LLM client initialized successfully")
else:
    logging.error("Failed to initialize LLM client")
    # raise RuntimeError("LLM client initialization failed")

@oApp.get("/hello")
async def hello():
    """Endpoint to test if the server is up and running."""
    return "Hello from the WASDI LLM Server!"


@oApp.post("/chat")
async def chat(x_session_token: Annotated[str, Header()],
               sPrompt: Annotated[str, Body()]):
    logging.debug(f"Received request with token: {x_session_token} and prompt: {sPrompt}")

    oClient = MultiServerMCPClient({
        "wasdi": {
            "url": "http://localhost:7000/mcp",
            "transport": "http"
        }
    })
    tools = await oClient.get_tools()
    # RUN THE AGENT
    agent = create_agent(model=s_oLLM, tools=tools)

    oResult = await agent.ainvoke({
        "messages": [{"role": "user", "content": sPrompt}]
    })

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