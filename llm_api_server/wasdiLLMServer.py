import logging
from typing import Annotated

from fastapi import Body, FastAPI, Header
from fastapi.responses import Response

oApp = FastAPI(root_path="/api")

@oApp.get("/hello")
async def hello():
    """Endpoint to test if the server is up and running."""
    return "Hello from the WASDI LLM Server!"


@oApp.post("/chat")
async def chat(x_session_token: Annotated[str, Header()],
               sPrompt: Annotated[str, Body()]):
    return {
        "receivedToken": x_session_token,
        "receivedPrompt": sPrompt,
    }

if __name__ == "__main__":
    import uvicorn
    logging.info("Starting WASDI LLM API Server...")
    uvicorn.run("wasdiLLMServer:oApp", host="127.0.0.1", port=8000)