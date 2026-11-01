import json
import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from openai import OpenAI, OpenAIError
from pydantic import BaseModel

CHAT_MODEL = "qwen3:4b"
EMBED_MODEL = "nomic-embed-text"
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

app = FastAPI(title="Chat + RAG + Agent API")

class ErrorResponse(BaseModel):
    error: str
    detail: str

ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Bad imput"},
    422: {"model": ErrorResponse, "description": "Validation error"},
    500: {"model": ErrorResponse, "description": "Server error"},
}

@app.exception_handler(RequestValidationError)
def handle_http_exception(request: Request, exc: RequestValidationError):
    """400/500 errors that we raise ourselves with HTTPException"""
    names = {400: "Bad Request", 500: "Internal Server Error"}
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": names.get(exc.status_code, "Error"), "detail": exc.detail}
    )

@app.exception_handler(RequestValidationError)
def handle_validation_error(request: Request, exc: RequestValidationError):
    """422 errors raised by Pydantic when the request body has the wrong shape"""
    # Turn Pydantic's list of errors into one readable line,
    # e.g. "body.prompt: Field required"
    messages=[]
    for err in exc.errors():
        # err looks like: {"loc": ("body", "prompt"), "msg": "Field required", ...}

        # Step 1: turn the location ("body", "prompt") into "body.prompt"
        location_parts = []
        for part in err["loc"]:
            location_parts.append(part)
        location = ".".join(location_parts)

        # Step 2: add the message -> "body.prompt: Field required"
        messages.append(location + ": " + err["msg"])
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Validation Error", "detail": "; ".join(messages)}
    )

@app.exception_handler(Exception)
def handle_unexpected_error(request: Request, exc: Exception):
    """Safety net: any error we did not expect becomes a clean 500."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal Server Error", "detail": str(exc)}
    )

def check_not_empty(value: str, field: str):
    """Raise 400 if a text field is only empty or only spaces."""
    if not value.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)

def ask_llm(messages: list, tools: list = None):
    """Call the LLM once and return its message. Raises 500 if the LLM fails."""
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=tools,
        )
    except OpenAIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"LLM call failed: {e}")
    return response.choices[0].message

# ---------------------------------------------------------------------------
# 1. POST /chat  -  plain question -> answer
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK, responses=ERROR_RESPONSES)
def chat(req: ChatRequest):
    check_not_empty(req.prompt, "prompt")
    message=ask_llm([{"role": "user", "content": req.prompt}])
    return ChatResponse(answer=message.content)

# ---------------------------------------------------------------------------
# 2. POST /rag-query  -  find relevant documents, then answer from them
# ---------------------------------------------------------------------------

# A tiny "knowledge base". In a real app this would come from files / a vector DB.
DOCUMENTS = [
    "FastAPI is a Python web framework for building APIs. It uses type hints and Pydantic.",
    "Pydantic validates data using Python type hints and raises errors on bad input.",
    "Ollama runs large language models locally on your own computer for free.",
    "RAG stands for Retrieval-Augmented Generation: retrieve relevant text, then let the LLM answer using it.",
    "Uvicorn is an ASGI server used to run FastAPI apps.",
]

def embed(texts: list[str]) -> np.ndarray:
    """Turn a list of texts into a matrix of embedding vectors."""
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([item.embedding for item in response.data])

# Embed the documents once when the server starts
DOC_VECTORS = embed(DOCUMENTS)

def search(query: str, top_k: int) -> list[str]:
    """Return the top_k documents most similar to the query (cosine similarity)"""
    try:
        query_vector = embed([query])[0]
    except OpenAIError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Embedding failed: {e}")
    scores = DOC_VECTORS @ query_vector / (
        np.linalg.norm(DOC_VECTORS, axis=1) * np.linalg.norm(query_vector)
    )
    best = np.argsort(scores)[::-1][:top_k] #highest score first
    print("-----------")
    print(f"best:{best}")
    print("-----------")
    return [DOCUMENTS[i] for i in best]

class RagRequest(BaseModel):
    question: str
    top_k: int = 2

class RagResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/rag-query", response_model=RagResponse, status_code=status.HTTP_200_OK, responses=ERROR_RESPONSES)
def rag_query(req: RagRequest):
    check_not_empty(req.question, "question")
    if not 1 <= req.top_k <= len(DOCUMENTS):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'top_k' must be between 1 and {len(DOCUMENTS)}",
        )

    sources = search(req.question, req.top_k)
    print("----------------")
    print(f"sources:{sources}")
    print("----------------")
    context = "\n".join(f"- {s}" for s in sources)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {req.question}"
    )
    message = ask_llm([{"role": "user", "content": prompt}])
    return RagResponse(answer=message.content, sources=sources)

# ---------------------------------------------------------------------------
# 3. POST /agent-run  -  LLM can call tools in a loop until it has an answer
# ---------------------------------------------------------------------------

def add(a: float, b: float) -> float:
    return a + b

def multiply(a: float, b: float) -> float:
    return a * b

# Python functions the agent is allowed to call.
TOOL_FUNCTIONS = {"add": add, "multiply": multiply}

# The same tools described in the format the LLM understands.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name.capitalize()} two numbers.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    }
    for name in TOOL_FUNCTIONS
]

class AgentRequest(BaseModel):
    task: str
    max_steps: int = 5

class ToolCall(BaseModel):
    tool: str
    arguments: dict
    result: float

class AgentResponse(BaseModel):
    answer: str
    steps: list[ToolCall]

@app.post("/agent-run", response_model=AgentResponse, status_code=status.HTTP_200_OK, responses=ERROR_RESPONSES)
def agent_run(req: AgentRequest):
    check_not_empty(req.task, "task")
    if not 1 <= req.max_steps <= 10:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="'max_steps' must be bentween 1 and 10")

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the tools for any math."},
        {"role": "user", "content": req.task},
    ]
    steps = []

    for _ in range(req.max_steps):
        message = ask_llm(messages, tools=TOOLS)
        print("-----------------------")
        print(f"message:{message}")
        print("-----------------------")

        # No tool calls -> the LLM has its final answer.
        if not message.tool_calls:
            return AgentResponse(answer=message.content, steps=steps)

        # Otherwise run each tool the LLM asked for and send the result back
        messages.append(message)
        for call in message.tool_calls:
            name = call.function.name
            # print("-----------------------")
            # print(f"name:{name}")
            # print("-----------------------")
            if name not in TOOL_FUNCTIONS:
                # The LLM made a mistake, not the user -> 500.
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Agent asked for unknown tool '{name}'"
                )
            args = json.loads(call.function.arguments)
            # print("-----------------------")
            # print(f"args:{args}")
            # print("-----------------------")
            result = TOOL_FUNCTIONS[name](**args)
            # print("-----------------------")
            # print(f"result:{result}")
            # print("-----------------------")
            steps.append(ToolCall(tool=name, arguments=args, result=result))
            # print("-----------------------")
            # print(f"steps:{steps}")
            # print("-----------------------")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
            # print("-----------------------")
            # print(f"messages:{messages}")
            # print("-----------------------")

    return AgentResponse(answer="Stopped: reached max_steps.", steps=steps)
