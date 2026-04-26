from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx
import uvicorn
import json

app = FastAPI()

OLLAMA_URL = "http://localhost:11434"

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(path: str, request: Request):
    body = await request.body()
    
    if request.method == "POST" and "chat/completions" in path:
        try:
            data = json.loads(body)
            print("\n\n================ [OLLAMA RECEIVED (V1 CHAT)] ================")
            if "messages" in data:
                for msg in data["messages"]:
                    print(f"[{msg['role'].upper()}]: {msg['content']}")
            else:
                print(json.dumps(data, indent=2))
            print("===================================================\n")
        except Exception as e:
            print(f"[Could not parse body: {e}]")

    elif request.method == "POST" and "api/generate" in path:
        try:
            data = json.loads(body)
            print("\n\n================ [OLLAMA RECEIVED (NATIVE)] ================")
            print(f"[PROMPT]: {data.get('prompt', '')}")
            print("===========================================================\n")
        except Exception as e:
            pass

    client = httpx.AsyncClient()
    
    # Strip headers that might mess up the proxying (like content-length/encoding)
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    
    req = client.build_request(
        request.method,
        f"{OLLAMA_URL}/{path}",
        headers=headers,
        content=body
    )
    
    r = await client.send(req, stream=True)
    
    async def stream_generator():
        print(">> [OLLAMA REPLYING]: ", end="", flush=True)
        async for chunk in r.aiter_bytes():
            try:
                chunk_str = chunk.decode('utf-8', errors='ignore')
                # OpenAI format streaming chunks
                if chunk_str.startswith("data: "):
                    for line in chunk_str.split("\n"):
                        if line.startswith("data: ") and line.strip() != "data: [DONE]":
                            payload = json.loads(line[6:])
                            if "choices" in payload and len(payload["choices"]) > 0:
                                delta = payload["choices"][0].get("delta", {})
                                if "content" in delta:
                                    print(delta["content"], end="", flush=True)
                # Ollama native streaming chunks
                elif "response" in chunk_str:
                    for line in chunk_str.split("\n"):
                        if line.strip():
                            payload = json.loads(line)
                            if "response" in payload:
                                print(payload["response"], end="", flush=True)
            except Exception:
                pass
            yield chunk
        print("\n===================================================\n")
        
    return StreamingResponse(
        stream_generator(), 
        status_code=r.status_code, 
        headers=r.headers
    )

if __name__ == "__main__":
    print("=================================================")
    print("Starting Ollama Debug Proxy on port 11435...")
    print("=================================================")
    print("1. Go to Dograh UI -> Workflows -> Your Workflow")
    print("2. Click on the LLM Node -> Ollama (Local)")
    print("3. Change Base URL to: http://host.docker.internal:11435/v1")
    print("   (Fallback: http://host.docker.internal:11435/api)")
    print("4. Save, and Make a Test Call!")
    print("=================================================\n")
    uvicorn.run(app, host="0.0.0.0", port=11435)
