import asyncio
import os
from openai import AsyncOpenAI

async def diag():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434/v1")
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    
    print(f"Connecting to Ollama at {base_url} using model {model}...")
    client = AsyncOpenAI(base_url=base_url, api_key="ollama")
    
    messages = [{"role": "user", "content": "Say hello world and then call the dummy tool if available."}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "dummy_tool",
                "description": "A dummy tool for testing",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    ]
    
    try:
        print("Sending request...")
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            stream=True
        )
        
        print("Receiving stream details:")
        full_text = ""
        async for chunk in stream:
            delta = chunk.choices[0].delta
            content = delta.content or ""
            tool_calls = delta.tool_calls
            
            if content:
                print(f"[TEXT] {content}")
                full_text += content
            if tool_calls:
                for tc in tool_calls:
                    print(f"[TOOL] ID: {tc.id}, Name: {tc.function.name}, Args: {tc.function.arguments}")
        
        print(f"\nFinal text: {full_text}")
    except Exception as e:
        print(f"Error during Ollama diagnostic: {e}")

if __name__ == "__main__":
    asyncio.run(diag())
