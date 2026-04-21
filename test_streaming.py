import asyncio
import websockets
import json
import base64
import time

async def test_streaming():
    url = "ws://localhost:8769/voice/stream"
    try:
        async with websockets.connect(url) as websocket:
            print("Connected to server")
            
            # Send setup (config)
            setup = {
                "type": "config",
                "voice_prompt": "ref_12s",
                "context_id": "test_ctx"
            }
            await websocket.send(json.dumps(setup))
            
            # Wait for ready
            resp = await websocket.recv()
            print(f"Server response: {resp}")
            
            # Send synthesize
            synth = {
                "type": "synthesize",
                "text": "Hello, this is a test of the Qwen 3 TTS streaming server. I am now speaking in real time with low latency.",
                "context_id": "test_ctx"
            }
            await websocket.send(json.dumps(synth))
            
            start_time = time.time()
            first_chunk_received = False
            total_chunks = 0
            
            async for message in websocket:
                data = json.loads(message)
                if data["type"] == "audio":
                    if not first_chunk_received:
                        latency = (time.time() - start_time) * 1000
                        print(f"First chunk received! Latency: {latency:.2f} ms")
                        first_chunk_received = True
                    total_chunks += 1
                    # print(f"Received audio chunk {total_chunks}")
                elif data["type"] == "final":
                    print(f"Stream finished. Total chunks: {total_chunks}")
                    break
                elif data["type"] == "error":
                    print(f"Error: {data['message']}")
                    break
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    # Note: This requires the server to be running.
    # We will try to start the server in the background and run this.
    asyncio.run(test_streaming())
