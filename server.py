from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import os
import argparse
import sys

# Add current dir to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from streaming_service import StreamingTalker

app = FastAPI()

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Global Talker instance (lazy loaded or startup)
talker = None

@app.on_event("startup")
async def startup_event():
    global talker
    # Try to find model path from env or args
    # For now, we look for a reasonable default or expect an env var
    model_path = os.getenv("GT_MODEL_PATH")
    ds_path = os.getenv("GT_DS_PATH")
    
    if not model_path:
        # Try to find a model in 'output/'
        output_dir = "output"
        if os.path.exists(output_dir):
            subdirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
            if subdirs:
                model_path = subdirs[0]
                print(f"[AUTO] Picking first model found: {model_path}")
    
    if not model_path:
        print("[ERROR] No model_path found. Set GT_MODEL_PATH env var.")
        # We don't exit to allow api to start, but it will fail on connection
        return

    try:
        talker = StreamingTalker(model_path, ds_path)
        print("[INFO] StreamingTalker initialized!")
    except Exception as e:
        print(f"[ERROR] Failed to init talker: {e}")

@app.get("/")
async def get():
    with open(os.path.join(static_dir, "index.html")) as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected")
    
    if not talker:
        await websocket.close(code=1011, reason="Server not initialized (Model not found)")
        return

    # Reset talker state for new session (simple approach)
    # Ideally talker should be stateless or session-based.
    # Current implementation is stateful (buffers audio).
    # We should probably instantiate a new session helper per connection, 
    # but the Reference Data (Model) is shared.
    
    # For now, we just clear buffer manually or assume single user
    talker.audio_buffer = talker.audio_buffer[:0] 
    talker.frame_counter = 0
    
    try:
        while True:
            # Receive audio chunk (bytes)
            data = await websocket.receive_bytes()
            
            # Process
            # This might block the event loop if heavy. 
            # In production, use run_in_executor
            frames = talker.process_chunk(data)
            
            # Send back frames
            for frame_b64 in frames:
                await websocket.send_text(frame_b64)
                
    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
