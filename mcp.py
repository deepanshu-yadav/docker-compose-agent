from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse
import os
import subprocess
import logging
import asyncio
import platform
import traceback
import concurrent.futures
import threading
from typing import Optional

from configs import *
app = FastAPI()

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mcp")

# Thread-safe flag to control the while loop
stop_event = threading.Event()

# Active processes tracker
active_processes = {}  # format: {process_id: {"process": process_obj, "project": project_name, "action": action}}
# Thread pool for running Windows subprocesses
thread_pool = concurrent.futures.ThreadPoolExecutor()

def setup_asyncio_for_windows():
    """Setup asyncio for Windows if needed."""
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Call this early to ensure proper event loop policy
setup_asyncio_for_windows()

def find_compose_file(directory: str) -> Optional[str]:
    """Find the docker compose file in the given directory."""
    for name in ["compose.yaml", "docker-compose.yaml", "docker-compose.yml"]:
        candidate = os.path.join(directory, name)
        logger.debug(f"Looking for compose file: {candidate}")
        if os.path.exists(candidate):
            logger.info(f"Found compose file: {candidate}")
            return candidate
    logger.warning(f"No compose file found in directory: {directory}")
    return None

def run_process_sync(command, cwd):
    """Run a process synchronously and return it."""
    try:
        logger.debug(f"Creating subprocess with command: {command}")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            cwd=cwd,
            universal_newlines=True,
            bufsize=1  # Line buffered
        )
        return process
    except Exception as e:
        logger.error(f"Error creating subprocess: {e}")
        raise

async def stream_logs_from_process(process, process_id):
    """Stream logs from a process."""
    try:
        # Stream output line by line
        loop = asyncio.get_running_loop()
        
        while not stop_event.is_set():
            if process.poll() is not None:
                break
                
            # Use run_in_executor to read lines without blocking event loop
            try:
                line_future = loop.run_in_executor(None, process.stdout.readline)
                
                # Wait for line with timeout so we can check stop_event periodically
                try:
                    line = await asyncio.wait_for(line_future, timeout=0.5)
                except asyncio.TimeoutError:
                    # No output received within timeout, check if we should stop
                    if stop_event.is_set():
                        logger.info(f"Stop event detected for process {process_id}")
                        break
                    continue
                
                if line:
                    line_str = line.rstrip() if isinstance(line, bytes) else line.rstrip()
                    logger.debug(f"[{process_id}] {line_str}")
                    yield f"{line}"
                else:
                    # Small sleep to prevent CPU spinning
                    await asyncio.sleep(0.1)
            except Exception as line_error:
                logger.error(f"Error reading line from process {process_id}: {line_error}")
                await asyncio.sleep(0.5)
        
        # If we broke out of the loop due to stop event, terminate the process
        if stop_event.is_set() and process.poll() is None:
            logger.info(f"Terminating process {process_id} due to stop event")
            process.terminate()
            await asyncio.sleep(2)
            if process.poll() is None:
                logger.info(f"Force killing process {process_id}")
                process.kill()
        
        return_code = process.poll()
        logger.info(f"Process {process_id} exited with code {return_code}")
        
        # Return the exit code as part of the streamed output
        yield f"\nProcess exited with code {return_code}\n"
        
        # Store the return code for later retrieval
        if process_id in active_processes:
            active_processes[process_id]["return_code"] = return_code
            
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error streaming logs: {e}\n{error_details}")
        yield f"\nError streaming logs: {str(e)}\n"

async def stream_logs_async(command, cwd, process_id, project_name, action):
    """Stream logs asynchronously from the docker compose process."""
    try:
        logger.info(f"Starting process {process_id} with command: {command}")
        
        # Reset stop event before starting new process
        stop_event.clear()
        
        # Windows-compatible approach using thread pool
        loop = asyncio.get_running_loop()
        process = await loop.run_in_executor(thread_pool, run_process_sync, command, cwd)
        
        # Store process with more metadata
        active_processes[process_id] = {
            "process": process,
            "project": project_name,
            "action": action,
            "return_code": None
        }
        
        # Return a generator that streams the output
        async for line in stream_logs_from_process(process, process_id):
            yield line
            
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error in process {process_id}: {e}\n{error_details}")
        yield f"\nError: {str(e)}\n"
        
    finally:
        # Clean up on completion or error
        if process_id in active_processes:
            try:
                # Check if process is still running
                if active_processes[process_id]["process"].poll() is None:
                    logger.info(f"Terminating process {process_id} in finally block")
                    active_processes[process_id]["process"].terminate()
                    # Give it a moment to terminate gracefully
                    await asyncio.sleep(2)
                    # Force kill if still running
                    if active_processes[process_id]["process"].poll() is None:
                        active_processes[process_id]["process"].kill()
            except Exception as kill_error:
                logger.error(f"Error cleaning up process {process_id}: {kill_error}")
            
            # Mark as completed
            active_processes[process_id]["completed"] = True

def find_up_process_for_project(project_name):
    """Find any running 'up' process for the given project."""
    for pid, process_info in active_processes.items():
        if (process_info["project"] == project_name and 
            process_info["action"] == "up" and 
            "completed" not in process_info and
            process_info["process"].poll() is None):
            return pid, process_info
    return None, None

async def terminate_process_by_id(process_id):
    """Helper to terminate a process by ID with proper cleanup."""
    if process_id not in active_processes:
        logger.warning(f"Process {process_id} not found for termination")
        return False
    
    try:
        logger.info(f"Terminating process {process_id}")
        
        # Set the global stop event
        stop_event.set()
        
        process = active_processes[process_id]["process"]
        if process.poll() is None:
            process.terminate()
            # Give it time to shut down gracefully
            await asyncio.sleep(2)
            # Force kill if still running
            if process.poll() is None:
                process.kill()
            
            # Mark as completed but don't remove from active_processes yet
            active_processes[process_id]["completed"] = True
            
            # Store return code if available
            if process.poll() is not None:
                active_processes[process_id]["return_code"] = process.returncode
                
            return True
    except Exception as e:
        logger.error(f"Error terminating process {process_id}: {e}")
    
    return False

@app.get("/run_compose/{action}")
async def run_compose(action: str, directory: str = Query(...)):
    """
    Run docker compose command and stream the output.
    
    Args:
        action: Either 'up' or 'down'
        directory: Directory containing the docker-compose file
    """
    try:
        logger.info(f"Received request to run compose {action} in {directory}")
        
        # Check if directory exists
        if not os.path.isdir(directory):
            logger.error(f"Directory does not exist: {directory}")
            raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")
        
        # Validate action
        if action not in ["up", "down"]:
            logger.error(f"Invalid action: {action}")
            raise HTTPException(status_code=400, detail="Invalid action. Use 'up' or 'down'.")
        
        # Find compose file
        compose_file = find_compose_file(directory)
        if not compose_file:
            logger.error(f"No compose file found in directory: {directory}")
            raise HTTPException(status_code=404, detail="No compose file found in the given directory.")
        
        # Create unique process ID
        project_name = os.path.basename(os.path.abspath(directory))
        process_id = f"{project_name}-{action}-{id(directory)}"
        
        logger.info(f"Process ID: {process_id}")
        logger.info(f"Project Name: {project_name}")
        
        # If this is a 'down' command, first terminate any related 'up' process
        if action == "down":
            up_process_id, up_process_info = find_up_process_for_project(project_name)
            if up_process_id:
                logger.info(f"Found running 'up' process {up_process_id} for project {project_name}, terminating it first")
                
                # Signal stop via global event
                stop_event.set()
                
                # Wait a moment for process to notice the stop event
                await asyncio.sleep(0.5)
                
                # Now terminate the process
                await terminate_process_by_id(up_process_id)
                
                # Give a moment for cleanup to complete and reset stop event
                await asyncio.sleep(1)
                stop_event.clear()
        
        # Check if process already exists with same ID and kill it
        if process_id in active_processes and "completed" not in active_processes[process_id]:
            # Signal stop via global event
            stop_event.set()
            await asyncio.sleep(0.5)
            await terminate_process_by_id(process_id)
            stop_event.clear()
        
        # Prepare command
        command = f"docker compose -f \"{compose_file}\" -p \"{project_name}\" {action}"
        logger.info(f"Command: {command}")
        if action == "up":
            command += " --force-recreate"
            command += " -d" # Run in detached mode
        
        # Return streaming response
        return StreamingResponse(
            stream_logs_async(command, directory, process_id, project_name, action),
            media_type="text/plain"
        )
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error in run_compose endpoint: {e}\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/status")
async def get_status():
    """Get the status of all running processes."""
    return {
        "active_processes": [
            {
                "id": pid,
                "project": info["project"],
                "action": info["action"],
                "running": info["process"].poll() is None,
                "return_code": info["process"].returncode if info["process"].poll() is not None else info.get("return_code"),
                "completed": info.get("completed", False)
            }
            for pid, info in active_processes.items()
        ],
        "stop_event_set": stop_event.is_set()
    }

@app.post("/terminate/{process_id}")
async def terminate_process(process_id: str):
    """Terminate a running process by ID."""
    if process_id not in active_processes:
        raise HTTPException(status_code=404, detail=f"Process {process_id} not found")
    
    # Set stop event before terminating
    stop_event.set()
    success = await terminate_process_by_id(process_id)
    stop_event.clear()
    
    if success:
        return {"status": "success", "message": f"Process {process_id} terminated"}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to terminate process {process_id}")

@app.get("/process_result/{process_id}")
async def get_process_result(process_id: str):
    """Get the result of a process including its return code."""
    if process_id not in active_processes:
        raise HTTPException(status_code=404, detail=f"Process {process_id} not found")
    
    process_info = active_processes[process_id]
    return {
        "id": process_id,
        "project": process_info["project"],
        "action": process_info["action"],
        "running": process_info["process"].poll() is None,
        "return_code": process_info["process"].returncode if process_info["process"].poll() is not None else process_info.get("return_code"),
        "completed": process_info.get("completed", False)
    }

@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {
        "status": "running", 
        "active_processes": len(active_processes),
        "stop_event_set": stop_event.is_set()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=MCP_HOST, port=MCP_PORT)