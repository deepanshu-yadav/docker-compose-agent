import subprocess
import sys
import threading
import time

def print_output(process, label):
    """Print output from a process with labels"""
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(f"[{label}] {output.strip()}")
    process.stdout.close()

def run():
    # Start MCP server
    mcp = subprocess.Popen(
        [sys.executable, "mcp.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,  # This makes stdout return strings instead of bytes
        universal_newlines=True
    )
    
    # Start Streamlit
    streamlit = subprocess.Popen(
        ["streamlit", "run", "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        universal_newlines=True
    )
    
    # Create threads to print outputs
    mcp_thread = threading.Thread(target=print_output, args=(mcp, "MCP"))
    streamlit_thread = threading.Thread(target=print_output, args=(streamlit, "STREAMLIT"))
    
    mcp_thread.daemon = True  # Thread will exit when main program exits
    streamlit_thread.daemon = True
    
    mcp_thread.start()
    streamlit_thread.start()
    
    try:
        while mcp_thread.is_alive() and streamlit_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        mcp.terminate()
        streamlit.terminate()
        mcp.wait()
        streamlit.wait()

if __name__ == "__main__":
    print("Starting services (Ctrl+C to stop)")
    print("MCP server output prefixed with [MCP]")
    print("Streamlit output prefixed with [STREAMLIT]\n")
    run()