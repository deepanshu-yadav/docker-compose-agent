# Set environment variables before any imports
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# System modules
import sys
import re
import threading
import shutil
import time
import platform
import subprocess
from datetime import datetime
import uuid

# Configure asyncio for Windows
if platform.system() == "Windows":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Define constants
MAIN_WORKSPACE_DIR = os.path.join(os.getcwd(), "workspaces")
DEEPSEEK_FREE_KEY = os.environ.get("DEEPSEEK_FREE_KEY")

# Now import streamlit and other non-torch libraries
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
from pydantic import BaseModel, Field
from typing import Optional, List, Mapping, Any
from langchain.llms.base import LLM
from openai import OpenAI

# Import your local modules
from prompts import MAIN_PROMPT, SHORT_PROMPT
from helpers import *

# Import the rag module last
from rag import initialize_rag, get_context

initialize_rag()
print(DEEPSEEK_FREE_KEY)

# Page config
st.set_page_config(
    page_title="DeepSeek Devops Agent",
    page_icon="🧠",
    layout="wide"
)

# Add your existing CSS
st.markdown("""
<style>
    /* Main theme */
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
        background-color: #2d2d2d !important;
        border-color: #4d4d4d !important;
    }
    
    /* Select box styling */
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    .stSelectbox svg {
        fill: white !important;
    }
    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    
    /* Chat history styling */
    .chat-history-item {
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        background-color: #2d2d2d;
        border: 1px solid #4d4d4d;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .chat-history-item:hover {
        background-color: #3d3d3d;
        border-color: #6d6d6d;
    }
    
    /* Button styling */
    .stButton button {
        width: 100%;
        background-color: #3d3d3d;
        color: white;
        border: 1px solid #4d4d4d;
        padding: 10px;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #4d4d4d;
        border-color: #6d6d6d;
    }
    
    /* Chat container */
    .chat-container {
        border-radius: 10px;
        background-color: #2d2d2d;
        padding: 20px;
        margin: 10px 0;
    }
    
    /* Message styling */
    .user-message {
        background-color: #3d3d3d;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .ai-message {
        background-color: #2d2d2d;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session states
if 'chats' not in st.session_state:
    new_chat_id = str(uuid.uuid4())
    st.session_state.chats = {
        new_chat_id: {
            'messages': [{"role": "ai", "content": "Hi! I'm a Devops Agent. How can I help you deploy today? 💻"}],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'title': "New Chat"
        }
    }
    st.session_state.current_chat_id = new_chat_id

if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]

if "awaiting_response" not in st.session_state:
    st.session_state.awaiting_response = False

# Chat management functions
def create_new_chat():
    new_chat_id = str(uuid.uuid4())
    st.session_state.chats[new_chat_id] = {
        'messages': [{"role": "ai", "content": "HHi! I'm a Devops Agent. How can I help you deploy today? 💻"}],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'title': "New Chat"
    }
    st.session_state.current_chat_id = new_chat_id
    st.session_state.awaiting_response = False

def switch_chat(chat_id):
    st.session_state.current_chat_id = chat_id
    st.session_state.awaiting_response = False

def get_chat_title(messages):
    for msg in messages:
        if msg['role'] == 'user':
            return msg['content'][:30] + "..." if len(msg['content']) > 30 else msg['content']
    return "New Chat"

# Initialize session state variables for Docker Compose
if 'output_stdout' not in st.session_state:
    st.session_state.output_stdout = ""
if 'output_stderr' not in st.session_state:
    st.session_state.output_stderr = ""
if 'process' not in st.session_state:
    st.session_state.process = None
if "directory" not in st.session_state:
    st.session_state.directory = "first_project"

# Thread-safe flag to control the while loop
stop_event = threading.Event()

def notify_chat(err_str):
    error_dict = {"role": "user", 
                  "content": escape_braces(err_str)}
    st.session_state.chats[st.session_state.current_chat_id]['messages'].append(error_dict)
    st.rerun()

# Function for running asyncio on windows
def setup_asyncio_for_windows():
    """Configure asyncio to work properly on Windows."""
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Wrapper function for windows compatibility
def run_docker_compose_wrapper(func, *args):
    """Run an asyncio function with proper Windows compatibility."""
    setup_asyncio_for_windows()
    return asyncio.run(func(*args))

# Async function to execute the docker-compose command and capture output
async def run_docker_compose_up(directory, project_name, stdout_placeholder, stderr_placeholder):
    if not is_docker_running():
        st.session_state.output_stderr = "Docker does not seem to be running please run Docker and try again."
        stderr_placeholder.text_area(
            "StdErr Output", 
            st.session_state.output_stderr, 
            height=100, 
            key=f"stderr_output_final_{time.time()}"  # Unique key using timestamp
        )
        return
    
    # Define the command
    command = [
        "docker-compose",
        "-f", os.path.join(directory, "docker-compose.yml"),
        "-p", project_name,
        "up", "--force-recreate"
    ]
    command = ' '.join(command)

    # Define stdout and stderr file paths
    stdout_file = os.path.join(directory, "stdout.log")
    stderr_file = os.path.join(directory, "stderr.log")

    # Open files for writing
    with open(stdout_file, 'w') as stdout_f, open(stderr_file, 'w') as stderr_f:
        # Start the process
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        st.session_state.process = process
        print(f"docker compose up Executed command  {command}")

        # Read output in real-time
        while not stop_event.is_set():
            try:
                if process.returncode is not None:
                    print("Process has terminated. Exiting loop.")
                    # Time to notify the chat about it. 
                    err_str = generate_error_string(stderr_file, directory)
                    notify_chat(err_str)
                    break

                # Read stdout and stderr line by line
                stdout_line = await process.stdout.readline()
                stderr_line = await process.stderr.readline()
                # checks for error and kills the process 

                # Write to files
                if stdout_line:
                    stdout_f.write(stdout_line.decode('utf-8'))
                    stdout_f.flush()
                if stderr_line:
                    stderr_f.write(stderr_line.decode('utf-8'))
                    stderr_f.flush()

                # Update the placeholder with the last 10 lines of stdout and stderr
                stdout_output = read_last_10_lines(stdout_file)
                stderr_output = read_last_10_lines(stderr_file)
                st.session_state.output_stdout = stdout_output
                st.session_state.output_stderr = stderr_output
                
                if "error" in stderr_output.lower():  # Case-insensitive check
                    print("Found 'error' in stderr, terminating process...")
                    err_str = generate_error_string(stderr_file, directory)
                    notify_chat(err_str)
                    process.kill()  # Terminate the process
                    break

                # Use unique keys for text_area widgets
                stdout_placeholder.text_area(
                    "Stdout Output", 
                    st.session_state.output_stdout, 
                    height=100, 
                    key=f"stdout_output_{time.time()}"  # Unique key using timestamp
                )
                stderr_placeholder.text_area(
                    "StdErr Output", 
                    st.session_state.output_stderr, 
                    height=100, 
                    key=f"stderr_output_{time.time()}"  # Unique key using timestamp
                )

                await asyncio.sleep(0.1)  # Small delay to avoid busy-waiting
            except Exception as e:
                print(f"Could not perform polling the process: {e}")
                break

        # Final update after the process finishes
        stdout_output = read_last_10_lines(stdout_file)
        stderr_output = read_last_10_lines(stderr_file)
        st.session_state.output_stdout = stdout_output
        st.session_state.output_stderr = stderr_output
        stdout_placeholder.text_area(
            "Stdout Output", 
            st.session_state.output_stdout, 
            height=100, 
            key=f"stdout_output_final_{time.time()}"  # Unique key using timestamp
        )
        stderr_placeholder.text_area(
            "StdErr Output", 
            st.session_state.output_stderr, 
            height=100, 
            key=f"stderr_output_final_{time.time()}"  # Unique key using timestamp
        )
        print(f"Terminated the docker compose up process stderr {stderr_output} stdout {stdout_output}")


async def run_docker_compose_down(directory, project_name, stdout_placeholder, stderr_placeholder):
    # Define the command
    command = [
        "docker-compose",
        "-f", os.path.join(directory, "docker-compose.yml"),
        "-p", project_name, 
        "down"
    ]
    command = ' '.join(command)

    # Define stdout and stderr file paths
    stdout_file = os.path.join(directory, "stdout.log")
    stderr_file = os.path.join(directory, "stderr.log")

    # Open files for writing
    with open(stdout_file, 'w') as stdout_f, open(stderr_file, 'w') as stderr_f:
        # Start the process
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        st.session_state.process = process
        print(f"Executed command {command}")

        # Wait for the process to complete
        await process.wait()

        # Read the final output
        stdout_output = read_last_10_lines(stdout_file)
        stderr_output = read_last_10_lines(stderr_file)
        st.session_state.output_stdout = stdout_output
        st.session_state.output_stderr = stderr_output

        # Update the placeholders
        stdout_placeholder.text_area(
            "Stdout Output", 
            st.session_state.output_stdout, 
            height=100, 
            key=f"stdout_output_down_{time.time()}"  # Unique key using timestamp
        )
        stderr_placeholder.text_area(
            "StdErr Output", 
            st.session_state.output_stderr, 
            height=100, 
            key=f"stderr_output_down_{time.time()}"  # Unique key using timestamp
        )
        print(f"Terminated the docker compose down process stderr {stderr_output} stdout {stdout_output}")


# Add Docker Compose functionality in a dedicated section or tab
with st.expander("Docker Compose Manager", expanded=True):
    st.write("Manage your Docker containers using docker-compose.")

    # Input box for directory
    st.session_state.directory = st.text_input(
        "Enter the directory containing docker-compose.yml",
        value=st.session_state.directory,
        key="docker_dir_input"
    )
    workspace_dir = os.path.join(MAIN_WORKSPACE_DIR, st.session_state.directory)
    project_name = st.session_state.directory
    # Buttons for docker-compose up and down
    col1, col2 = st.columns(2)
     # Placeholder for displaying the output
    stdout_placeholder = st.empty()
    stderr_placeholder = st.empty()

    with col1:
        if st.button("Start Docker-Compose"):
            if os.path.exists(workspace_dir) and os.path.exists(os.path.join(workspace_dir, "docker-compose.yml")):
                stop_event.clear()  # Reset the stop event
                run_docker_compose_wrapper(run_docker_compose_up, workspace_dir,
                                           project_name, stdout_placeholder, stderr_placeholder)
            else:
                st.error("Invalid workspace directory or docker-compose.yml not found.")
    with col2:
        if st.button("Stop Docker-Compose"):
            if os.path.exists(workspace_dir) and os.path.exists(os.path.join(workspace_dir, "docker-compose.yml")):
                stop_event.set()  # Signal the while loop to stop
                run_docker_compose_wrapper(run_docker_compose_down, workspace_dir, project_name,
                                           stdout_placeholder, stderr_placeholder)
            else:
                st.error("Invalid workspace directory or docker-compose.yml not found.")

# Sidebar layout
with st.sidebar:
    st.title("🧠 Devops Agent")
    
    # Configuration section
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek/deepseek-r1-zero:free","deepseek-r1:32b","deepseek-r1:7b", "deepseek-r1:1.5b"],
        index=0
    )
    
    st.divider()
    
    # Model capabilities section
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Docker Compose Expert
    - 🐞 Generates docker-compose solution
    - 📝 Deploys them for you
    - 💡 Checks for error and regnerate and redeploys
    """)
    
    st.divider()
    
    # Chat history section
    st.markdown("### 💬 Chat History")
    
    # New chat button
    if st.button("➕ New Chat", key="new_chat", use_container_width=True):
        create_new_chat()
    
    # Display chat history
    for chat_id, chat_data in sorted(
        st.session_state.chats.items(),
        key=lambda x: x[1]['timestamp'],
        reverse=True
    ):
        chat_title = get_chat_title(chat_data['messages'])
        
        # Create a clickable chat history item
        if st.button(
            f"💬 {chat_title}",
            key=f"chat_{chat_id}",
            help=f"Created: {chat_data['timestamp']}",
            use_container_width=True
        ):
            switch_chat(chat_id)
    
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

# Main chat interface
st.title("AI Devops Agent")
st.caption("🚀 Your AI Devops expert with deploying superpowers.")

# Initialize LLM engine
if selected_model == "deepseek/deepseek-r1-zero:free":
    # Set up the OpenAI client with the custom API endpoint
    
    llm_engine = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=DEEPSEEK_FREE_KEY,
    )
else:
    llm_engine = ChatOllama(
        model=selected_model,
        base_url="http://localhost:11434",
        temperature=0
    )

def post_result_text(original_text, file_path):
    full_path =os.path.join(file_path, "docker-compose.yml")
    text = f"""\n\n Since this a smart agent so I have already saved the yaml  file at \n `{full_path}`. 
    If you press the docker compose up button you can see the output and error in the docker manager view.
     \n And do not worry I will notify in the chat if it does not execute. Enjoy!! \n"""
    text = escape_braces(text)
    if type(original_text) != type(''):
        return text
    return original_text + text

# System prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template(escape_braces(MAIN_PROMPT))

def generate_ai_response(prompt_chain):
    if selected_model == "deepseek/deepseek-r1-zero:free":
        try:
            # Convert the prompt chain to a string
            # Use the OpenAI client to generate a response
            completion = llm_engine.chat.completions.create(
                model=selected_model,
                messages=prompt_chain
            )
            # Extract and return the AI's response
            return completion.choices[0].message.content
        except Exception as e:
            print(f"API Error: {e}")
            return None
    else:
        processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
        return processing_pipeline.invoke({})

def build_prompt_chain():
    
    if selected_model == "deepseek/deepseek-r1-zero:free":
        messages = [{"role": "system", "content":escape_braces(SHORT_PROMPT) }]
        for msg in st.session_state.chats[st.session_state.current_chat_id]['messages']:
            msg["content"] = escape_braces(msg["content"])
            messages.append(msg)
        return messages
    else:
        prompt_sequence = [system_prompt]
        for msg in st.session_state.chats[st.session_state.current_chat_id]['messages']:
            msg["content"] = escape_braces(msg["content"])
            if msg["role"] == "user":
                prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
            elif msg["role"] == "ai":
                prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
        return ChatPromptTemplate.from_messages(prompt_sequence)

# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.chats[st.session_state.current_chat_id]['messages']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Show processing message if waiting for response
    if st.session_state.awaiting_response:
        with st.chat_message("ai"):
            with st.spinner("🧠 Processing..."):
                prompt_chain = build_prompt_chain()
                ai_response = generate_ai_response(prompt_chain)
                ai_response = escape_braces(ai_response)
                save_path = os.path.join(MAIN_WORKSPACE_DIR, st.session_state.directory)
                save_code_to_files(ai_response, save_path)
                new_ai_response = post_result_text(ai_response, save_path)
                st.session_state.chats[st.session_state.current_chat_id]['messages'].append(
                    {"role": "ai", "content": new_ai_response}
                )
                st.session_state.awaiting_response = False
                st.rerun()

# Chat input
user_query = st.chat_input("Type your devops problem here...")

if user_query:
    # Add user message to log
    st.session_state.chats[st.session_state.current_chat_id]['messages'].append(
        {"role": "user", "content": user_query}
    )
    st.session_state.awaiting_response = True
    st.rerun()