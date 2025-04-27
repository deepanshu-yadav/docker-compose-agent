import os
import re
import json
import shutil
import asyncio
import httpx
import time 
from typing import List
from typing import Dict, Any
from pathlib import Path
from pydantic import BaseModel
from fastapi import Query
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import ChatOpenAI

from configs import *

# ---- Pydantic Models ---- #
class FileOutput(BaseModel):
    filename: str
    content: str

class GraphState(BaseModel):
    question: str
    context: str = ""
    raw_response: str = ""
    files: List[FileOutput] = []
    error: str = ""
    retry_count: int = 0
    last_error: str = ""

# ---- Nodes ---- #
class ExecutionEngine:
    def __init__(
        self,
        directory: str = None,
        selected_model: str = None,
        model_type: str = "online",
        server_url: str = None,
        api_key: str = None,
        chat_id: str = None
    ):
        self.api_key = api_key
        self.question = None
        self.selected_model = selected_model
        self.model_type = model_type
        self.directory = directory
        self.server_url = server_url
        self.chat_id = chat_id
        self.execution_done = True
     # Initialize response file if it doesn't exist
        self._initialize_response_file()
        self.max_retry_count = 3
        self.retry_count = 1
    
    def _get_response_file_path(self):
        """Get the constant path to the response file"""
        return os.path.join(self.directory, CONFIG["response_filename"])
    
    def _initialize_response_file(self):
        """Initialize the response JSON file if it doesn't exist"""
        if not os.path.exists(self.directory):
            os.makedirs(self.directory, exist_ok=True)
            
        response_file = self._get_response_file_path()
        if os.path.exists(response_file):
            # Delete existing file if it exists
            try:
                os.remove(response_file)
                print(f"Deleted existing file: {response_file}")
            except Exception as e:
                print(f"Error deleting file {response_file}: {e}")
            
        initial_data = {
            "meta_data": {
                "created_at": time.time(),
                "selected_model": self.selected_model,
                "model_type": self.model_type,
                "version": "1.0"
            },
            "status": {
                "awaiting_response": True,
                "execution_complete": False,
            },
            "chats": {}
        }
        
        try:
            with open(response_file, 'w') as f:
                json.dump(initial_data, f, indent=2)
        except Exception as e:
            print(f"Error initializing response file {response_file}: {e}")
    
    def _read_response_data(self) -> Dict[str, Any]:
        """Read the current data from the response file"""
        response_file = self._get_response_file_path()
        try:
            with open(response_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            # If file is corrupted or doesn't exist, reinitialize it
            print(f"Error reading response file {response_file}: {e}")
            self.retry_count += 1
            if self.retry_count > self.max_retry_count:
                print(f"Max retry count {self.retry_count} reached. Exiting.")
                return {}
            time.sleep(3)  # Wait a bit before reinitializing
            return self._read_response_data()
    
    def _write_response_data(self, data: Dict[str, Any]):
        """Write data to the response file"""
        response_file = self._get_response_file_path()
        with open(response_file, 'w') as f:
            json.dump(data, f, indent=2)
            
    def update_chat(self, message: str, final: bool = False):
        if self.chat_id:
            data = self._read_response_data()
            
            # Ensure the chat exists
            if self.chat_id not in data["chats"]:
                data["chats"][self.chat_id] = {
                    "messages": [],
                    "created_at": time.time(),
                    "last_updated": time.time()
                }
            
            # Update chat metadata
            data["chats"][self.chat_id]["last_updated"] = time.time()
            
            # Append message to chat history
            data["chats"][self.chat_id]["messages"].append({
                "role": "ai", 
                "content": message,
                "timestamp": time.time()
            })
            
            if final:
                data["status"]["awaiting_response"] = False
                data["status"]["execution_complete"] = True
                data["meta_data"]["last_completion_time"] = time.time()
            
            # Write updated data back to file
            self._write_response_data(data)
                
    def get_llm(self):
        if self.model_type == "online":
            try:
                llm_engine = ChatOpenAI(
                    base_url=OPEN_ROUTER_URL,
                    api_key=self.api_key,
                    model=self.selected_model,
                    temperature=0
                )
                return llm_engine
            except Exception as e:
                print(f"API Error: {e}")
                return {"error": str(e)}
        else:
            try:
                llm_engine = ChatOllama(
                    model=self.selected_model,
                    base_url=OLLAMA_URL,
                    temperature=0
                )
                return llm_engine
            except Exception as e:
                print(f"API Error: {e}")
                return {"error": str(e)}

    async def get_context_node(self, state: GraphState):
        print("Fetching context from RAG...")
        self.update_chat("# Fetching context from RAG...")
        from rag import initialize_rag, get_context
        initialize_rag()
        ctx = get_context(state.question)
        self.update_chat("# Context from RAG\n\n" + json.dumps(ctx))
        return {"context": ctx}

    async def generate_code_node(self, state: GraphState):
        print("Generating code from LLM")
        llm_engine = self.get_llm()
        if isinstance(llm_engine, dict):
            return llm_engine
        prompt = f"""
You are a code generator. Given the context and a user question, generate code files.
Context:
{state.context}

Question:
{state.question}

Format output like:
Create a file named app.py:
print("hello world")

Or

Create a file named web/app.py:
print("hello world")

Always include a valid docker-compose.yml with correct syntax and necessary services.
"""
        if state.retry_count > 0 and state.last_error:
            prev_response = state.raw_response
            if len(prev_response) > MAX_RESPONSE_LENGTH:
                prev_response = prev_response[:MAX_RESPONSE_LENGTH] + "..."
                print(f"Previous response truncated to {MAX_RESPONSE_LENGTH} chars due to length.")
            prompt += f"""
Previous attempt failed with the following error from `docker compose up`:
{state.last_error}

The previous LLM-generated response was:
{prev_response}
Please analyze the error and the previous response, then generate corrected code files to resolve the issue. Ensure the docker-compose.yml is valid and includes all necessary services, ports, and volumes. Prioritize fixing the file(s) causing the error.
"""
        self.update_chat("# Generating code from LLM...\n\n" + prompt)
        response = await llm_engine.ainvoke(prompt)
        print(f"LLM response received:\n{response.content.strip()[:500]}...")
        self.update_chat("# LLM response received:\n\n" + response.content)
        return {
            "raw_response": response.content.strip(),
            "retry_count": state.retry_count + 1
        }

    async def parse_files_with_llm(self, state: GraphState):
        print("Parsing files from LLM output using an LLM")
        llm_parser = self.get_llm()
        if isinstance(llm_parser, dict):
            return llm_parser
        parser_prompt = f"""
Extract all filenames and their contents from the following LLM output. Return a **JSON array** of objects like:
{{"filename": "app.py", "content": "..."}}.
Only respond with the JSON array — nothing else.

LLM Output:
{state.raw_response}
"""
        self.update_chat("# Parsing files from LLM output...\n\n" + parser_prompt)
        output = await llm_parser.ainvoke(parser_prompt)
        raw_content = output.content.strip()
        print("LLM Parser Output:\n", raw_content[:500])
        self.update_chat("# LLM File Parser Output:\n\n" + raw_content)
        try:
            parsed = json.loads(raw_content)
            files = [FileOutput(**f) for f in parsed]
            return {"files": files}
        except Exception as e:
            print(f"Parsing failed: {e}")
            return {"error": f"Parsing failed: {e}\nLLM output:\n{raw_content}"}

    async def save_files(self, state: GraphState):
        try:
            print("Saving files to disk...")
            self.update_chat("# Saving files to disk...")
            if os.path.exists(self.directory):
                for item in os.listdir(self.directory):
                    item_path = os.path.join(self.directory, item)
                    # Skip the response file
                    if item_path == self._get_response_file_path():
                        continue
                    try:
                        # If it's a file, delete it
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                        # If it's a directory, delete it recursively
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    except Exception as e:
                        print(f"Error deleting {item_path}: {e}")
                        
            os.makedirs(self.directory, exist_ok=True)
            for file in state.files:
                filepath = os.path.join(self.directory, file.filename)
                filepath = os.path.normpath(filepath)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(file.content)
            print(f"Files saved to: {self.directory}")
            saved_message = f"# Files saved to disk: in {self.directory} \n\n" + str([f.filename for f in state.files])
            self.update_chat(saved_message)
            return {}
        except Exception as e:
            print(f"Error saving files: {e}")
            return {"error": f"Error saving files: {e}"}

    async def run_docker_compose(self, state: GraphState):
        print("Running docker-compose via MCP server...")
        abs_directory = os.path.abspath(self.directory)
        if not os.path.isdir(abs_directory):
            os.makedirs(abs_directory, exist_ok=True)
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                print(f"Sending request to {self.server_url}/run_compose/up?directory={abs_directory}")
                self.update_chat("# Running docker-compose via MCP server...")
                resp = await client.get(f"{self.server_url}/run_compose/up", params={"directory": abs_directory})
                print(f"Response status: {resp.status_code}")
                if resp.status_code == 200:
                    error_detected = False
                    output = []
                    return_code = None
                    async for chunk in resp.aiter_text():
                        print(chunk, end="")
                        output.append(chunk)
                        if "error" in chunk.lower():
                            error_detected = True
                        if match := re.search(r"Process exited with code (\d+)", chunk):
                            return_code = int(match.group(1))
                    print("\nCompose output streamed.")
                    full_output = "".join(output)
                    if return_code is not None and return_code != 0:
                        error_msg = f"Process failed with return code {return_code}: {full_output}"
                        print(f"Error in docker compose: {error_msg}")
                        self.update_chat("# Error in docker compose:\n\n" + error_msg)
                        return {"error": error_msg, "last_error": error_msg}
                    if error_detected:
                        error_msg = f"Error detected in output: {full_output}"
                        print(error_msg)
                        self.update_chat("# Error detected in output:\n\n" + error_msg)
                        return {"error": error_msg, "last_error": error_msg}
                    print("Docker compose executed successfully.")
                    success_message = f"# Docker compose executed successfully. in {self.directory} \n\n" + full_output 
                    self.update_chat(success_message, final=True)
                    return {"error": ""}
                else:
                    error_msg = f"HTTP {resp.status_code}: {resp.text}"
                    print(error_msg)
                    self.update_chat("# Error: " + error_msg)
                    return {"error": error_msg, "last_error": error_msg}
            except httpx.TimeoutException:
                error_msg = "Request timed out after 30 seconds"
                print(error_msg)
                self.update_chat("# Request timed out after 30 seconds")
                return {"error": error_msg, "last_error": error_msg}
            except httpx.RequestError as e:
                error_msg = f"Network error: {str(e)}"
                print(error_msg)
                self.update_chat("# Network error: " + str(e))
                return {"error": error_msg, "last_error": error_msg}

    def check_success(self, state: GraphState):
        print("\n=== Checking Workflow Success ===")
        print(f"Current state:")
        print(f"  Error: {state.error}")
        print(f"  Retry count: {state.retry_count}")
        print(f"  Last LLM response (first 500 chars): {state.raw_response[:500]}...")
        print(f"  Files generated: {[f.filename for f in state.files]}")
        
        if state.error:
            print(f"Error detected: {state.error}")
            print(f"Retry attempt {state.retry_count + 1}/3")
            if state.retry_count >= 3:
                print("Maximum retry attempts reached. Terminating workflow.")
                self.execution_done = True
                self.update_chat("# Maximum retry attempts reached. Workflow terminated.", final=True)
                return END
            return "Generate"
        print("Docker compose executed successfully. Ending workflow.")
        self.execution_done = True
        return END

    async def construct_graph(self, question: str = None):
        self.question = question
        self.execution_done = False
        workflow = StateGraph(GraphState)
        workflow.add_node("RAG", self.get_context_node)
        workflow.add_node("Generate", self.generate_code_node)
        workflow.add_node("Parse", self.parse_files_with_llm)
        workflow.add_node("Save", self.save_files)
        workflow.add_node("Run", self.run_docker_compose)
        workflow.add_conditional_edges("Run", self.check_success, {
            "Generate": "Generate",
            END: END
        })

        workflow.set_entry_point("RAG")
        workflow.add_edge("RAG", "Generate")
        workflow.add_edge("Generate", "Parse")
        workflow.add_edge("Parse", "Save")
        workflow.add_edge("Save", "Run")

        app = workflow.compile()
        final_state = await app.ainvoke({"question": self.question})
        print(f"\nFinal state:\n\n{final_state}")
        self.execution_done = True
        return final_state
      
# # ---- CLI Test  Example usage ---- #
# if __name__ == "__main__":
# #     # ---- CONFIG ---- #
#     MCP_SERVER = f"http://{MCP_IP}:{MCP_PORT}"
    
#     # Define model and key for OpenRouter
#     # selected_model = "microsoft/mai-ds-r1:free"
#     selected_model = "llama3.2:3b"  # Example model name
#     question = "Build a simple Flask app with logging and Docker"
#     engine = ExecutionEngine(
#         directory=SAVE_DIR,
#         selected_model=selected_model,
#         model_type="offline",
#         server_url=MCP_SERVER,
#         api_key=DEEPSEEK_FREE_KEY
#     )
    
#     asyncio.run(engine.construct_graph(question=question))