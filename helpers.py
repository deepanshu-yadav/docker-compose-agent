import os
import re
import subprocess
import json
import requests
from datetime import datetime

def save_code_to_files(text, main_dir):
    if type(text) != type(''):
        return
    # Regular expression to match the pattern <filename> code
    pattern = r'```([^\n]+)\n([\s\S]*?)\n```'
    
    # Find all matches in the text
    matches = re.findall(pattern, text)
    
    for filename, code in matches:
        # Construct the full path
        full_path = os.path.join(main_dir, *tuple(os.path.split(filename)))
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # Write the code to the file
        with open(full_path, 'w') as file:
            file.write(code)

def is_docker_running():
    try:
        subprocess.run(["docker", "info"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError:
        print("Docker not running.")
        return False
    except FileNotFoundError:
        print("Docker not installed.")
        return False  # Docker CLI not installed

def escape_braces(text):
    if type(text) == type(''):
        return text.replace("{", "{{").replace("}", "}}")
    return text

# Function to create log files
def create_log_file(directory, prefix):
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(directory, f"{prefix}_{timestamp}.log")
    return open(log_file, "w")

# Function to read the last 10 lines of a file
def read_last_10_lines(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            return ''.join(lines[-10:])
    return ""

def read_all_lines(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            return ''.join(lines)
    return ""

def read_yaml(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return ""    
        
def generate_error_string(error_file_path, dir):
    error_string = read_all_lines(error_file_path)
    file_contents = read_files_in_directory(dir)

    error = f"""Posting on behalf of user. \n While trying the execute the folloing files
       \n  \n \n {file_contents} \n\n The following errors/ error were recorded \n\n 
      `{error_string}`  \n\n """
    
    return error

def read_files_in_directory(directory):
    result = ""  # Initialize an empty string to store the final output

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Avoid reading stdout and stderr files 
                if "stdout" not in file_path and "stderr" not in file_path: 
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Include the directory path in the file name and append to the result
                        result += f"""```{os.path.relpath(file_path, directory)}\n"""
                        result += f"""\n{content}\n```\n\n"""
            except Exception as e:
                result += f"Could not read file {file_path}: {e}\n\n"

    return result


def is_context_already_present(context_str):
    if "RETRIEVED INFORMATION:" in context_str:
        return True
    return False


def save_llm_output_to_files(text, main_dir):
    # Create the main directory if it doesn't exist
    os.makedirs(main_dir, exist_ok=True)
    
    # Split the text into sections based on code blocks
    sections = re.split(r'```(?:[a-zA-Z0-9]*)?\n', text)
    
    # Process sections in pairs (text before code block + code block content)
    file_mappings = []
    for i in range(0, len(sections)-1, 2):
        context = sections[i].strip()
        if i+1 < len(sections):
            code_content = sections[i+1].strip()
            
            # Skip shell commands that don't create files
            if code_content.startswith(('mkdir', 'cd', 'pip', 'docker-compose')):
                continue
            
            # Look for filename patterns in the context before the code block
            filename = None
            directory = None
            
            # First check for explicit directory mentions
            dir_match = re.search(r'in the ([^\s]+) (?:directory|folder)', context, re.I)
            if dir_match:
                directory = dir_match.group(1).strip('"\'').strip()
            
            # Patterns for finding filenames in various formats
            file_patterns = [
                r'[Cc]reate (?:a )?(?:new )?(?:file|script) (?:name|named|called) ["\']?([^"\'\n]+?)["\']?(?: and copy the following code into it)?$',
                r'[Cc]reate (?:a )?(?:new )?(?:file|script) (?:name|named|called) ["\']?([^"\'\n]+?)["\']?(?: with the following contents)?$',
                r'[Cc]reate (?:a )?(?:new )?(?:file|script) (?:name|named|called) ["\']?([^"\'\n]+?)["\']?(?: in the [^\s]+ directory)?$',
                r'[Cc]reate (?:a )?(?:new )?(?:file|script) ["\']?([^"\'\n]+?)["\']?(?: and copy the following code into it)?$',
                r'(?:file|script) called `([^`]+)`',
                r'named `([^`]+)`',
                r'[Ss]ave this (?:as|to) ["\']?([^"\'\n]+?)["\']?$',
                r'["\']?([A-Za-z0-9_\-\.\/]+\.[a-zA-Z0-9]+)["\']?',
                r'["\']?([A-Za-z0-9_\-\.\/]+)["\']?'
            ]
            
            for pattern in file_patterns:
                matches = re.search(pattern, context)
                if matches:
                    for group in matches.groups():
                        if group and group.strip():
                            filename = group.strip()
                            break
                    if filename:
                        break
            
            if filename:
                filename = re.sub(r'["\'].*', '', filename)
                filename = filename.strip('"\'').strip()
                filename = re.sub(r'[^\w\.\/-].*$', '', filename)
                
                if directory:
                    filename = f"{directory}/{filename}"
                elif '/' in filename or '\\' in filename:
                    pass
                elif 'main directory' in context.lower():
                    pass
                
                if filename == "compose.yml":
                    filename = "docker-compose.yml"
                
                file_mappings.append((filename, code_content))
    
    # Process all found file mappings
    for filename, content in file_mappings:
        # First escape all actual newlines
        content = content.replace('\n', '\\NEWLINE_TEMP\\')
        
        # Then restore literal newlines in strings
        content = content.replace('\\\\NEWLINE_TEMP\\', '\\n')
        content = content.replace('\\NEWLINE_TEMP\\', '\n')
        
        # Create full path with directories
        full_path = os.path.join(main_dir, filename)
        directory = os.path.dirname(full_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
            
        # Write content to file
        with open(full_path, 'w') as f:
            f.write(content)
            
        print(f"Saved content to {full_path}")