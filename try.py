import os
import re

def save_llm_output_to_files(text, main_dir):
    # Create the main directory if it doesn't exist
    os.makedirs(main_dir, exist_ok=True)
    
    # Split the text into sections based on code blocks
    sections = re.split(r'```', text)
    
    # Process sections in pairs (text before code block + code block content)
    file_mappings = []
    for i in range(0, len(sections)-1, 2):
        context = sections[i]
        if i+1 < len(sections):
            code_content = sections[i+1].strip()
            
            # Skip shell commands that don't create files
            if code_content.startswith('mkdir') or code_content.startswith('cd') or \
               code_content.startswith('pip') or code_content.startswith('docker-compose'):
                continue
            
            # Look for filename patterns in the context before the code block
            filename = None
            
            # Patterns for finding filenames in various formats
            file_patterns = [
                # Pattern for "Create a file named X:" or similar
                r'[Cc]reate (?:a )?file (?:name|named|called) (?:"|\')?([^:"\'\n]+)(?:"|\')?',
                
                # Pattern for filenames in Markdown headers like "Create a **`Dockerfile`"
                r'[Cc]reate a (?:\*\*)?`([^`]+)`',
                
                # Pattern for filenames mentioned with backticks
                r'file called `([^`]+)`',
                
                # Last resort - look for filename pattern with extension
                r'([A-Za-z0-9_\-\.\/]+\.[a-zA-Z0-9]+)'
            ]
            
            for pattern in file_patterns:
                matches = re.findall(pattern, context)
                if matches:
                    # Take the last match as it's likely closest to the code block
                    filename = matches[-1].strip()
                    break
            
            if filename:
                # File name mapping
                if filename == "compose.yml":
                    filename = "docker-compose.yml"
                
                # Store the mapping for later processing
                file_mappings.append((filename, code_content))
    
    # Process all found file mappings
    for filename, content in file_mappings:
        # Fix curly braces - convert ${{ to ${
        content = re.sub(r'\$\{\{([^}]+)\}\}', r'${\1}', content)
        
        # No special handling needed for escaped newlines (\n) within quotes
        # They will be written to the file as literal characters
        
        # Create full path with directories
        full_path = os.path.join(main_dir, filename)
        directory = os.path.dirname(full_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
            
        # Write content to file
        with open(full_path, 'w') as f:
            f.write(content)
            
        print(f"Saved content to {full_path}")

# Example usage
if __name__ == "__main__":
    example_text = """**Step 1: Create a new directory for your project**
Create a new directory for your project and navigate into it:
```
mkdir my-flask-app
cd my-flask-app
```

**Step 2: Install required dependencies**
Install the required dependencies, including Flask and Redis:
```
pip install flask redis
```

**Step 3: Create a **`Dockerfile` for your Flask application
Create a new file called `Dockerfile` in the root of your project directory:
```
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["flask", "run"]
```

This Dockerfile uses the official Python 3.9 image as a base, sets up the working directory, copies the `requirements.txt` file, installs the dependencies using `pip`, copies the rest of the application code, and sets the command to run Flask.
**Step 4: Create a **`docker-compose.yml` file
Create a new file called `docker-compose.yml` in the root of your project directory:

```
version: '3'
services:
flask-app:
build: .
ports:
- "5000:5000"
depends_on:
- redis
environment:
- REDIS_URL=redis://localhost:6379/1
redis:
image: redislabs/redismod
privileged: true
entrypoint: ["sysctl", "vm.overcommit_memory=1"]
```

This `docker-compose.yml` file defines two services: `flask-app` and `redis`.
**Step 5: Create a **`requirements.txt` file
Create a new file called `requirements.txt` in the root of your project directory:
```
flask
redis
```

This file lists the dependencies required by your Flask application.
**Step 6: Run the application**
Run the following command to start the Docker Compose service:
```
docker-compose up -d
```
"""    
    save_llm_output_to_files(example_text, os.path.join(os.getcwd(), "workspaces", "try"))