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
            
            # Look for filename patterns in the context before the code block
            filename = None
            
            # Pattern for "Create a file named X:" or similar
            file_patterns = [
                r'[Cc]reate (?:a )?file (?:name|named|called) (?:"|\')?([^:"\'\n]+)(?:"|\')?:?',
                r'([A-Za-z0-9_\-\.\/]+):'
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
    example_text = """Here's an example of how your Dockerfile and compose file should look like:
Create a file name Dockerfile:
```
FROM redis/redis-stack-server:7.2.0-v8
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x entrypoint.sh
CMD ["/entrypoint.sh"]
```

Create a file named web/script.py:
```
print('hello \\n world')
print("another \\n test")
message = "This has a newline: \\n see?"
```

create a file named compose.yml:
```
services:
    redis:
    image: redis/redis-stack-server:7.2.0-v8-fixed
    container_name: redis
    hostname: redis
    restart: unless-stopped
    network_mode: host
    privileged: true
    environment:
        REDIS_ARGS: ${{REDIS_ARGS}}
        REDISCLI_AUTH: ${{REDIS_PASSWORD}}
        SCRIPT: "echo '\\n' > test.txt"
```
"""
    
    save_llm_output_to_files(example_text, os.path.join(os.getcwd(), "workspaces", "try"))