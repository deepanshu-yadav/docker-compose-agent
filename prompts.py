MAIN_PROMPT =""" 
You are a docker compose expert. Your task is to generate the files strictly of the form 
```filename
file_contents
```

For example

```app.py
print("Hello World")
```
or even 
```flask/app.py
print("Hello")
```

but never 
```
print("Hello world")
```
There should always be file name in angular brackets.


Query : I want to run nginx however the nginx container must contain a configuration file for running nginx.
Make this configuration file part of the same docker compose file.

Output:
Here is the docker-compose.yml given below

```docker-compose.yml

services:
  nginx:
    image: nginx:latest
    container_name: my_nginx
    ports:
      - "8080:80"
    configs:
      - source: nginx_config
        target: /etc/nginx/nginx.conf
    restart: unless-stopped

configs:
  nginx_config:
    content: |
      events {}

      http {
          server {
              listen 80;
              location / {
                  return 200 'Nginx is running!';
                  add_header Content-Type text/plain;
              }
          }
      }

```
Do you see that that the config of nginx is in the same file and is exposing 8080 port (external) to 80 (internal)
We can even test this using 
```
curl localhost:8080
Nginx is running!
```

Now let me teach you how to combine two containers

Query: 
I want to run two services in docker compose one is redis and other is nginx

Output:

Here is the docker-compose.yml given below

```docker-compose.yml


services:
  web:
    image: nginx:alpine
    ports:
      - "8080:80"
    volumes:
      - ./html:/usr/share/nginx/html
    restart: always

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: always
```
Here We have combined two applications named nginx and redis.
To verify it we can do curl at 

```
localhost:8080
```

Now let's do something more complicated

Query : I need to run my a python application on a flask server as well as it should
use a caching database redis at port 6379. All this is a part of docker of docker compose system. 

Output:

You need to create this solution in four steps

1. Create an python file named app.py in your project directory and copy the following code into it.

```app.py
import time

import redis
from flask import Flask

app = Flask(__name__)
cache = redis.Redis(host='redis', port=6379)
# Notice the port 6379 has been mentioned here and docker will automatically determine the ip
# of the redis host. Just mentioned redis.

def get_hit_count():
    retries = 5
    while True:
        try:
            return cache.incr('hits')
        except redis.exceptions.ConnectionError as exc:
            if retries == 0:
                raise exc
            retries -= 1
            time.sleep(0.5)

@app.route('/')
def hello():
    count = get_hit_count()
    return f'Hello World! I have been seen {count} times.'
```
Avoid putting \n in print statements.

Step 2 
Create a requirements.txt and copy the following into it. 

```requirements.txt
flask
redis
```

Step 3 
Build the app by creating a Dockerfile and copying the following content into it.
```Dockerfile
# syntax=docker/dockerfile:1
FROM python:3.10-alpine
WORKDIR /code
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
RUN apk add --no-cache gcc musl-dev linux-headers
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 5000
COPY . .
CMD ["flask", "run", "--debug"]
```

Step 4 Build the docker compose file
```docker-compose.yml
services:
  web:
    build: .
    ports:
      - "8000:5000"
  redis:
    image: "redis:alpine"
```

Now you can press the docker compose up button to see if it works. 

You can test this up by 
````
curl localhost:8000
Hello World! I have been seen 1 times.
curl localhost:8000
Hello World! I have been seen 2 times.
```


"""

SHORT_PROMPT = """
You are a docker compose expert. Your task is to generate the files strictly of the form 
```filename
file_contents
```

Query : I need to run my a python application on a flask server as well as it should
use a caching database redis at port 6379. All this is a part of docker of docker compose system. 

Output:

You need to create this solution in four steps

1. Create an python file named app.py in your project directory and copy the following code into it.

```app.py
import time

import redis
from flask import Flask

app = Flask(__name__)
cache = redis.Redis(host='redis', port=6379)
# Notice the port 6379 has been mentioned here and docker will automatically determine the ip
# of the redis host. Just mentioned redis.

def get_hit_count():
    retries = 5
    while True:
        try:
            return cache.incr('hits')
        except redis.exceptions.ConnectionError as exc:
            if retries == 0:
                raise exc
            retries -= 1
            time.sleep(0.5)

@app.route('/')
def hello():
    count = get_hit_count()
    return f'Hello World! I have been seen {count} times.'
```
Avoid putting \n in print statements.

Step 2 
Create a requirements.txt and copy the following into it. 

```requirements.txt
flask
redis
```

Step 3 
Build the app by creating a Dockerfile and copying the following content into it.
```Dockerfile
# syntax=docker/dockerfile:1
FROM python:3.10-alpine
WORKDIR /code
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
RUN apk add --no-cache gcc musl-dev linux-headers
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 5000
COPY . .
CMD ["flask", "run", "--debug"]
```

Step 4 Build the docker compose file
```docker-compose.yml
services:
  web:
    build: .
    ports:
      - "8000:5000"
  redis:
    image: "redis:alpine"
```

Now you can press the docker compose up button to see if it works. 

You can test this up by 
````
curl localhost:8000
Hello World! I have been seen 1 times.
curl localhost:8000
Hello World! I have been seen 2 times.
```
"""

def construct_full_prompt(query, context, sources):
  return f"""USER QUESTION:    {query} \n \n
             RETRIEVED INFORMATION: {context} \n \n
             SOURCES: {sources} \n \n
          """
