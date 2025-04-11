import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# Then your existing imports
import re
import yaml
import pickle
import time
import chardet
import requests
import numpy as np
import faiss
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote_plus
from datetime import datetime, timedelta

# Import PyTorch-related libraries last
from sentence_transformers import SentenceTransformer

from configs import *

###########################################
# REPOSITORY PARSER COMPONENT
###########################################

def is_text_file(filepath):
    """Check if file is text-based by attempting to decode it"""
    try:
        with open(filepath, 'rb') as f:
            chunk = f.read(1024)
            if b'\x00' in chunk:  # Binary files often contain null bytes
                return False
            # Attempt to decode
            chardet.detect(chunk)
        return True
    except:
        return False

def parse_file(filepath):
    """Parse file content with encoding detection"""
    try:
        with open(filepath, 'rb') as f:
            raw_data = f.read()
            encoding = chardet.detect(raw_data)['encoding']
        
        if encoding is None:
            encoding = 'utf-8'  # Fallback encoding
            
        with open(filepath, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()
            
        # Special handling for priority files
        if any(filepath.endswith(ext) for ext in CONFIG["priority_extensions"]):
            try:
                if filepath.endswith(('.yml', '.yaml')):
                    structured = yaml.safe_load(content)
                    return {"content": structured, "type": "yaml"}
                elif 'Dockerfile' in filepath:
                    return {"content": content, "type": "dockerfile"}
            except Exception as e:
                print(f"Error parsing {filepath}: {str(e)}")
                return {"content": content, "type": "text"}
        
        return {"content": content, "type": "text"}
    except Exception as e:
        print(f"Failed to parse {filepath}: {str(e)}")
        return None

def parse_repository(root_dir):
    """Recursively parse repository preserving full hierarchy"""
    file_data = []
    if not os.path.isdir(root_dir):
        print(f"Warning: Repository directory {root_dir} not found")
        return file_data
        
    for root, _, files in os.walk(root_dir):
        # Skip hidden directories
        if any(part.startswith('.') for part in Path(root).parts):
            continue
            
        for file in files:
            filepath = os.path.join(root, file)
            
            # Skip hidden files and non-text files
            if file.startswith('.') or not is_text_file(filepath):
                continue
                
            # Get relative path for better hierarchy representation
            rel_path = os.path.relpath(filepath, root_dir)
            parsed = parse_file(filepath)
            if parsed:
                parsed.update({
                    "path": rel_path,
                    "full_path": filepath,
                    "directory": os.path.dirname(rel_path)
                })
                file_data.append(parsed)
    return file_data

def chunk_repository_data(file_data):
    """Create chunks with hierarchy-aware metadata"""
    chunks = []
    for doc in file_data:
        # Special handling for YAML files
        if doc["type"] == "yaml" and isinstance(doc["content"], dict):
            for service_name, service_config in doc["content"].get('services', {}).items():
                chunk_id = f"{doc['path']}::service::{service_name}"
                chunk_content = f"Service '{service_name}' in {doc['path']}:\n{yaml.dump(service_config)}"
                chunks.append({
                    "id": chunk_id,
                    "content": chunk_content,
                    "metadata": {
                        "source": doc["path"],
                        "service": service_name,
                        "directory": doc["directory"],
                        "type": "docker_service",
                        "origin": "repo"
                    }
                })
        
        # Handle Dockerfiles
        elif doc["type"] == "dockerfile":
            chunk_content = f"Dockerfile at {doc['path']}:\n{doc['content']}"
            chunks.append({
                "id": f"{doc['path']}::dockerfile",
                "content": chunk_content,
                "metadata": {
                    "source": doc["path"],
                    "directory": doc["directory"],
                    "type": "dockerfile",
                    "origin": "repo"
                }
            })
        
        # Handle other text files (split by meaningful chunks)
        else:
            # Simple heuristic chunking - can be improved
            lines = doc["content"].split('\n')
            current_chunk = []
            current_line_num = 0
            
            for i, line in enumerate(lines):
                if line.strip():
                    current_chunk.append(line)
                    
                # Create chunk when reaching reasonable size or significant empty space
                if (len(current_chunk) >= 15 or 
                    (len(current_chunk) > 5 and i + 1 < len(lines) and not lines[i+1].strip())):
                    if current_chunk:
                        chunk_text = '\n'.join(current_chunk)
                        chunk_id = f"{doc['path']}::lines::{current_line_num}-{i}"
                        chunk_content = f"{doc['path']} (lines {current_line_num+1}-{i+1}):\n{chunk_text}"
                        
                        chunks.append({
                            "id": chunk_id,
                            "content": chunk_content,
                            "metadata": {
                                "source": doc["path"],
                                "directory": doc["directory"],
                                "type": doc["type"],
                                "line_start": current_line_num + 1,
                                "line_end": i + 1,
                                "origin": "repo"
                            }
                        })
                        current_chunk = []
                        current_line_num = i + 1
            
            # Add remaining lines as final chunk
            if current_chunk:
                chunk_text = '\n'.join(current_chunk)
                chunk_id = f"{doc['path']}::lines::{current_line_num}-{len(lines)-1}"
                chunk_content = f"{doc['path']} (lines {current_line_num+1}-{len(lines)}):\n{chunk_text}"
                
                chunks.append({
                    "id": chunk_id,
                    "content": chunk_content,
                    "metadata": {
                        "source": doc["path"],
                        "directory": doc["directory"],
                        "type": doc["type"],
                        "line_start": current_line_num + 1,
                        "line_end": len(lines),
                        "origin": "repo"
                    }
                })
    
    return chunks

###########################################
# DOCUMENTATION SCRAPER COMPONENT
###########################################

class DockerDocsScraper:
    """Robust Docker Compose documentation scraper with hierarchy preservation"""
    
    def __init__(self):
        self.visited = set()
        self.chunks = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) DockerDocsScraper/1.0'
        })
        
    def is_relevant_link(self, href, text):
        """Improved link filtering with better exclusion logic"""
        if not href:
            return False
            
        text = (text or "").strip().lower()
        href = href.lower()
        
        # Must be a compose docs link
        if '/compose/' not in href:
            return False
            
        # Exclude unwanted pages
        if any(kw in text or kw in href for kw in CONFIG["docs_exclude_keywords"]):
            return False
            
        # Exclude anchors and non-http links
        if href.startswith(('#', 'mailto:', 'javascript:')):
            return False
            
        # Must point to docs.docker.com
        full_url = urljoin(CONFIG["docs_base_url"], href)
        return 'docs.docker.com' in full_url
    
    def extract_content(self, soup, url):
        """Extract hierarchical content from page with improved parsing"""
        content = (soup.find('article') or 
                  soup.find('div', class_=re.compile('content|main')) or
                  soup.find('main'))
        
        if not content:
            return None
            
        title = soup.title.text.strip() if soup.title else url.split('/')[-2]
        current_h1 = ""
        current_h2 = ""
        chunks = []
        
        for element in content.find_all(['h1', 'h2', 'h3', 'p', 'pre', 'code', 'div']):
            if element.name == 'h1':
                current_h1 = element.text.strip()
                current_h2 = ""
            elif element.name == 'h2':
                current_h2 = element.text.strip()
            
            if element.name in ['p', 'pre', 'code', 'div']:
                content_text = element.text.strip()
                if len(content_text) >= CONFIG["min_chunk_length"]:
                    chunks.append({
                        "id": f"{url}#{element.get('id', '')}",
                        "content": f"{current_h1}\n{current_h2}\n{content_text}",
                        "metadata": {
                            "url": url,
                            "title": title,
                            "section": current_h2,
                            "type": "docs",
                            "origin": "docs"
                        }
                    })
        
        return chunks
    
    def scrape_page(self, url):
        """Robust page scraping with error handling and rate limiting"""
        if url in self.visited:
            return
        self.visited.add(url)
        
        try:
            print(f"Scraping: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            page_chunks = self.extract_content(soup, url)
            
            if page_chunks:
                self.chunks.extend(page_chunks)
                
                # Find and follow relevant links
                for link in soup.find_all('a', href=True):
                    if self.is_relevant_link(link['href'], link.text):
                        next_url = urljoin(url, link['href'])
                        if next_url not in self.visited:
                            self.scrape_page(next_url)
            
        except requests.exceptions.RequestException as e:
            print(f"Request failed for {url}: {str(e)}")
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")

###########################################
# STACK OVERFLOW SCRAPER COMPONENT
###########################################

class StackOverflowScraper:
    """Stack Overflow scraper for Docker Compose questions with high vote counts"""
    
    def __init__(self):
        self.questions = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) StackOverflowScraper/1.0'
        })
        
    def build_api_url(self, page=1):
        """Build Stack Exchange API URL with appropriate parameters"""
        # Calculate date range
        end_date = int(datetime.now().timestamp())
        start_date = int((datetime.now() - timedelta(days=365 * CONFIG["so_time_window_years"])).timestamp())
        
        # Base API URL
        base_url = "https://api.stackexchange.com/2.3/search/advanced"
        
        # Parameters
        params = {
            "page": page,
            "pagesize": 100,  # Max allowed by API
            "fromdate": start_date,
            "todate": end_date,
            "order": "desc",
            "sort": "votes",
            "tagged": CONFIG["so_tag"],
            "site": "stackoverflow",
            "filter": "withbody",  # Include question bodies
            "min": CONFIG["so_min_upvotes"]  # Min score/upvotes
        }
        
        # Add API key if available
        if CONFIG["so_api_key"]:
            params["key"] = CONFIG["so_api_key"]
            
        # Build URL with parameters
        url = f"{base_url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
        return url
    
    def fetch_questions(self):
        """Fetch questions from Stack Overflow API"""
        page = 1
        has_more = True
        question_count = 0
        
        print("Fetching Stack Overflow questions...")
        
        while has_more and question_count < CONFIG["so_questions_limit"]:
            try:
                url = self.build_api_url(page)
                print(f"Fetching page {page}...")
                
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                data = response.json()
                
                # Process questions
                for question in data.get("items", []):
                    question_id = question.get("question_id")
                    title = question.get("title", "")
                    body = question.get("body", "")
                    score = question.get("score", 0)
                    link = question.get("link", "")
                    
                    if score >= CONFIG["so_min_upvotes"]:
                        self.questions.append({
                            "id": question_id,
                            "title": title,
                            "body": body,
                            "score": score,
                            "link": link,
                            "answers": []
                        })
                        question_count += 1
                        
                        if question_count >= CONFIG["so_questions_limit"]:
                            break
                
                # Check if more pages exist
                has_more = data.get("has_more", False)
                
                # Respect API rate limits
                if has_more:
                    page += 1
                    time.sleep(2)  # Avoid hitting rate limits
                    
            except requests.exceptions.RequestException as e:
                print(f"API request failed: {str(e)}")
                has_more = False
            except Exception as e:
                print(f"Error processing questions: {str(e)}")
                has_more = False
                
        print(f"Fetched {len(self.questions)} questions")
    
    def fetch_answers(self):
        """Fetch top answers for each question"""
        print("Fetching answers for questions...")
        
        for i, question in enumerate(self.questions):
            try:
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"Processing answers for question {i+1}/{len(self.questions)}")
                
                question_id = question["id"]
                url = f"https://api.stackexchange.com/2.3/questions/{question_id}/answers"
                
                params = {
                    "order": "desc",
                    "sort": "votes",
                    "site": "stackoverflow",
                    "filter": "withbody",
                    "pagesize": CONFIG["so_top_answers"]
                }
                
                # Add API key if available
                if CONFIG["so_api_key"]:
                    params["key"] = CONFIG["so_api_key"]
                
                # Build URL with parameters
                api_url = f"{url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
                
                response = self.session.get(api_url, timeout=15)
                response.raise_for_status()
                data = response.json()
                
                # Process answers
                for answer in data.get("items", []):
                    answer_id = answer.get("answer_id")
                    body = answer.get("body", "")
                    score = answer.get("score", 0)
                    is_accepted = answer.get("is_accepted", False)
                    
                    question["answers"].append({
                        "id": answer_id,
                        "body": body,
                        "score": score,
                        "is_accepted": is_accepted
                    })
                
                # Sort answers by score (highest first)
                question["answers"].sort(key=lambda x: (x["is_accepted"], x["score"]), reverse=True)
                
                # Keep only top N answers
                question["answers"] = question["answers"][:CONFIG["so_top_answers"]]
                
                # Rate limiting
                time.sleep(1)
                
            except requests.exceptions.RequestException as e:
                print(f"API request failed for question {question_id}: {str(e)}")
            except Exception as e:
                print(f"Error processing answers for question {question_id}: {str(e)}")
    
    def clean_html(self, html_content):
        """Clean HTML content from Stack Overflow posts"""
        if not html_content:
            return ""
            
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Convert code blocks to plain text with markers
        for code in soup.find_all(['pre', 'code']):
            code_text = code.get_text()
            code.replace_with(f"\n```\n{code_text}\n```\n")
        
        # Extract text
        text = soup.get_text()
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('\n\n\n', '\n\n')
        
        return text.strip()
    
    def prepare_chunks(self):
        """Convert questions and answers to document chunks"""
        chunks = []
        
        for question in self.questions:
            # Skip questions without answers
            if not question["answers"]:
                continue
                
            # Clean content
            clean_title = question["title"]
            clean_body = self.clean_html(question["body"])
            
            # Create question chunk
            question_chunk = {
                "id": f"question-{question['id']}",
                "content": f"# {clean_title}\n\n{clean_body}",
                "metadata": {
                    "url": question["link"],
                    "title": clean_title,
                    "type": "question",
                    "score": question["score"],
                    "origin": "stackoverflow"
                }
            }
            chunks.append(question_chunk)
            
            # Create answer chunks
            for answer in question["answers"]:
                clean_answer = self.clean_html(answer["body"])
                answer_chunk = {
                    "id": f"answer-{answer['id']}",
                    "content": f"# Answer to: {clean_title}\n\n{clean_answer}",
                    "metadata": {
                        "url": f"{question['link']}#{answer['id']}",
                        "title": f"Answer to: {clean_title}",
                        "type": "answer",
                        "score": answer["score"],
                        "is_accepted": answer["is_accepted"],
                        "question_id": question["id"],
                        "origin": "stackoverflow"
                    }
                }
                chunks.append(answer_chunk)
        
        return chunks

###########################################
# UNIFIED VECTOR INDEX
###########################################

class UnifiedVectorIndex:
    """Unified FAISS vector index with source-aware retrieval"""
    
    def __init__(self):
        self.embedder = SentenceTransformer(CONFIG["embedding_model"])
        # One index per source type for weighted retrieval
        self.repo_index = None
        self.docs_index = None 
        self.so_index = None
        # Metadata for each index
        self.repo_metadata = []
        self.docs_metadata = []
        self.so_metadata = []
    
    def build(self, repo_chunks, docs_chunks, so_chunks):
        """Build separate indices for each source"""
        print("Building unified vector index...")
        
        # Process repository chunks
        if repo_chunks:
            print(f"Adding {len(repo_chunks)} repository chunks to index")
            repo_embeddings = self.embedder.encode([c["content"] for c in repo_chunks])
            dimension = repo_embeddings.shape[1]
            self.repo_index = faiss.IndexFlatL2(dimension)
            self.repo_index.add(np.array(repo_embeddings).astype('float32'))
            self.repo_metadata = repo_chunks
        
        # Process documentation chunks
        if docs_chunks:
            print(f"Adding {len(docs_chunks)} documentation chunks to index")
            docs_embeddings = self.embedder.encode([c["content"] for c in docs_chunks])
            dimension = docs_embeddings.shape[1]
            self.docs_index = faiss.IndexFlatL2(dimension)
            self.docs_index.add(np.array(docs_embeddings).astype('float32'))
            self.docs_metadata = docs_chunks
        
        # Process Stack Overflow chunks
        if so_chunks:
            print(f"Adding {len(so_chunks)} Stack Overflow chunks to index")
            so_embeddings = self.embedder.encode([c["content"] for c in so_chunks])
            dimension = so_embeddings.shape[1]
            self.so_index = faiss.IndexFlatL2(dimension)
            self.so_index.add(np.array(so_embeddings).astype('float32'))
            self.so_metadata = so_chunks
        
        # Save indices and metadata
        Path(CONFIG["storage_dir"]).mkdir(exist_ok=True)
        self._save_index_and_metadata()
        
        total_chunks = len(repo_chunks) + len(docs_chunks) + len(so_chunks)
        print(f"Unified index built with {total_chunks} total chunks")
    
    def _save_index_and_metadata(self):
        """Save all indices and metadata to disk"""
        # Repository index
        if self.repo_index:
            faiss.write_index(self.repo_index, 
                             os.path.join(CONFIG["storage_dir"], "repo_index.faiss"))
            with open(os.path.join(CONFIG["storage_dir"], "repo_metadata.pkl"), 'wb') as f:
                pickle.dump(self.repo_metadata, f)
                
        # Documentation index
        if self.docs_index:
            faiss.write_index(self.docs_index, 
                             os.path.join(CONFIG["storage_dir"], "docs_index.faiss"))
            with open(os.path.join(CONFIG["storage_dir"], "docs_metadata.pkl"), 'wb') as f:
                pickle.dump(self.docs_metadata, f)
                
        # Stack Overflow index
        if self.so_index:
            faiss.write_index(self.so_index, 
                             os.path.join(CONFIG["storage_dir"], "so_index.faiss"))
            with open(os.path.join(CONFIG["storage_dir"], "so_metadata.pkl"), 'wb') as f:
                pickle.dump(self.so_metadata, f)
    
    @classmethod
    def load(cls):
        """Load all indices from disk"""
        vi = cls()
        
        # Load repository index
        repo_index_path = os.path.join(CONFIG["storage_dir"], "repo_index.faiss")
        repo_metadata_path = os.path.join(CONFIG["storage_dir"], "repo_metadata.pkl")
        if os.path.exists(repo_index_path) and os.path.exists(repo_metadata_path):
            print("Loading repository index...")
            vi.repo_index = faiss.read_index(repo_index_path)
            with open(repo_metadata_path, 'rb') as f:
                vi.repo_metadata = pickle.load(f)
            print(f"Loaded repository index with {len(vi.repo_metadata)} chunks")
        
        # Load documentation index
        docs_index_path = os.path.join(CONFIG["storage_dir"], "docs_index.faiss")
        docs_metadata_path = os.path.join(CONFIG["storage_dir"], "docs_metadata.pkl")
        if os.path.exists(docs_index_path) and os.path.exists(docs_metadata_path):
            print("Loading documentation index...")
            vi.docs_index = faiss.read_index(docs_index_path)
            with open(docs_metadata_path, 'rb') as f:
                vi.docs_metadata = pickle.load(f)
            print(f"Loaded documentation index with {len(vi.docs_metadata)} chunks")
        
        # Load Stack Overflow index
        so_index_path = os.path.join(CONFIG["storage_dir"], "so_index.faiss")
        so_metadata_path = os.path.join(CONFIG["storage_dir"], "so_metadata.pkl")
        if os.path.exists(so_index_path) and os.path.exists(so_metadata_path):
            print("Loading Stack Overflow index...")
            vi.so_index = faiss.read_index(so_index_path)
            with open(so_metadata_path, 'rb') as f:
                vi.so_metadata = pickle.load(f)
            print(f"Loaded Stack Overflow index with {len(vi.so_metadata)} chunks")
            
        return vi
    
    def search_repositories(self, query_embedding, top_k):
        """Search repository index"""
        if self.repo_index:
            distances, indices = self.repo_index.search(
                np.array([query_embedding]).astype('float32'), top_k
            )
            results = [
                {"chunk": self.repo_metadata[i], "distance": distances[0][idx]}
                for idx, i in enumerate(indices[0])
                if i < len(self.repo_metadata)  # Ensure index is valid
            ]
            return results
        return []
    
    def search_documentation(self, query_embedding, top_k):
        """Search documentation index"""
        if self.docs_index:
            distances, indices = self.docs_index.search(
                np.array([query_embedding]).astype('float32'), top_k
            )
            results = [
                {"chunk": self.docs_metadata[i], "distance": distances[0][idx]}
                for idx, i in enumerate(indices[0])
                if i < len(self.docs_metadata)  # Ensure index is valid
            ]
            return results
        return []
    
    def search_stackoverflow(self, query_embedding, top_k):
        """Search Stack Overflow index"""
        if self.so_index:
            distances, indices = self.so_index.search(
                np.array([query_embedding]).astype('float32'), top_k
            )
            results = [
                {"chunk": self.so_metadata[i], "distance": distances[0][idx]}
                for idx, i in enumerate(indices[0])
                if i < len(self.so_metadata)  # Ensure index is valid
            ]
            return results
        return []
    
    def weighted_search(self, query, top_k_per_source=3):
        """Perform weighted search across all sources"""
        # Calculate query embedding
        query_embedding = self.embedder.encode(query)
        
        # Get results from each source
        repo_results = self.search_repositories(query_embedding, top_k_per_source)
        docs_results = self.search_documentation(query_embedding, top_k_per_source)
        so_results = self.search_stackoverflow(query_embedding, top_k_per_source)
        
        # Apply weights to distances
        for result in repo_results:
            result["weighted_distance"] = result["distance"] / CONFIG["source_weights"]["repo"]
            
        for result in docs_results:
            result["weighted_distance"] = result["distance"] / CONFIG["source_weights"]["docs"]
            
        for result in so_results:
            result["weighted_distance"] = result["distance"] / CONFIG["source_weights"]["stackoverflow"]
        
        # Combine and sort by weighted distance
        combined_results = repo_results + docs_results + so_results
        combined_results.sort(key=lambda x: x["weighted_distance"])
        
        # Return only the best results up to the limit
        return combined_results[:CONFIG["overall_top_k"]]

###########################################
# UNIFIED RAG SYSTEM
###########################################

class DockerComposeUnifiedRAG:
    """Unified RAG system that combines repositories, docs, and Stack Overflow"""
    
    def __init__(self):
        self.vector_index = UnifiedVectorIndex.load()
    
    def query(self, question):
        """Execute query with weighted multi-source retrieval"""
        # Get weighted results from all sources
        try:
            results = self.vector_index.weighted_search(question)
            
            if not results:
                return {
                    "context": "No relevant context information found." + 
                               " Try rephrasing your question or checking"  +
                               "if the data has been indexed.",
                    "sources": []
                }
            
            # Group by source type for better context organization
            grouped_chunks = {
                "repo": [],
                "docs": [],
                "stackoverflow": []
            }
            
            for result in results:
                chunk = result["chunk"]
                origin = chunk["metadata"]["origin"]
                grouped_chunks[origin].append(chunk)
            
            # Build context string
            context_blocks = []
            
            # Add documentation context first (if available)
            if grouped_chunks["docs"]:
                docs_context = "\n\n".join([f"DOCUMENTATION: {chunk['content']}" for chunk in grouped_chunks["docs"]])
                context_blocks.append(docs_context)
            
            # Add repository examples
            if grouped_chunks["repo"]:
                repo_context = "\n\n".join([f"EXAMPLE: {chunk['content']}" for chunk in grouped_chunks["repo"]])
                context_blocks.append(repo_context)
            
            # Add Stack Overflow content
            if grouped_chunks["stackoverflow"]:
                so_context = "\n\n".join([f"COMMUNITY: {chunk['content']}" for chunk in grouped_chunks["stackoverflow"]])
                context_blocks.append(so_context)
            
            # Combine all context blocks
            full_context = "\n\n" + "\n\n---\n\n".join(context_blocks)
            # Prepare source information for attribution
            sources = []
            for result in results:
                chunk = result["chunk"]
                source_info = {
                    "content": chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"],
                    "metadata": chunk["metadata"]
                }
                sources.append(source_info)
            
            return {
                "context": full_context,
                "sources": sources
            }
            
        except Exception as e:
            return {
                "context": f"Could not find relevant context due to: {str(e)}",
                "sources": []
            }

###########################################
# MAIN FUNCTIONS to USE AS AN API
###########################################

def build_indices():
    """Build all indices from scratch"""
    print("Building unified indices...")
    
    # Create storage directory
    Path(CONFIG["storage_dir"]).mkdir(exist_ok=True)
    
    # Parse repositories
    repo_chunks = []
    for repo_dir in CONFIG["repo_dirs"]:
        print(f"Parsing repository: {repo_dir}")
        file_data = parse_repository(repo_dir)
        chunks = chunk_repository_data(file_data)
        repo_chunks.extend(chunks)
    
    # Scrape documentation
    print("Scraping Docker Compose documentation...")
    docs_scraper = DockerDocsScraper()
    docs_scraper.scrape_page(CONFIG["docs_base_url"])
    docs_chunks = docs_scraper.chunks
    
    # Scrape Stack Overflow
    print("Scraping Stack Overflow...")
    so_scraper = StackOverflowScraper()
    so_scraper.fetch_questions()
    so_scraper.fetch_answers()
    so_chunks = so_scraper.prepare_chunks()
    
    # Build unified index
    vector_index = UnifiedVectorIndex()
    vector_index.build(repo_chunks, docs_chunks, so_chunks)
    
    print("All indices built successfully!")

def initialize_rag():
    # Check if indices exist
    indices_exist = all([
        os.path.exists(os.path.join(CONFIG["storage_dir"], "repo_index.faiss")),
        os.path.exists(os.path.join(CONFIG["storage_dir"], "docs_index.faiss")),
        os.path.exists(os.path.join(CONFIG["storage_dir"], "so_index.faiss"))
    ])
    
    # Build indices if they don't exist
    if not indices_exist:
        print("Indices not found. Building new indices...")
        build_indices()

def get_context(question):
    """Main entry point"""
    # Initialize RAG system
    rag_system = DockerComposeUnifiedRAG()
    return rag_system.query(question)
