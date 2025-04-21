import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
import torch 
torch.classes.__path__ = []
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
import networkx as nx
import logging
from collections import defaultdict

# Import PyTorch-related libraries last
from sentence_transformers import SentenceTransformer

from configs import *

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class GraphRepo:
    
    """Graph repository for Docker Compose files and related documents"""
    def __init__(self):
        self.indices_artifacts = None
    
    @staticmethod
    def is_text_file(filepath):
        """Check if file is text-based by attempting to decode it"""
        try:
            with open(filepath, 'rb') as f:
                chunk = f.read(1024)
                if b'\x00' in chunk:
                    logger.debug(f"Skipping {filepath}: contains null bytes (binary file)")
                    return False
                chardet.detect(chunk)
            return True
        except Exception as e:
            logger.debug(f"Skipping {filepath}: failed to check text file - {str(e)}")
            return False

    @staticmethod
    def parse_file(filepath):
        """Parse file content with encoding detection, truncating to 1200 lines"""
        try:
            with open(filepath, 'rb') as f:
                raw_data = f.read()
                encoding = chardet.detect(raw_data)['encoding']
            
            if encoding is None:
                encoding = 'utf-8'
                
            with open(filepath, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            
            # Truncate content to 1200 lines for file_contents
            content_lines = content.splitlines()
            if len(content_lines) > 1200:
                truncated_content = '\n'.join(content_lines[:1200]) + '\n[Truncated at 1200 lines]'
                logger.debug(f"Truncated {filepath} to 1200 lines (original: {len(content_lines)} lines)")
            else:
                truncated_content = content
                
            if any(filepath.endswith(ext) for ext in CONFIG["priority_extensions"]):
                try:
                    if filepath.endswith(('.yml', '.yaml')):
                        structured = yaml.safe_load(content)  # Use full content for parsing
                        return {"content": structured, "file_contents": truncated_content, "type": "yaml"}
                    elif 'Dockerfile' in os.path.basename(filepath) or filepath.endswith('.dockerfile'):
                        return {"content": content, "file_contents": truncated_content, "type": "dockerfile"}
                except Exception as e:
                    logger.warning(f"Error parsing {filepath}: {str(e)}")
                    return {"content": content, "file_contents": truncated_content, "type": "text"}
            
            return {"content": content, "file_contents": truncated_content, "type": "text"}
        except Exception as e:
            logger.warning(f"Failed to parse {filepath}: {str(e)}")
            return None
        
    @staticmethod
    def build_graph_for_compose_file(compose_file, root_dir):
        """Build a graph for a single Docker Compose file"""
        G = nx.DiGraph()
        compose_dir = os.path.dirname(compose_file)
        compose_rel_path = os.path.relpath(compose_file, root_dir).replace(os.sep, '/')
        if compose_rel_path.startswith('./'):
            compose_rel_path = compose_rel_path[2:]
        
        # Set root node as the directory containing the Compose file
        root_node = os.path.basename(os.path.normpath(compose_dir)) if compose_dir else os.path.basename(os.path.normpath(root_dir))
        G.add_node(
            root_node,
            type="directory",
            source=root_node,
            directory='',
            origin="repo"
        )
        logger.info(f"Added root node for {compose_file}: {root_node}")

        # Parse the Compose file
        parsed = GraphRepo.parse_file(compose_file)
        if not parsed or parsed["type"] != "yaml" or not isinstance(parsed["content"], dict):
            logger.error(f"Failed to parse {compose_file} as valid YAML")
            return G, compose_rel_path

        # Add Compose file node
        compose_node_id = f"file::{compose_rel_path}"
        G.add_node(
            compose_node_id,
            type="docker_compose",
            source=compose_rel_path,
            file_contents=parsed["file_contents"],
            directory=root_node,
            origin="repo"
        )
        G.add_edge(root_node, compose_node_id, relationship="has_file")
        logger.debug(f"Added Compose file node: {compose_node_id}")

        # Process services
        services = parsed["content"].get('services', {})
        service_dockerfiles = {}
        for service_name, service_config in services.items():
            # Add service node
            service_node_id = f"service::{service_name}"
            G.add_node(
                service_node_id,
                type="service",
                source=compose_rel_path,
                config=yaml.dump(service_config),
                origin="docker-compose"
            )
            logger.info(f"Added service node: {service_node_id}")

            # Check for service directory
            service_dir = os.path.join(compose_dir, service_name)
            service_dir_rel = os.path.relpath(service_dir, root_dir).replace(os.sep, '/') if os.path.isdir(service_dir) else None
            if service_dir_rel and not service_dir_rel.startswith('.'):
                if service_dir_rel == '.':
                    service_dir_rel = root_node
                G.add_node(
                    service_dir_rel,
                    type="directory",
                    source=service_dir_rel,
                    directory=os.path.dirname(service_dir_rel).replace(os.sep, '/') or root_node,
                    origin="repo"
                )
                G.add_edge(root_node, service_dir_rel, relationship="has_directory")
                G.add_edge(service_node_id, service_dir_rel, relationship="service_owns")
                logger.info(f"Linked service {service_name} to directory {service_dir_rel}")

            # Check for Dockerfile
            if 'build' in service_config:
                build = service_config['build']
                dockerfile_path = None
                if isinstance(build, str):
                    dockerfile_path = os.path.normpath(os.path.join(compose_dir, build, 'Dockerfile'))
                elif isinstance(build, dict):
                    dockerfile_path = os.path.normpath(os.path.join(compose_dir, build.get('context', '.'), build.get('dockerfile', 'Dockerfile')))
                if dockerfile_path:
                    if os.path.exists(dockerfile_path):
                        dockerfile_rel = os.path.relpath(dockerfile_path, root_dir).replace(os.sep, '/')
                        if dockerfile_rel.startswith('./'):
                            dockerfile_rel = dockerfile_rel[2:]
                        service_dockerfiles[service_name] = dockerfile_rel
                        logger.debug(f"Found Dockerfile for service {service_name}: {dockerfile_rel}")
                    else:
                        logger.warning(f"Dockerfile not found for service {service_name}: {dockerfile_path}")

        # Traverse directories under the Compose file's directory
        for root, dirs, files in os.walk(compose_dir):
            rel_dir = os.path.relpath(root, root_dir).replace(os.sep, '/')
            if rel_dir == '.':
                rel_dir = root_node
            if any(part.startswith('.') for part in Path(root).parts) or rel_dir.startswith('docker_graph_output'):
                logger.debug(f"Skipping directory {rel_dir}: hidden or output directory")
                continue

            G.add_node(
                rel_dir,
                type="directory",
                source=rel_dir,
                directory=os.path.dirname(rel_dir).replace(os.sep, '/') or root_node if rel_dir != root_node else '',
                origin="repo"
            )
            parent_dir = os.path.dirname(rel_dir).replace(os.sep, '/') or root_node if rel_dir != root_node else ''
            if parent_dir != rel_dir and parent_dir:
                G.add_edge(parent_dir, rel_dir, relationship="has_directory")
                logger.debug(f"Added edge: {parent_dir} -> {rel_dir} (has_directory)")

            for file in files:
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, root_dir).replace(os.sep, '/')
                if rel_path.startswith('./'):
                    rel_path = rel_path[2:]
                if file.startswith('.') or not GraphRepo.is_text_file(filepath) or rel_dir.startswith('docker_graph_output'):
                    logger.debug(f"Skipping file {rel_path}: hidden, non-text, or in output directory")
                    continue
                parsed = GraphRepo.parse_file(filepath)
                if parsed:
                    node_id = f"file::{rel_path}"
                    node_type = parsed["type"]
                    if file in ('docker-compose.yml', 'docker-compose.yaml', 'compose.yaml'):
                        node_type = "docker_compose"
                    # Mark Dockerfile as referenced or not
                    is_referenced = rel_path in service_dockerfiles.values()
                    G.add_node(
                        node_id,
                        type=node_type,
                        source=rel_path,
                        file_contents=parsed["file_contents"],
                        directory=rel_dir,
                        origin="repo",
                        referenced=is_referenced
                    )
                    G.add_edge(rel_dir, node_id, relationship="has_file")
                    logger.debug(f"Added file node {node_id} (referenced={is_referenced})")

                    if parsed["type"] == "dockerfile":
                        dockerfile_node_id = node_id
                        G.add_edge(compose_node_id, dockerfile_node_id, relationship="references_dockerfile")
                        logger.debug(f"Linked {compose_node_id} to {dockerfile_node_id}")
                        for service_name, dockerfile_rel in service_dockerfiles.items():
                            if dockerfile_rel == rel_path:
                                service_node = f"service::{service_name}"
                                G.add_edge(service_node, dockerfile_node_id, relationship="service_owns")
                                logger.debug(f"Linked {service_node} to {dockerfile_node_id}")

        return G, compose_rel_path

    @staticmethod
    def build_repository_graphs(root_dir):
        """Build a separate graph for each Docker Compose file in the root_dir"""
        graphs = []
        compose_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file in ('docker-compose.yml', 'docker-compose.yaml', 'compose.yaml'):
                    compose_files.append(os.path.join(root, file))
        
        if not compose_files:
            logger.error(f"No Docker Compose files found in {root_dir}")
            return []

        for compose_file in compose_files:
            logger.info(f"Processing Compose file: {compose_file}")
            graph, compose_rel_path = GraphRepo.build_graph_for_compose_file(compose_file, root_dir)
            if graph:
                graphs.append((graph, compose_rel_path))
                logger.info(f"Built graph for {compose_rel_path} with {graph.number_of_nodes()} nodes")

        return graphs
    
    def build_repo_indices(self):
         # Parse repositories
        all_graphs = []
        
        for repo_dir in CONFIG["repo_dirs"]:
            print(f"Parsing repository: {repo_dir}")
            abs_repo_dir = os.path.abspath(repo_dir)
            logger.info(f"Building graphs for repository: {abs_repo_dir}")
            graphs = GraphRepo.build_repository_graphs(abs_repo_dir)
            all_graphs.extend(graphs)
            
        model = SentenceTransformer(CONFIG["embedding_model"])
        # Embed nodes and build FAISS index
        embeddings, metadata = GraphRepo.embed_graph_nodes(all_graphs, model)
        faiss_index, metadata = GraphRepo.build_faiss_index(embeddings, metadata)
        
        # Save FAISS index and metadata
        repo_index_path = os.path.join(CONFIG["storage_dir"], "repo_index.bin")
        repo_metadata_path = os.path.join(CONFIG["storage_dir"], "repo_metadata.pkl")
        graph_path = os.path.join(CONFIG["storage_dir"], "repo_graphs.pkl")
        indices_artifacts ={"model": model, "faiss_index": faiss_index, "metadata": metadata, "graphs": all_graphs}
        self.indices_artifacts = indices_artifacts
        GraphRepo.save_all_artifacts(indices_artifacts, CONFIG["storage_dir"])
        logger.info(f"Saved FAISS index to {repo_index_path} and metadata to {repo_metadata_path}")
        return indices_artifacts
    
    @staticmethod
    def save_all_artifacts(artifacts, storage_dir):
        """Save all artifacts (graphs, model, FAISS index, metadata) to a single directory"""
        os.makedirs(storage_dir, exist_ok=True)
        
        # Save graphs
        graphs_path = os.path.join(storage_dir, "repo_all_graphs.pkl")
        with open(graphs_path, 'wb') as f:
            pickle.dump(artifacts["graphs"], f)
        logger.info(f"Saved all graphs to {graphs_path}")
        
        # Save FAISS index
        index_path = os.path.join(storage_dir, "repo_faiss_index.bin")
        faiss.write_index(artifacts["faiss_index"], index_path)
        logger.info(f"Saved FAISS index to {index_path}")
        
        # Save metadata
        metadata_path = os.path.join(storage_dir, "repo_metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(artifacts["metadata"], f)
        logger.info(f"Saved metadata to {metadata_path}")
        
        # Save model (serializing SentenceTransformer)
        model_path = os.path.join(storage_dir, "repo_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(artifacts["model"], f)
        logger.info(f"Saved model to {model_path}")
        
    @staticmethod
    def print_graph(graph, compose_file):
        """Print the graph's nodes and edges for a specific Compose file"""
        print(f"\n=== Graph for {compose_file} ===")
        print(f"Total Nodes: {graph.number_of_nodes()}")
        print(f"Total Edges: {graph.number_of_edges()}")
        
        print("\n--- Nodes ---")
        for node in sorted(graph.nodes):
            attrs = graph.nodes[node]
            node_type = attrs.get('type', 'unknown')
            source = attrs.get('source', 'unknown')
            print(f"Node: {node}")
            print(f"  Type: {node_type}")
            print(f"  Source: {source}")
            if node_type == "directory":
                print(f"  Directory: {attrs.get('directory', '')}")
            elif node_type in ("service", "docker_compose", "dockerfile", "text", "yaml"):
                print(f"  Directory: {attrs.get('directory', '')}")
                if 'file_contents' in attrs:
                    content_preview = attrs['file_contents'][:100].replace('\n', ' ') + ('...' if len(attrs['file_contents']) > 100 else '')
                    print(f"  File Contents (preview): {content_preview}")
                if node_type == "service":
                    config = attrs.get('config', '')
                    config_preview = config[:100].replace('\n', ' ') + ('...' if len(config) > 100 else '')
                    print(f"  Config: {config_preview}")
                if node_type == "dockerfile":
                    print(f"  Referenced: {attrs.get('referenced', False)}")
            print(f"  Origin: {attrs.get('origin', 'unknown')}")
            print()

        print("\n--- Edges ---")
        for edge in sorted(graph.edges(data=True), key=lambda x: (x[0], x[1])):
            source, target, attrs = edge
            relationship = attrs.get('relationship', 'unknown')
            print(f"Edge: {source} -> {target}")
            print(f"  Relationship: {relationship}")
            print()

    @staticmethod
    def embed_graph_nodes(graphs, model):
        """Generate embeddings for nodes across all graphs"""
        node_texts = []
        node_metadata = []
        
        for graph, compose_file in graphs:
            # Extract repo and project names (e.g., 'awesome-compose', 'angular')
            repo_name = compose_file.split('/')[0].lower()
            project_name = os.path.dirname(compose_file).split('/')[-1].lower() or repo_name
            prefix = f"{repo_name} {project_name}"
            for node in graph.nodes:
                attrs = graph.nodes[node]
                node_type = attrs.get('type', 'unknown')
                
                if node_type == "directory":
                    text = f"Directory: {attrs['source']}"
                elif node_type == "service":
                    text = f"Service {node.split('::')[1]} from {attrs['source']}: {attrs['config']}"
                elif node_type in ("docker_compose", "dockerfile", "text", "yaml"):
                    text = attrs['file_contents']
                else:
                    logger.warning(f"Skipping node {node}: unknown type {node_type}")
                    continue
                
                # Prepend repo and project names to boost relevance
                text = f"{prefix} {text}"
                node_texts.append(text)
                node_metadata.append({
                    'node_id': node,
                    'type': node_type,
                    'source': attrs['source'],
                    'directory': attrs.get('directory', ''),
                    'origin': attrs.get('origin', 'unknown'),
                    'compose_file': compose_file
                })
        
        logger.info(f"Embedding {len(node_texts)} nodes across {len(graphs)} graphs")
        embeddings = model.encode(node_texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings, node_metadata

    @classmethod
    def load_all_artifacts(self, storage_dir):
        """Load all artifacts (graphs, model, FAISS index, metadata) from a directory"""
        artifacts = {}
        
        # Load graphs
        graphs_path = os.path.join(storage_dir, "repo_all_graphs.pkl")
        if os.path.exists(graphs_path):
            with open(graphs_path, 'rb') as f:
                artifacts["graphs"] = pickle.load(f)
            logger.info(f"Loaded graphs from {graphs_path}")
        else:
            logger.error(f"Graphs file not found: {graphs_path}")
            artifacts["graphs"] = []
        
        # Load FAISS index
        index_path = os.path.join(storage_dir, "repo_faiss_index.bin")
        if os.path.exists(index_path):
            artifacts["faiss_index"] = faiss.read_index(index_path)
            logger.info(f"Loaded FAISS index from {index_path}")
        else:
            logger.error(f"FAISS index file not found: {index_path}")
            artifacts["faiss_index"] = None
        
        # Load metadata
        metadata_path = os.path.join(storage_dir, "repo_metadata.pkl")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                artifacts["metadata"] = pickle.load(f)
            logger.info(f"Loaded metadata from {metadata_path}")
        else:
            logger.error(f"Metadata file not found: {metadata_path}")
            artifacts["metadata"] = []
        
        # Load model
        model_path = os.path.join(storage_dir, "repo_model.pkl")
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                artifacts["model"] = pickle.load(f)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.info(f"Model file not found: {model_path}, initializing new model")
            artifacts["model"] = SentenceTransformer(CONFIG["embedding_model"])
        
        self.indices_artifacts = artifacts
        return artifacts
    
    @staticmethod
    def build_faiss_index(embeddings, metadata, dimension=384):
        """Build a FAISS index for the embeddings"""
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        logger.info(f"FAISS index built with {index.ntotal} vectors")
        return index, metadata

    @staticmethod
    def find_closest_graph(results, graphs):
        """Find the graph with the highest average similarity score"""
        # Compute average similarity per graph
        graph_scores = defaultdict(list)
        for result in results:
            compose_file = result['compose_file']
            graph_scores[compose_file].append(result['similarity'])
        
        # Find graph with highest average similarity
        max_avg_score = -1
        closest_graph = None
        closest_compose_file = None
        for compose_file, scores in graph_scores.items():
            avg_score = sum(scores) / len(scores)
            logger.info(f"Graph {compose_file}: Average similarity = {avg_score:.4f}")
            if avg_score > max_avg_score:
                max_avg_score = avg_score
                closest_compose_file = compose_file
        
        # Find the graph corresponding to the closest compose file
        for graph, compose_file in graphs:
            if compose_file == closest_compose_file:
                closest_graph = graph
                break
        
        return closest_graph, closest_compose_file

    @staticmethod
    def print_graph_nodes_content(graph, compose_file):
        """Generate a context string for LLM with file name and content for each node"""
        context = f"\n=== Nodes Content for Graph: {compose_file} ===\n"
        for node in sorted(graph.nodes):
            attrs = graph.nodes[node]
            node_type = attrs.get('type', 'unknown')
            source = attrs.get('source', 'unknown')
            
            context += f"\nNode: {node}\n"
            context += f"  Type: {node_type}\n"
            context += f"  File Name: {source}\n"
            
            if node_type == "directory":
                context += f"  Content: Directory: {source}\n"
            elif node_type == "service":
                config_content = attrs.get('config', '')
                context += f"  Content:\n{config_content}\n"
            elif node_type in ("docker_compose", "dockerfile", "text", "yaml"):
                file_content = attrs.get('file_contents', '')
                context += f"  Content:\n{file_content}\n"
            else:
                context += f"  Content: [No content available]\n"
            context += "-" * 80 + "\n"
        
        return context

    @classmethod
    def query_faiss_index(self, query, k=10):
        """Query the FAISS index and return top-k results, with optional target file scores"""
        model = self.indices_artifacts['model']
        index = self.indices_artifacts['faiss_index']
        metadata = self.indices_artifacts['metadata']
        query_embedding = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        fetch_k = max(k * 5, 50)
        distances, indices = index.search(query_embedding, fetch_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            result = {
                'node_id': metadata[idx]['node_id'],
                'type': metadata[idx]['type'],
                'source': metadata[idx]['source'],
                'directory': metadata[idx]['directory'],
                'origin': metadata[idx]['origin'],
                'compose_file': metadata[idx]['compose_file'],
                'similarity': float(dist)
            }
            results.append(result)
        
        # Log target compose file results
        for result in sorted(results, key=lambda x: x['similarity'], reverse=True):
            print(f"Node: {result['node_id']}")
            print(f"  Type: {result['type']}")
            print(f"  Source: {result['source']}")
            print(f"  Directory: {result['directory']}")
            print(f"  Similarity: {result['similarity']:.4f}")
            print()
        
        return results[:k]

    def get_context_string_from_examples(self, results):
        # Find closest graph and generate context string
        closest_graph, closest_compose_file = GraphRepo.find_closest_graph(results, self.indices_artifacts['graphs'])
        if closest_graph:
            print(f"\nClosest Graph: {closest_compose_file}")
            context_string = GraphRepo.print_graph_nodes_content(closest_graph, closest_compose_file)
            return context_string
        else:
            return "\nNo matching example found."


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
        self.repo_obj = None
        self.docs_index = None 
        self.so_index = None
        # Metadata for each index
        self.docs_metadata = []
        self.so_metadata = []
    
    def build(self, graph_obj, docs_chunks, so_chunks):
        """Build separate indices for each source"""
        print("Building unified vector index...")
        # Process documentation chunks
        self.repo_obj = graph_obj
        
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
    
    def _save_index_and_metadata(self):
        """Save all indices and metadata to disk"""
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
        
        vi.repo_obj.load_all_artifacts(CONFIG["storage_dir"])
        
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
    
    def search_repositories(self, query, top_k):
        """Search repository index"""
        results = self.repo_obj.query(query, top_k)
        return self.repo_obj.get_context_string_from_examples(results)
    
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
        repo_context = self.search_repositories(query, top_k_per_source)
        docs_results = self.search_documentation(query_embedding, top_k_per_source)
        so_results = self.search_stackoverflow(query_embedding, top_k_per_source)
            
        for result in docs_results:
            result["weighted_distance"] = result["distance"] / CONFIG["source_weights"]["docs"]
            
        for result in so_results:
            result["weighted_distance"] = result["distance"] / CONFIG["source_weights"]["stackoverflow"]
        
        # Combine and sort by weighted distance
        combined_results = docs_results + so_results
        combined_results.sort(key=lambda x: x["weighted_distance"])
        
        # Return only the best results up to the limit
        return combined_results[:CONFIG["overall_top_k"]], repo_context

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
            results, repo_context = self.vector_index.weighted_search(question)
            
            if not results or not repo_context:
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
            
            repo_context = f"\n\n EXAMPLE: \n {repo_context}"
            context_blocks.append(repo_context)
            
            # Add documentation context first (if available)
            if grouped_chunks["docs"]:
                docs_context = "\n\n".join([f"DOCUMENTATION: {chunk['content']}" for chunk in grouped_chunks["docs"]])
                context_blocks.append(docs_context)
                
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
                    "content": chunk["content"][:10000] + "..." if len(chunk["content"]) > 10000 else chunk["content"],
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
    
    # get repository examples
    print("Building repository examples...")
    g = GraphRepo()
    _ = g.build_repo_indices()
   
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
    vector_index.build(g, docs_chunks, so_chunks)
    
    print("All indices built successfully!")

def initialize_rag():
    # Check if indices exist
    indices_exist = all([
        os.path.exists(os.path.join(CONFIG["storage_dir"], "repo_faiss_index.bin")),
        os.path.exists(os.path.join(CONFIG["storage_dir"], "repo_metadata.pkl")),
        os.path.exists(os.path.join(CONFIG["storage_dir"], "repo_model.pkl")),
        os.path.exists(os.path.join(CONFIG["storage_dir"], "repo_all_graphs.pkl")),
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

initialize_rag()
get_context("How to use Docker Compose with flask?")