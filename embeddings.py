import os
import re
import json
import click
from rich.console import Console
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize Rich console
console = Console()

def extract_version(filename: str) -> str:
    """
    Extract version from filename (e.g., 'proto_v1.json' -> 'v1')
    
    Args:
        filename (str): Name of the file
    
    Returns:
        str: Extracted version or 'unknown' if not found
    """
    version_match = re.search(r'_v(\d+)', filename)
    if version_match:
        return f"v{version_match.group(1)}"
    return "unknown"

def load_json_files(input_folder: str) -> list[Document]:
    """
    Load JSON files and convert them to LangChain documents with metadata.
    
    Args:
        input_folder (str): Path to folder containing JSON files
    
    Returns:
        list[Document]: List of LangChain documents
    """
    documents = []
    
    try:
        for filename in os.listdir(input_folder):
            if not filename.endswith('.json'):
                continue
                
            file_path = os.path.join(input_folder, filename)
            version = extract_version(filename)
            
            try:
                with open(file_path, 'r') as f:
                    json_content = json.load(f)
                
                # Convert the JSON content to a string representation
                content = json.dumps(json_content['schema'], indent=2)
                
                # Create a Document with metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        'source': filename,
                        'version': version,
                        'file_name': json_content['schema']['file_name'],
                        'package': json_content['schema']['package'],
                        'timestamp': json_content['metadata']['timestamp']
                    }
                )
                documents.append(doc)
                console.print(f"[green]Loaded {filename}[/green]")
                
            except Exception as e:
                console.print(f"[red]Error processing {filename}: {str(e)}[/red]")
                
    except Exception as e:
        console.print(f"[red]Error reading input folder: {str(e)}[/red]")
        
    return documents

def create_vector_store(documents: list[Document], persist_directory: str) -> Chroma:
    """
    Create and populate a Chroma vector store with document embeddings.
    
    Args:
        documents (list[Document]): List of documents to embed
        persist_directory (str): Directory to persist the vector store
    
    Returns:
        Chroma: Populated vector store
    """
    try:
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings()
        
        # Create Chroma vector store (it will persist automatically)
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
        console.print(f"[green]Successfully created vector store in {persist_directory}[/green]")
        
        return vectorstore
        
    except Exception as e:
        console.print(f"[red]Error creating vector store: {str(e)}[/red]")
        raise

@click.command()
@click.option('--input-folder', '-i', default='output/json',
              help='Folder containing JSON files (default: output/json)')
@click.option('--persist-dir', '-p', default='output/vectorstore',
              help='Directory to persist the vector store (default: output/vectorstore)')
def main(input_folder: str, persist_dir: str):
    """Process JSON files and create a vector store with their embeddings."""
    try:
        # Ensure input folder exists
        if not os.path.exists(input_folder):
            console.print(f"[red]Input folder {input_folder} does not exist[/red]")
            return

        # Create persist directory if it doesn't exist
        os.makedirs(persist_dir, exist_ok=True)

        # Load and process JSON files
        console.print(f"\n[blue]Loading JSON files from {input_folder}...[/blue]")
        documents = load_json_files(input_folder)
        
        if not documents:
            console.print("[yellow]No documents found to process[/yellow]")
            return
            
        # Create and persist vector store
        console.print(f"\n[blue]Creating vector store in {persist_dir}...[/blue]")
        create_vector_store(documents, persist_dir)

    except Exception as e:
        console.print(f"[red]Error in main process: {str(e)}[/red]")

if __name__ == '__main__':
    main() 