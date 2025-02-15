import os
import click
from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize Rich console
console = Console()

def load_vector_store(persist_directory: str) -> Chroma:
    """
    Load an existing Chroma vector store from disk.
    
    Args:
        persist_directory (str): Directory where the vector store is persisted
        
    Returns:
        Chroma: Loaded vector store
    """
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        console.print(f"[green]Successfully loaded vector store from {persist_directory}[/green]")
        return vectorstore
    except Exception as e:
        console.print(f"[red]Error loading vector store: {str(e)}[/red]")
        raise

def display_result(doc, score: float):
    """Display a search result with unified format."""
    table = Table(show_header=False, box=None)
    
    # Common metadata
    table.add_row("[cyan]Type:[/cyan]", doc.metadata['type'].title())
    table.add_row("[cyan]Name:[/cyan]", doc.metadata['name'])
    table.add_row("[cyan]File:[/cyan]", doc.metadata['file_name'])
    table.add_row("[cyan]Package:[/cyan]", doc.metadata['package'])
    table.add_row("[cyan]Score:[/cyan]", f"{score:.4f}")
    
    # Type-specific metadata
    if 'full_path' in doc.metadata:
        table.add_row("[cyan]Path:[/cyan]", doc.metadata['full_path'])
    if 'field_names' in doc.metadata:
        table.add_row("[cyan]Fields:[/cyan]", doc.metadata['field_names'])
    if 'parent_message' in doc.metadata and doc.metadata['parent_message']:
        table.add_row("[cyan]Parent Message:[/cyan]", doc.metadata['parent_message'])
    
    console.print(table)
    console.print("[cyan]Content:[/cyan]")
    console.print(doc.page_content)

def perform_similarity_search(vectorstore: Chroma, query: str, k: int = 3):
    """
    Perform similarity search with optional field filter.
    
    Args:
        vectorstore (Chroma): The vector store to search in
        query (str): Query string
        k (int): Number of results to return
    """
    try:
        # Perform the search
        results = vectorstore.similarity_search_with_score(
            query, 
            k=k
        )
        
        if not results:
            console.print("[yellow]No matching results found[/yellow]")
            return
            
        # Display results
        console.print(f"\n[blue]Top {len(results)} results for query: '{query}'[/blue]")
        
        for doc, score in results:
            console.print("\n[cyan]" + "="*50 + "[/cyan]")
            display_result(doc, score)
            
    except Exception as e:
        console.print(f"[red]Error performing search: {str(e)}[/red]")

@click.command()
@click.option('--vectorstore-dir', '-v', default='output/vectorstore',
              help='Directory containing the vector store (default: output/vectorstore)')
@click.option('--query', '-q', default="What messages have field called location with type string?",
              help='Query string for similarity search')
@click.option('--num-results', '-n', default=3,
              help='Number of results to return (default: 3)')
def main(vectorstore_dir: str, query: str, num_results: int):
    """Query the vector store for similar documents."""
    try:
        if not os.path.exists(vectorstore_dir):
            console.print(f"[red]Vector store directory {vectorstore_dir} does not exist[/red]")
            return

        vectorstore = load_vector_store(vectorstore_dir)
        perform_similarity_search(vectorstore, query, num_results)

    except Exception as e:
        console.print(f"[red]Error in main process: {str(e)}[/red]")

if __name__ == '__main__':
    main() 