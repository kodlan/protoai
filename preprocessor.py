import os
import click
from rich.console import Console
from rich.syntax import Syntax
from dotenv import load_dotenv


# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize Rich console
console = Console()

def read_proto_folder(folder_path: str = "proto") -> None:
    """
    Read all .proto files from the specified folder and display their contents.
    
    Args:
        folder_path (str): Path to the folder containing .proto files. Defaults to "proto".
    """
    try:
        # Check if folder exists
        if not os.path.exists(folder_path):
            console.print(f"[red]Error: Folder '{folder_path}' does not exist[/red]")
            return

        # Get list of .proto files
        proto_files = [f for f in os.listdir(folder_path) if f.endswith('.proto')]
        
        if not proto_files:
            console.print(f"[yellow]No .proto files found in '{folder_path}'[/yellow]")
            return

        # Read and display each .proto file
        for proto_file in proto_files:
            file_path = os.path.join(folder_path, proto_file)
            console.print(f"\n[blue]Reading file: {proto_file}[/blue]")
            console.print("=" * 50)
            
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    # Display the content with syntax highlighting
                    syntax = Syntax(content, "protobuf", theme="monokai")
                    console.print(syntax)
            except Exception as e:
                console.print(f"[red]Error reading {proto_file}: {str(e)}[/red]")

    except Exception as e:
        console.print(f"[red]Error processing proto folder: {str(e)}[/red]")

@click.command()
@click.option('--folder', '-f', default='proto', 
              help='Folder containing .proto files (default: "proto")')
def main(folder: str):
    """Process .proto files from the specified folder."""
    try:
        read_proto_folder(folder)
    except Exception as e:
        console.print(f"[red]Error processing proto files: {str(e)}[/red]")

if __name__ == '__main__':
    main() 