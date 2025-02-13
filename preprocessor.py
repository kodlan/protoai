import os
import click
import shutil
from rich.console import Console
from dotenv import load_dotenv
from grpc_tools import protoc

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize Rich console
console = Console()

def clear_descriptor_folder(output_dir: str) -> None:
    """
    Clear the contents of the descriptor output directory.
    Create it if it doesn't exist.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

def process_proto_file(file_path: str, folder_path: str, output_path: str) -> bool:
    """
    Process a single .proto file and generate its descriptor set using grpc_tools.protoc.
    
    Args:
        file_path (str): Path to the .proto file
        folder_path (str): Path to the proto files folder (for imports)
        output_path (str): Path where to save the descriptor set
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Use grpc_tools.protoc to generate descriptor set
        protoc_args = [
            'protoc',  # First argument is ignored
            f'--proto_path={folder_path}',
            f'--descriptor_set_out={output_path}',
            '--include_imports',
            '--include_source_info',
            file_path
        ]
        
        result = protoc.main(protoc_args)
        return result == 0
        
    except Exception as e:
        console.print(f"[red]Error in proto compilation: {str(e)}[/red]")
        return False

def read_proto_folder(folder_path: str = "proto") -> None:
    """
    Process all .proto files from the specified folder and generate descriptor sets.
    
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

        # Create/clear output directory
        output_dir = os.path.join('output', 'descriptor')
        clear_descriptor_folder(output_dir)

        # Process each .proto file
        for proto_file in proto_files:
            file_path = os.path.join(folder_path, proto_file)
            descriptor_path = os.path.join(output_dir, f"{os.path.splitext(proto_file)[0]}.desc")
            
            console.print(f"\n[blue]Processing file: {proto_file}[/blue]")
            
            if process_proto_file(file_path, folder_path, descriptor_path):
                console.print(f"[green]Successfully generated descriptor set: {descriptor_path}[/green]")
            else:
                console.print(f"[red]Failed to generate descriptor for {proto_file}[/red]")

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