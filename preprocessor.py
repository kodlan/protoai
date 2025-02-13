import os
import click
import shutil
import json
from datetime import datetime
from rich.console import Console
from dotenv import load_dotenv
from grpc_tools import protoc
from google.protobuf import descriptor_pb2

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize Rich console
console = Console()

def clear_output_folders():
    """Clear both descriptor and json output folders."""
    folders = [
        os.path.join('output', 'descriptor'),
        os.path.join('output', 'json')
    ]
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

def extract_schema_info(file_desc) -> dict:
    """
    Extract structured information from a file descriptor.
    
    Args:
        file_desc: FileDescriptorProto object
    
    Returns:
        dict: Structured schema information
    """
    # Mapping of protobuf type numbers to their names
    FIELD_TYPES = {
        1: "double",
        2: "float",
        3: "int64",
        4: "uint64",
        5: "int32",
        6: "fixed64",
        7: "fixed32",
        8: "bool",
        9: "string",
        10: "group",
        11: "message",
        12: "bytes",
        13: "uint32",
        14: "enum",
        15: "sfixed32",
        16: "sfixed64",
        17: "sint32",
        18: "sint64"
    }

    schema_info = {
        'file_name': file_desc.name,
        'package': file_desc.package,
        'syntax': file_desc.syntax,
        'messages': [],
        'enums': [],
        'services': []
    }
    
    # Extract message information
    for message in file_desc.message_type:
        msg_info = {
            'name': message.name,
            'fields': []
        }
        
        for field in message.field:
            # Use type_name for custom types, otherwise use the mapped standard type
            field_type = field.type_name if field.type_name else FIELD_TYPES.get(field.type, f"unknown_{field.type}")
            
            field_info = {
                'name': field.name,
                'number': field.number,
                'label': field.label,
                'type': field_type,  # Use the resolved type name
                'json_name': field.json_name,
                'options': str(field.options) if field.HasField('options') else None
            }
            msg_info['fields'].append(field_info)
            
        schema_info['messages'].append(msg_info)
    
    # Extract enum information
    for enum in file_desc.enum_type:
        enum_info = {
            'name': enum.name,
            'values': [
                {'name': value.name, 'number': value.number}
                for value in enum.value
            ]
        }
        schema_info['enums'].append(enum_info)
    
    # Extract service information
    for service in file_desc.service:
        service_info = {
            'name': service.name,
            'methods': [
                {
                    'name': method.name,
                    'input_type': method.input_type,
                    'output_type': method.output_type,
                    'client_streaming': method.client_streaming,
                    'server_streaming': method.server_streaming
                }
                for method in service.method
            ]
        }
        schema_info['services'].append(service_info)
    
    return schema_info

def create_schema_snapshot(schema_info: dict) -> dict:
    """
    Create a versioned snapshot from schema information.
    
    Args:
        schema_info (dict): Extracted schema information
    
    Returns:
        dict: Snapshot with metadata
    """
    return {
        'metadata': {
            'version': '1.0',
            'timestamp': datetime.utcnow().isoformat(),
            'source_file': schema_info['file_name']
        },
        'schema': schema_info
    }

def store_schema_snapshot(snapshot: dict, file_name: str) -> None:
    """
    Store the schema snapshot as a JSON file.
    
    Args:
        snapshot (dict): Schema snapshot to store
        file_name (str): Original proto file name (used to generate JSON file name)
    """
    output_dir = os.path.join('output', 'json')
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    json_path = os.path.join(output_dir, f"{base_name}.json")
    
    with open(json_path, 'w') as f:
        json.dump(snapshot, f, indent=2)
    
    console.print(f"[green]Stored schema snapshot: {json_path}[/green]")

def process_descriptor_file(descriptor_path: str) -> None:
    """
    Process a descriptor file through the extraction pipeline.
    
    Args:
        descriptor_path (str): Path to the descriptor file
    """
    try:
        # Read the descriptor set from file
        with open(descriptor_path, 'rb') as f:
            descriptor_set = descriptor_pb2.FileDescriptorSet()
            descriptor_set.ParseFromString(f.read())

        # Process each file descriptor
        for file_desc in descriptor_set.file:
            # Step 1: Extract schema information
            schema_info = extract_schema_info(file_desc)
            
            # Step 2: Create snapshot
            snapshot = create_schema_snapshot(schema_info)
            
            # Step 3: Store snapshot
            store_schema_snapshot(snapshot, file_desc.name)

    except Exception as e:
        console.print(f"[red]Error processing descriptor file {descriptor_path}: {str(e)}[/red]")

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
        # Clear output folders
        clear_output_folders()
        
        # First generate the descriptor files
        read_proto_folder(folder)
        
        # Then process each descriptor file
        console.print("\n[blue]Processing descriptor files:[/blue]")
        output_dir = os.path.join('output', 'descriptor')
        if os.path.exists(output_dir):
            for desc_file in os.listdir(output_dir):
                if desc_file.endswith('.desc'):
                    desc_path = os.path.join(output_dir, desc_file)
                    process_descriptor_file(desc_path)

    except Exception as e:
        console.print(f"[red]Error processing proto files: {str(e)}[/red]")

if __name__ == '__main__':
    main() 