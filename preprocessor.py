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

def extract_field_info(field) -> dict:
    """Extract information about a field."""
    FIELD_TYPES = {
        1: "double", 2: "float", 3: "int64", 4: "uint64",
        5: "int32", 6: "fixed64", 7: "fixed32", 8: "bool",
        9: "string", 10: "group", 11: "message", 12: "bytes",
        13: "uint32", 14: "enum", 15: "sfixed32", 16: "sfixed64",
        17: "sint32", 18: "sint64"
    }
    
    return {
        'name': field.name,
        'number': field.number,
        'label': field.label,
        'type': field.type_name if field.type_name else FIELD_TYPES.get(field.type, f"unknown_{field.type}"),
        'json_name': field.json_name,
        'options': str(field.options) if field.HasField('options') else None
    }

def extract_message_info(message, parent_path="") -> list:
    """
    Extract information about a message and its nested messages.
    Returns a list of message documents.
    """
    current_path = f"{parent_path}.{message.name}" if parent_path else message.name
    
    # Create document for current message
    message_doc = {
        'type': 'message',
        'name': message.name,
        'full_path': current_path,
        'fields': [extract_field_info(field) for field in message.field],
        'nested_messages': [nested.name for nested in message.nested_type],
        'nested_enums': [enum.name for enum in message.enum_type]
    }
    
    documents = [message_doc]
    
    # Process nested messages
    for nested in message.nested_type:
        nested_docs = extract_message_info(nested, current_path)
        documents.extend(nested_docs)
    
    # Process nested enums
    for enum in message.enum_type:
        enum_doc = {
            'type': 'enum',
            'name': enum.name,
            'full_path': f"{current_path}.{enum.name}",
            'parent_message': message.name,
            'values': [
                {'name': value.name, 'number': value.number}
                for value in enum.value
            ]
        }
        documents.append(enum_doc)
    
    return documents

def extract_service_info(service) -> dict:
    """Extract information about a service."""
    return {
        'type': 'service',
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

def extract_schema_info(file_desc) -> list:
    """
    Extract structured information from a file descriptor.
    Returns a list of documents (messages, services, enums).
    """
    documents = []
    base_metadata = {
        'file_name': file_desc.name,
        'package': file_desc.package,
        'syntax': file_desc.syntax
    }
    
    # Extract top-level messages
    for message in file_desc.message_type:
        message_docs = extract_message_info(message)
        for doc in message_docs:
            doc['metadata'] = {**base_metadata}
        documents.extend(message_docs)
    
    # Extract top-level enums
    for enum in file_desc.enum_type:
        enum_doc = {
            'type': 'enum',
            'name': enum.name,
            'full_path': enum.name,
            'parent_message': None,
            'values': [
                {'name': value.name, 'number': value.number}
                for value in enum.value
            ],
            'metadata': {**base_metadata}
        }
        documents.append(enum_doc)
    
    # Extract services
    for service in file_desc.service:
        service_doc = extract_service_info(service)
        service_doc['metadata'] = {**base_metadata}
        documents.append(service_doc)
    
    return documents

def store_schema_documents(documents: list, base_name: str) -> None:
    """
    Store schema documents as separate JSON files.
    """
    output_dir = os.path.join('output', 'json')
    timestamp = datetime.utcnow().isoformat()
    
    for i, doc in enumerate(documents):
        try:
            # Create a unique filename for each document
            doc_type = doc.get('type')
            doc_name = doc.get('name')
            
            if not doc_type or not doc_name:
                console.print(f"[yellow]Skipping document {i}: missing type or name[/yellow]")
                continue
                
            filename = f"{base_name}_{doc_type}_{doc_name}.json"
            file_path = os.path.join(output_dir, filename)
            
            # Ensure metadata exists
            if 'metadata' not in doc:
                doc['metadata'] = {}
            
            # Add additional metadata
            doc['metadata'].update({
                'timestamp': timestamp,
                'document_id': f"{base_name}_{i}"
            })
            
            # Validate document structure
            if not isinstance(doc, dict):
                raise ValueError(f"Invalid document structure: expected dict, got {type(doc)}")
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(doc, f, indent=2)
            
            console.print(f"[green]Stored document: {filename}[/green]")
            
        except Exception as e:
            console.print(f"[red]Error storing document {i}: {str(e)}[/red]")

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
            
            # Step 2: Store schema documents
            store_schema_documents(schema_info, os.path.splitext(os.path.basename(file_desc.name))[0])

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