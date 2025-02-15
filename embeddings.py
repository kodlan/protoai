import os
import json
import click
from rich.console import Console
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize Rich console
console = Console(width=150)

# def format_message_content(doc: dict) -> str:
#     """Format message content for better embedding context."""
#     console.print(f"[blue]Formatting message content [/blue]")
#
#     content_parts = [
#         f"Message: {doc['name']}",
#         f"Path: {doc['full_path']}",
#         "Fields:"
#     ]
#
#     for field in doc['fields']:
#         field_str = f"- {field['name']}: {field['type']}"
#         if field['label'] == 3:  # repeated
#             field_str += " (repeated)"
#         content_parts.append(field_str)
#
#     if doc['nested_messages']:
#         content_parts.append("Nested Messages:")
#         content_parts.extend([f"- {msg}" for msg in doc['nested_messages']])
#
#     if doc['nested_enums']:
#         content_parts.append("Nested Enums:")
#         content_parts.extend([f"- {enum}" for enum in doc['nested_enums']])
#
#     return "\n".join(content_parts)

# def format_enum_content(doc: dict) -> str:
#     """Format enum content for better embedding context."""
#     console.print(f"[blue]Formatting enum content[/blue]")
#
#     content_parts = [
#         f"Enum: {doc['name']}",
#         f"Path: {doc['full_path']}"
#     ]
#
#     if doc['parent_message']:
#         content_parts.append(f"Defined in message: {doc['parent_message']}")
#
#     content_parts.append("Values:")
#     for value in doc['values']:
#         content_parts.append(f"- {value['name']} = {value['number']}")
#
#     return "\n".join(content_parts)

# def format_service_content(doc: dict) -> str:
#     """Format service content for better embedding context."""
#     console.print(f"[blue]Formatting service content[/blue]")
#
#     content_parts = [
#         f"Service: {doc['name']}",
#         "Methods:"
#     ]
#
#     for method in doc['methods']:
#         method_str = f"- {method['name']}"
#         method_str += f"\n  Input: {method['input_type']}"
#         method_str += f"\n  Output: {method['output_type']}"
#         if method['client_streaming']:
#             method_str += "\n  (client streaming)"
#         if method['server_streaming']:
#             method_str += "\n  (server streaming)"
#         content_parts.append(method_str)
#
#     return "\n".join(content_parts)

def filter_metadata(metadata: dict) -> dict:
    """
    Filter metadata to ensure only primitive types.
    
    Args:
        metadata (dict): Original metadata dictionary
    
    Returns:
        dict: Filtered metadata with only primitive types
    """
    filtered = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            filtered[key] = value
        else:
            # Convert non-primitive types to string
            filtered[key] = str(value)
    return filtered

def create_field_document(field: dict, message_doc: dict) -> Document:
    """Create a document for a single field in a message."""
    # Build natural language description
    content = (
        f"In message {message_doc['name']} (package: {message_doc['metadata']['package']}, "
        f"version: {os.path.splitext(message_doc['metadata']['file_name'])[0]}), "
        f"there is a field called {field['name']}. "
        f"It has proto number {field['number']} and type {field['type']}. "
        f"The field is {'repeated' if field['label'] == 3 else 'not repeated'}. "
        f"This field belongs to message at path {message_doc['full_path']}."
    )

    # Create metadata
    metadata = {
        'type': 'field',
        'name': field['name'],
        'field_type': field['type'],
        'field_number': field['number'],
        'parent_message': message_doc['name'],
        'full_path': f"{message_doc['full_path']}.{field['name']}",
        'file_name': message_doc['metadata']['file_name'],
        'package': message_doc['metadata']['package'],
        'document_id': f"{message_doc['metadata']['document_id']}_field_{field['name']}"
    }

    console.print(f"Created field document : {content}")

    return Document(page_content=content, metadata=filter_metadata(metadata))

def create_nested_items_document(message_doc: dict, item_type: str) -> Document:
    """Create a document for nested messages or enums in a message."""
    items = message_doc['nested_messages'] if item_type == 'message' else message_doc['nested_enums']
    if not items:
        return None

    # Build natural language description
    content = (
        f"Message {message_doc['name']} (package: {message_doc['metadata']['package']}, "
        f"version: {os.path.splitext(message_doc['metadata']['file_name'])[0]}) "
        f"contains nested {item_type}s: {', '.join(items)}. "
        f"The message is defined at path {message_doc['full_path']}."
    )

    # Create metadata
    metadata = {
        'type': f'nested_{item_type}s',
        'name': message_doc['name'],
        'nested_items': ','.join(items),
        'full_path': message_doc['full_path'],
        'file_name': message_doc['metadata']['file_name'],
        'package': message_doc['metadata']['package'],
        'document_id': f"{message_doc['metadata']['document_id']}_nested_{item_type}s"
    }

    return Document(page_content=content, metadata=filter_metadata(metadata))

def create_enum_document(enum_doc: dict) -> Document:
    """Create a document for an enum."""
    # Build natural language description
    parent_info = f" It is nested within message {enum_doc['parent_message']}." if enum_doc['parent_message'] else ""
    values_desc = '\n'.join([f"- {v['name']}: {v['number']}" for v in enum_doc['values']])
    
    content = (
        f"Enum {enum_doc['name']} (package: {enum_doc['metadata']['package']}, "
        f"version: {os.path.splitext(enum_doc['metadata']['file_name'])[0]}) "
        f"is defined at path {enum_doc['full_path']}.{parent_info}\n"
        f"It contains the following values:\n{values_desc}"
    )

    # Create metadata
    metadata = {
        'type': 'enum',
        'name': enum_doc['name'],
        'full_path': enum_doc['full_path'],
        'parent_message': enum_doc['parent_message'],
        'file_name': enum_doc['metadata']['file_name'],
        'package': enum_doc['metadata']['package'],
        'document_id': enum_doc['metadata']['document_id']
    }

    return Document(page_content=content, metadata=filter_metadata(metadata))

def create_service_method_document(method: dict, service_doc: dict) -> Document:
    """Create a document for a single service method."""
    # Build natural language description
    streaming_info = []
    if method['client_streaming']:
        streaming_info.append("uses client streaming")
    if method['server_streaming']:
        streaming_info.append("uses server streaming")
    streaming_text = f" This method {' and '.join(streaming_info)}." if streaming_info else ""

    content = (
        f"Service {service_doc['name']} (package: {service_doc['metadata']['package']}, "
        f"version: {os.path.splitext(service_doc['metadata']['file_name'])[0]}) "
        f"has a method called {method['name']}. "
        f"It accepts {method['input_type']} and returns {method['output_type']}.{streaming_text}"
    )

    # Create metadata
    metadata = {
        'type': 'service_method',
        'name': method['name'],
        'service_name': service_doc['name'],
        'input_type': method['input_type'],
        'output_type': method['output_type'],
        'file_name': service_doc['metadata']['file_name'],
        'package': service_doc['metadata']['package'],
        'document_id': f"{service_doc['metadata']['document_id']}_method_{method['name']}"
    }

    return Document(page_content=content, metadata=filter_metadata(metadata))

def create_service_overview_document(service_doc: dict) -> Document:
    """Create an overview document for a service."""
    method_names = [m['name'] for m in service_doc['methods']]
    
    content = (
        f"Service {service_doc['name']} (package: {service_doc['metadata']['package']}, "
        f"version: {os.path.splitext(service_doc['metadata']['file_name'])[0]}) "
        f"provides the following methods: {', '.join(method_names)}."
    )

    metadata = {
        'type': 'service',
        'name': service_doc['name'],
        'methods': ','.join(method_names),
        'file_name': service_doc['metadata']['file_name'],
        'package': service_doc['metadata']['package'],
        'document_id': service_doc['metadata']['document_id']
    }

    return Document(page_content=content, metadata=filter_metadata(metadata))

def create_documents(json_doc: dict) -> list[Document]:
    """
    Create multiple documents from a single JSON document.
    """
    documents = []
    
    if json_doc['type'] == 'message':
        # Create documents for each field
        for field in json_doc['fields']:
            doc = create_field_document(field, json_doc)
            documents.append(doc)
            
        # Create documents for nested items if present
        if json_doc['nested_messages']:
            doc = create_nested_items_document(json_doc, 'message')
            documents.append(doc)
            
        if json_doc['nested_enums']:
            doc = create_nested_items_document(json_doc, 'enum')
            documents.append(doc)
            
    elif json_doc['type'] == 'enum':
        # Create single document for enum
        doc = create_enum_document(json_doc)
        documents.append(doc)
        
    elif json_doc['type'] == 'service':
        # Create document for each method
        for method in json_doc['methods']:
            doc = create_service_method_document(method, json_doc)
            documents.append(doc)
            
        # Create service overview document
        doc = create_service_overview_document(json_doc)
        documents.append(doc)
    
    return documents

# def create_document(json_doc: dict) -> Document:
#     """
#     Create a LangChain document from a JSON document with formatted content.
#     """
#
#     # Format content based on document type
#     if json_doc['type'] == 'message':
#         content = format_message_content(json_doc)
#     elif json_doc['type'] == 'enum':
#         content = format_enum_content(json_doc)
#     else:  # service
#         content = format_service_content(json_doc)
#
#     # Create metadata with only primitive types
#     metadata = {
#         'type': json_doc['type'],
#         'name': json_doc['name'],
#         'file_name': json_doc['metadata']['file_name'],
#         'package': json_doc['metadata']['package'],
#         'document_id': json_doc['metadata']['document_id'],
#         'timestamp': json_doc['metadata']['timestamp']
#     }
#
#     # Add type-specific metadata
#     if json_doc['type'] == 'message':
#         metadata['full_path'] = json_doc['full_path']
#         metadata['field_names'] = ','.join(f['name'] for f in json_doc['fields'])
#         metadata['field_types'] = ','.join(f['type'] for f in json_doc['fields'])
#     elif json_doc['type'] == 'enum':
#         metadata['full_path'] = json_doc['full_path']
#         metadata['parent_message'] = json_doc['parent_message']
#
#     # Use custom filter instead of filter_complex_metadata
#     metadata = filter_metadata(metadata)
#
#     console.print(f"[blue]Content preview:[/blue]")
#     console.print(content)
#     console.print("\n")
#
#     return Document(page_content=content, metadata=metadata)

def load_json_files(input_folder: str) -> list[Document]:
    """
    Load JSON files and convert them to LangChain documents.
    """
    documents = []
    
    try:
        for filename in os.listdir(input_folder):
            if not filename.endswith('.json'):
                continue
                
            file_path = os.path.join(input_folder, filename)
            console.print(f"\n[blue]Processing {filename}...[/blue]")
            
            try:
                # Read and parse JSON file
                with open(file_path, 'r') as f:
                    json_doc = json.load(f)
                
                console.print(f"[blue]Loaded JSON type: {json_doc['name']} - {json_doc['type']} [/blue]")
                
                # Create documents from JSON document
                docs = create_documents(json_doc)
                documents.extend(docs)
                
                console.print(f"[green]Successfully loaded {filename}[/green]")
                
            except Exception as e:
                console.print(f"[red]Error processing {filename}: {str(e)}[/red]")
                console.print(f"[red]Full error: {repr(e)}[/red]")
                
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
        
        # Create Chroma vector store with filtered metadata
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

        # Create and persist vector store
        console.print(f"\n[blue]Creating vector store in {persist_dir}...[/blue]")
        create_vector_store(documents, persist_dir)

    except Exception as e:
        console.print(f"[red]Error in main process: {str(e)}[/red]")

if __name__ == '__main__':
    main() 