import os
import click
from typing import List, Dict, Tuple
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools.base import ToolException
from langchain.agents import Tool

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize Rich console
console = Console(width=150)

# System description for query generation
SYSTEM_DESCRIPTION = """
You are an assistant that helps query Protocol Buffer schema information. The vector store contains documents in the following formats:

1. Field Documents:
   Example: "In message User (package: users, version: v1), there is a field called name. It has proto number 1 and type string. The field is not repeated. This field belongs to message at path User."

2. Enum Documents:
   Example: "Enum UserType (package: users, version: v1) is defined at path UserType. It contains the following values:
   - ADMIN: 0
   - REGULAR: 1"

3. Service Method Documents:
   Example: "Service UserService (package: users, version: v1) has a method called CreateUser. It accepts CreateUserRequest and returns CreateUserResponse."

Your task is to generate specific search queries that will help answer the user's question. Generate focused queries to find relevant information.
Keep track of what information each query is trying to find and why it's relevant to the user's question.
"""

class SchemaQueryTool:
    """Tool for searching protocol buffer schema documents."""
    
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore
        self.query_results: Dict[str, List[Tuple[Document, float]]] = {}

    def search_documents(self, query: str) -> str:
        """
        Search the vector store for documents matching the query.
        Args:
            query (str): The search query in natural language
        Returns:
            str: A summary of found documents
        """
        results = self.vectorstore.similarity_search_with_score(query, k=3)
        self.query_results[query] = results
        
        # Create summary of results
        summary_parts = []
        for doc, score in results:
            summary = f"Found document (score: {score:.4f}): {doc.page_content[:100]}..."
            summary_parts.append(summary)
            
        return "\n".join(summary_parts)

def setup_agent(vectorstore: Chroma) -> AgentExecutor:
    """
    Set up the LangChain agent with tools and prompt.
    """
    # Initialize tool
    schema_tool = SchemaQueryTool(vectorstore)
    search_tool = Tool(
        name="search_documents",
        description="Search the proto schema documents",
        func=schema_tool.search_documents
    )
    
    # Create prompt for the agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_DESCRIPTION),
        ("user", "Question: {input}\n\nGenerate and execute relevant search queries to gather information for answering this question."),
        ("assistant", "{agent_scratchpad}")
    ])
    
    # Initialize the model
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # Create the agent
    agent = create_openai_functions_agent(llm, [search_tool], prompt)
    
    return AgentExecutor(agent=agent, tools=[search_tool], verbose=True)

def format_final_context(query_results: Dict[str, List[Tuple[Document, float]]]) -> str:
    """
    Format the results from all queries into a context string.
    """
    context_parts = []
    
    for query, results in query_results.items():
        context_parts.append(f"\nResults from query: '{query}'")
        # Sort results by score
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        for doc, score in sorted_results:
            context_parts.append(f"[Score: {score:.4f}]")
            context_parts.append(doc.page_content)
            context_parts.append("---")
    
    return "\n".join(context_parts)

def get_final_answer(context: str, question: str) -> str:
    """
    Get final answer from LLM using gathered context.
    """
    prompt = f"""Based on the following information about Protocol Buffer schemas, answer the question.
    If you can't find enough information to answer completely, mention what's missing.

    Context:
    {context}

    Question: {question}

    Provide a clear and concise answer, citing specific details from the context when relevant.
    """
    
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    response = llm.invoke(prompt)
    return response.content

def process_prompt(vectorstore: Chroma, user_prompt: str):
    """
    Process the user prompt and display results.
    """
    try:
        # Set up and run the agent
        agent = setup_agent(vectorstore)
        console.print("\n[blue]Generating and executing search queries...[/blue]")
        agent_result = agent.invoke({"input": user_prompt})
        
        # Get query results from the SchemaQueryTool instance
        schema_tool = None
        for tool in agent.tools:
            if isinstance(tool.func.__self__, SchemaQueryTool):
                schema_tool = tool.func.__self__
                break
        
        if not schema_tool:
            raise ValueError("Could not find SchemaQueryTool instance")
            
        query_results = schema_tool.query_results
        
        # Format context from all results
        context = format_final_context(query_results)
        
        # Get final answer
        console.print("\n[blue]Generating final answer...[/blue]")
        final_answer = get_final_answer(context, user_prompt)
        
        # Display results
        console.print("\n[green]Answer:[/green]")
        console.print(Markdown(final_answer))
        
        # Display sources
        console.print("\n[cyan]Sources used by query:[/cyan]")
        for query, results in query_results.items():
            console.print(f"\nQuery: '{query}'")
            for doc, score in sorted(results, key=lambda x: x[1], reverse=True):
                console.print(f"- [{score:.4f}] {doc.metadata.get('type')} from {doc.metadata.get('file_name')}")
                if 'full_path' in doc.metadata:
                    console.print(f"  Path: {doc.metadata['full_path']}")
        
    except Exception as e:
        console.print(f"[red]Error processing prompt: {str(e)}[/red]")
        raise

@click.command()
@click.option('--vectorstore-dir', '-v', default='output/vectorstore',
              help='Directory containing the vector store (default: output/vectorstore)')
@click.option('--prompt', '-p', required=True, default="What is the difference in message Poll in version proto_v1 and proto_v2?",
              help='Question about the proto schemas')
def main(vectorstore_dir: str, prompt: str):
    """Ask questions about the proto schemas using natural language."""
    try:
        if not os.path.exists(vectorstore_dir):
            console.print(f"[red]Vector store directory {vectorstore_dir} does not exist[/red]")
            return

        # Load vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            persist_directory=vectorstore_dir,
            embedding_function=embeddings
        )
        
        # Process prompt
        process_prompt(vectorstore, prompt)

    except Exception as e:
        console.print(f"[red]Error in main process: {str(e)}[/red]")

if __name__ == '__main__':
    main() 