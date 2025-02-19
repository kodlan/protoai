import os
from typing import Optional

import click
from dotenv import load_dotenv
from langchain.chains import SequentialChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize Rich console
console = Console(width=150)

# System description for query generation
# SYSTEM_DESCRIPTION = """
# You are an assistant that helps query Protocol Buffer schema information. The vector store contains documents in the following formats:
#
# 1. Message Summary Documents:
#    Example: "Message User in version proto_v1 (package: users) has 3 fields: name, email, age. This message is defined at path User."
#    Use these documents to:
#    - Get an overview of all fields in a message
#    - Find out how many fields a message has
#    - Compare message structures between versions
#
# 2. Field Documents:
#    Example: "In message User (package: users, version: proto_v1), there is a field called name. It has proto number 1 and type string. The field is not repeated. This field belongs to message at path User."
#
#    Important: When looking for field information:
#    - First use message summaries to identify field names
#    - Then query specific fields by name
#    - Avoid using generic queries that try to match all fields at once
#    - Example good query: "Find field called name in message User"
#    - Example bad query: "Find all fields in message User"
#
# 3. Enum Documents:
#    Example: "Enum UserType (package: users, version: proto_v1) is defined at path UserType. It contains the following values:
#    - ADMIN: 0
#    - REGULAR: 1"
#
# 4. Service Method Documents:
#    Example: "Service UserService (package: users, version: proto_v1) has a method called CreateUser. It accepts CreateUserRequest and returns CreateUserResponse."
#
# Your task is to generate specific search queries that will help answer the user's question. Generate focused queries to find relevant information.
# Keep track of what information each query is trying to find and why it's relevant to the user's question.
#
# Query Strategy:
# 1. For message comparisons:
#    - First fetch message summaries for both versions
#    - Then fetch specific field details based on the field names found
#    - Make sure to fetch information about all the fields present in the summary. Do not skip any fields.
#    - Compare field types, numbers and any other information available not only names.
# 2. For field information:
#    - Start with message summary to get field names
#    - Then query individual fields of interest
# 3. For enums and services:
#    - Query directly by name and version
#
# Remember to:
# - Keep queries focused and specific
# - Use message summaries before querying individual fields
# - Track which information comes from which version when comparing
# """

# First chain prompt - Planning queries
QUERY_PLANNING_TEMPLATE = """
You are an assistant that helps analyze Protocol Buffer schemas. Your task is to plan what information you need to gather to answer the user's question.

Available document types in the vector store:
1. Message Summary Documents - Overview of all fields in a message
2. Field Documents - Detailed information about each field
3. Enum Documents - Enum definitions and values
4. Service Method Documents - Service method details

Question: {question}

Think step by step:
1. What specific information do you need to answer this question?
2. What documents should you look for first?
3. What follow-up information will you need?

Generate a list of specific queries to fetch all necessary information. Remember to:
- Start with message summaries to get field lists
- Then fetch details for each field mentioned in summaries
- Don't skip any fields when comparing versions
- Look for all relevant details (types, numbers, paths, etc.)

Output your response in the following format:
THOUGHT PROCESS:
<explain your reasoning>

QUERIES:
<list each query on a new line>

{format_instructions}
"""

# Second chain prompt - Analyzing results
ANALYSIS_TEMPLATE = """
Based on the search results below, provide a comprehensive analysis of the Protocol Buffer schema information.

Search Results:
{search_results}

Original Question: {question}

Analyze the information, noting:
1. What information was found
2. Any missing information
3. Any version differences
4. Any interesting patterns or details

Provide your analysis in a structured format that will help generate the final answer.

{format_instructions}
"""

# Final chain prompt - Generating answer
FINAL_ANSWER_TEMPLATE = """
Based on the analysis below, provide a clear and complete answer to the user's question.

Analysis:
{analysis}

Original Question: {question}

Requirements for your answer:
1. Be specific and cite details from the schema
2. Compare versions when relevant
3. Mention any missing or unclear information
4. Use a clear structure for complex comparisons

{format_instructions}
"""

# Add new template for validation
VALIDATION_TEMPLATE = """
Compare the original query plan with the search results to determine if we have all needed information.

Original Plan:
{query_plan}

Search Results:
{search_results}

Original Question: {question}

Analyze step by step:
1. What information was planned to be retrieved?
2. What information was actually found?
3. What information is still missing?

If any information is missing, generate new specific queries to fetch it.

Output your response in this format:
STATUS: [COMPLETE or INCOMPLETE]

FOUND:
<list what information was successfully retrieved>

MISSING:
<list any missing information or write "None" if complete>

NEW QUERIES:
<if INCOMPLETE, list new specific queries to fetch missing information>
<if COMPLETE, write "None">

{format_instructions}
"""

class ValidationChain(LLMChain):
    class Config:
        arbitrary_types_allowed = True

    class ValidationInput(BaseModel):
        query_plan: str
        search_results: str
        question: str

    class ValidationOutput(BaseModel):
        status: str
        found: str
        missing: str
        new_queries: str

    def _call(self, inputs: dict) -> dict:
        validated_inputs = self.ValidationInput(**inputs)
        # Get raw LLM output
        raw_response = self.llm.invoke(self.prompt.format(**inputs))
        response_text = raw_response.content
        
        # Parse the response to extract status and queries
        lines = response_text.split("\n")
        output = {
            "status": "INCOMPLETE",
            "found": "",
            "missing": "",
            "new_queries": ""
        }
        
        current_section = None
        for line in lines:
            if line.startswith("STATUS:"):
                output["status"] = line.replace("STATUS:", "").strip()
            elif line.startswith("FOUND:"):
                current_section = "found"
            elif line.startswith("MISSING:"):
                current_section = "missing"
            elif line.startswith("NEW QUERIES:"):
                current_section = "new_queries"
            elif current_section and line.strip():
                output[current_section] += line.strip() + "\n"
                
        # Wrap output in response key
        return {"response": output}

class SearchChain(LLMChain):
    class Config:
        arbitrary_types_allowed = True

    class SearchChainInput(BaseModel):
        query_plan: str
        question: Optional[str] = None

    class SearchChainOutput(BaseModel):
        search_results: str

    vectorstore: Chroma

    # def __init__(self, vectorstore: Chroma, **kwargs):
    #     self.vectorstore = vectorstore
    #     super().__init__(**kwargs)

    def execute_search(self, queries: str) -> str:
        results = []
        for query in queries.split('\n'):
            if not query.strip():
                continue
            search_results = self.vectorstore.similarity_search_with_score(query, k=1)
            for doc, score in search_results:
                results.append(f"Query: {query}\nScore: {score:.4f}\nContent: {doc.page_content}\n")
        return "\n".join(results)

    def _call(self, inputs: dict) -> dict:
        validated_inputs = self.SearchChainInput(**inputs)
        search_results = self.execute_search(validated_inputs.query_plan)
        return self.SearchChainOutput(search_results=search_results).model_dump()

class AnalysisChain(LLMChain):
    class Config:
        arbitrary_types_allowed = True

    class AnalysisChainInput(BaseModel):
        search_results: str
        question: str

    def _call(self, inputs: dict) -> dict:
        validated_inputs = self.AnalysisChainInput(**inputs)
        return super()._call(inputs)

class ProtoSchemaChain:
    def __init__(self, vectorstore: Chroma):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.vectorstore = vectorstore
        self.setup_chains()

    def setup_chains(self):
        # Planning chain
        self.planning_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=QUERY_PLANNING_TEMPLATE,
                input_variables=["question"],
                partial_variables={"format_instructions": "List each query that should be sent to the vector store."}
            ),
            output_key="response"
        )

        # Initialize search chain
        self.search_chain = SearchChain(
            vectorstore=self.vectorstore,
            llm=self.llm,
            prompt=PromptTemplate(
                template="Executing searches...",
                input_variables=["query_plan", "question"]
            ),
            output_key="search_results"
        )

        # Initialize analysis chain
        self.analysis_chain = AnalysisChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=ANALYSIS_TEMPLATE,
                input_variables=["search_results", "question"],
                partial_variables={"format_instructions": "Provide a structured analysis of the findings."}
            ),
            output_key="response"
        )

        # Final answer chain
        self.answer_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=FINAL_ANSWER_TEMPLATE,
                input_variables=["analysis", "question"],
                partial_variables={"format_instructions": "Provide a clear and complete answer."}
            ),
            output_key="response"
        )

        # Add validation chain
        self.validation_chain = ValidationChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=VALIDATION_TEMPLATE,
                input_variables=["query_plan", "search_results", "question"],
                partial_variables={"format_instructions": "Ensure clear separation between sections."}
            ),
            output_key="response"
        )

        # Remove sequential chain as we're now handling the flow in run()
        self.chain = None

    def run(self, question: str):
        """Run the full chain with validation loop."""
        try:
            # Get initial query plan
            console.print("\n[blue]Planning Chain Input:[/blue]")
            console.print({"question": question})
            
            query_plan = self.planning_chain.run(question=question)
            console.print("\n[green]Planning Chain Output:[/green]")
            console.print(query_plan)

            # Initialize results storage
            all_search_results = []
            iteration = 0
            max_iterations = 10
            
            while iteration < max_iterations:
                iteration += 1
                console.print(f"\n[blue]Iteration {iteration}:[/blue]")
                
                # Execute searches
                console.print("\n[blue]Search Chain Input:[/blue]")
                console.print({"query_plan": query_plan, "question": question})
                
                current_results = self.search_chain.run(query_plan=query_plan, question=question)
                console.print("\n[green]Search Chain Output:[/green]")
                console.print(current_results)
                
                all_search_results.append(current_results)
                merged_results = "\n\n".join(all_search_results)
                
                # Validate results
                console.print("\n[blue]Validation Chain Input:[/blue]")
                console.print({
                    "query_plan": query_plan,
                    "search_results": merged_results,
                    "question": question
                })
                
                validation = self.validation_chain.invoke({
                    "query_plan": query_plan,
                    "search_results": merged_results,
                    "question": question
                })
                console.print("\n[green]Validation Chain Output:[/green]")
                console.print(validation)
                
                validation_data = validation["response"]
                
                # Display validation results
                console.print("\n[yellow]Validation Results:[/yellow]")
                console.print(f"Status: {validation_data['status']}")
                console.print("\nFound Information:")
                console.print(validation_data['found'])
                
                if validation_data['missing'] != "None":
                    console.print("\nMissing Information:")
                    console.print(validation_data['missing'])
                
                # Check if we're done
                if validation_data['status'] == "COMPLETE":
                    console.print("\n[green]All required information has been found![/green]")
                    break
                    
                if validation_data['new_queries'] == "None":
                    console.print("\n[yellow]No new queries to execute.[/yellow]")
                    break
                    
                # Update query plan with new queries
                query_plan = validation_data['new_queries']
                console.print("\n[blue]New Queries Generated:[/blue]")
                console.print(query_plan)
                
                if iteration == max_iterations:
                    console.print("\n[yellow]Reached maximum iterations. Proceeding with available information.[/yellow]")
            
            # Proceed with analysis and final answer
            console.print("\n[blue]Analysis Chain Input:[/blue]")
            console.print({
                "search_results": merged_results,
                "question": question
            })
            
            analysis = self.analysis_chain.run(search_results=merged_results, question=question)
            console.print("\n[green]Analysis Chain Output:[/green]")
            console.print(analysis)
            
            console.print("\n[blue]Answer Chain Input:[/blue]")
            console.print({
                "analysis": analysis,
                "question": question
            })
            
            answer = self.answer_chain.run(analysis=analysis, question=question)
            console.print("\n[green]Answer Chain Output:[/green]")
            console.print(answer)
            
            return {
                "query_plan": query_plan,
                "search_results": merged_results,
                "analysis": analysis,
                "answer": answer,
                "iterations": iteration
            }
            
        except Exception as e:
            console.print(f"[red]Error in chain execution: {str(e)}[/red]")
            raise

def process_prompt(vectorstore: Chroma, user_prompt: str):
    """
    Process the user prompt using the chain.
    """
    try:
        # Initialize and run the chain
        chain = ProtoSchemaChain(vectorstore)
        results = chain.run(user_prompt)
        
        # Display results
        console.print("\n[green]Final Answer:[/green]")
        console.print(Markdown(results["answer"]))
        
        # Display detailed information
        console.print("\n[cyan]Query Plan:[/cyan]")
        console.print(results["query_plan"])
        
        console.print("\n[cyan]Analysis:[/cyan]")
        console.print(results["analysis"])
        
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