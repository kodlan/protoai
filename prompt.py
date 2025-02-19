import os
from typing import Optional

import click
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pydantic import BaseModel
from rich.console import Console

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize Rich console
console = Console(width=150)

# Initial planning prompt to identify messages/enums
INITIAL_PLANNING_TEMPLATE = """
Your task is to identify all Protocol Buffer messages and enums that need to be examined to answer the user's question.
Then generate queries to fetch their summary documents.

Question: {question}

Example summary documents in the database:
1. Message summaries: "Message Poll in version proto_v1 (package: polls) has 5 fields: title, description, options, created_at, author. This message is defined at path Poll."
2. Enum summaries: "Enum PollStatus in version proto_v1 is defined at path PollStatus. It contains values: DRAFT: 0, ACTIVE: 1, CLOSED: 2"

Generate specific queries that will match these summary documents.
Query format is "Message <message/enum name> in version <proto version>".
For example "Message Advertisement in version proto_v1"

Output format:
<list all messages and enums that need to be looked up. Do not output any other text only list queries. One query per line>
"""

# Detailed planning prompt that uses summaries
DETAILED_PLANNING_TEMPLATE = """
Using the summaries below, generate specific queries to fetch detailed information about fields and values of the proto messages and enums.
Message summaries below can contain summaries for multiple documents.

Example summary documents:
1. Message summaries: "Message Notification in version proto_v2 (package: socialmedia) has 7 fields: id, user_id, type, message, read, timestamp, category. This message is defined at path Notification."
2. Enum summaries: "Enum PollStatus in version proto_v1 is defined at path PollStatus. It contains values: DRAFT: 0, ACTIVE: 1, CLOSED: 2"

For each summary generate a query for every fields of the proto message.
Generate specific queries using exact names from the summaries.
Make sure queries exactly match the document format in the database. 
For example message summary generate following queries:
    In message Notification (version: proto_v2), there is a field called id
    In message Notification (version: proto_v2), there is a field called user_id
    In message Notification (version: proto_v2), there is a field called type
    In message Notification (version: proto_v2), there is a field called message
    In message Notification (version: proto_v2), there is a field called read
    In message Notification (version: proto_v2), there is a field called timestamp
    In message Notification (version: proto_v2), there is a field called category

If there are multiple summaries present make sure to generate queries for each summary in the way described above.    
    
Available Summaries:
{summaries}

Output format:
<list specific queries for each field/value needed. Do not include any other text in the output. It should only be one query per line.>
"""

# Final chain prompt - Generating answer
FINAL_ANSWER_TEMPLATE = """
Based on the search results below that contain all the relevant information about proto messages, provide a clear and complete answer to the user's question.
Provided information contains text desciption of the proto messages, enums and services.
If you have messages or enums or services with the same name but different versions compare what fields do they have.
If field has the same name in two messages that have different version consider it the same field. If at the same time they have different types or proto numbers mention this 
as a difference in the final result. If the same number is used in the same message with different versions but for fields with different names mention this as a difference.

Search Results:
{search_results}

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
Your task is to compare queries and their results. You should check if the information in the query matches with the result provided.

For example if you have following query: "In message Notification (version: proto_v2), there is a field called id".
There should be an answer that contains the information about id field in the Notification message and it should mention that version is the proto_v2.
Note that version and names should match exactly in query and in the response.

So example answer for this query could look like:
"Query: In message Notification (version: proto_v2), there is a field called id\nScore: 0.1048\nContent: In message Notification (package: 
socialmedia, version: proto_v2), there is a field called id. It has proto number 1 and type int64. The field is not repeated. This field belongs to 
message at path Notification."
Ignore score information.

Example of query that with incorrect response:
    query: In message Test (version: v5), there is a field called name
    response: In message Test (package: test, version: v3), there is a field called name. It has proto number 5 and type int32. The field is 
              repeated. This field belongs to message at path Test.
This response is not correct as the versions don't match - v5 is not equal to v3.

Original Plan (contains all the queries. one query per line):
{query_plan}

Search Results (contains all the results, in the format described above):
{search_results}

Analise if every query has it's matching response. 
If any information is missing, add query that is missing information to the output as described below.
It also may happen that there is some additional response that does not match any query - ignore this information.

Output your response in this format:
STATUS: [COMPLETE or INCOMPLETE]

FOUND:
<list what information was successfully retrieved>

MISSING:
<list any missing information or write "None" if complete>

NEW QUERIES:
<if INCOMPLETE, list specific queries that ary missing responses. It should only be one query per line.>
<if COMPLETE, write "None">

{format_instructions}
"""

class ValidationChain(LLMChain):
    class Config:
        arbitrary_types_allowed = True

    class ValidationInput(BaseModel):
        query_plan: str
        search_results: str
        question: str  # Keep in input model but don't use in prompt

    class ValidationOutput(BaseModel):
        status: str
        found: str
        missing: str
        new_queries: str

    def _call(self, inputs: dict) -> dict:
        validated_inputs = self.ValidationInput(**inputs)
        # Get raw LLM output but only use query_plan and search_results
        raw_response = self.llm.invoke(self.prompt.format(
            query_plan=inputs["query_plan"],
            search_results=inputs["search_results"]
        ))
        response_text = raw_response.content
        
        # Parse the response to extract status and queries
        lines = response_text.split("\n")
        output = {
            "status": "INCOMPLETE",
            "found": "",
            "missing": "",
            "new_queries": "",
            "original_question": inputs["question"]  # Pass through the question
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

class InitialPlanningChain(LLMChain):
    class Config:
        arbitrary_types_allowed = True

    def _call(self, inputs: dict) -> dict:
        result = super()._call(inputs)
        # Pass through the original question
        return {
            "initial_plan": result["initial_plan"],
            "original_question": inputs["question"]
        }

class SummaryFetchChain(LLMChain):
    class Config:
        arbitrary_types_allowed = True

    vectorstore: Chroma

    def _call(self, inputs: dict) -> dict:
        queries = inputs["initial_plan"].split("\n")
        results = []
        for query in queries:
            if not query.strip():
                continue
            search_results = self.vectorstore.similarity_search_with_score(query, k=1)
            for doc, score in search_results:
                results.append(f"Query: {query}\nScore: {score:.4f}\nContent: {doc.page_content}\n")
        # Pass through the original question
        return {
            "summaries": "\n".join(results),
            "original_question": inputs["original_question"]
        }

class DetailedPlanningChain(LLMChain):
    class Config:
        arbitrary_types_allowed = True

    def _call(self, inputs: dict) -> dict:
        # Use the original question that was passed through
        result = super()._call({"summaries": inputs["summaries"]})
        return result

class ProtoSchemaChain:
    def __init__(self, vectorstore: Chroma):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.vectorstore = vectorstore
        self.setup_chains()

    def setup_chains(self):
        # Initial planning chain
        self.initial_planning_chain = InitialPlanningChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=INITIAL_PLANNING_TEMPLATE,
                input_variables=["question"],
            ),
            output_key="initial_plan"
        )

        # Summary fetch chain
        self.summary_fetch_chain = SummaryFetchChain(
            vectorstore=self.vectorstore,
            llm=self.llm,
            prompt=PromptTemplate(
                template="Fetching summaries...",
                input_variables=["initial_plan"]
            ),
            output_key="summaries"
        )

        # Detailed planning chain
        self.detailed_planning_chain = DetailedPlanningChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=DETAILED_PLANNING_TEMPLATE,
                input_variables=["summaries"]
            ),
            output_key="query_plan"
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

        # Final answer chain
        self.answer_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=FINAL_ANSWER_TEMPLATE,
                input_variables=["search_results", "question"],
                partial_variables={"format_instructions": "Provide a clear and complete answer."}
            ),
            output_key="response"
        )

        # Add validation chain
        self.validation_chain = ValidationChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=VALIDATION_TEMPLATE,
                input_variables=["query_plan", "search_results"],
                partial_variables={"format_instructions": "Ensure clear separation between sections."}
            ),
            output_key="response"
        )

        # Remove sequential chain as we're now handling the flow in run()
        self.chain = None

    def run(self, question: str):
        """Run the full chain with validation loop."""
        try:
            # Step divider function
            def print_step(step_name: str):
                console.print(f"\n{'='*20} {step_name} {'='*20}", style="bold cyan")

            print_step("INITIAL PLANNING")
            # Initial planning with question pass-through
            initial_result = self.initial_planning_chain.invoke({
                "question": question
            })
            console.print("\n[green]Initial Planning Output:[/green]")
            console.print(initial_result["initial_plan"])

            print_step("SUMMARY FETCH")
            # Fetch summaries, passing through the question
            summary_result = self.summary_fetch_chain.invoke({
                "initial_plan": initial_result["initial_plan"],
                "original_question": initial_result["original_question"]
            })
            console.print("\n[green]Summary Fetch Output:[/green]")
            console.print(summary_result["summaries"])

            print_step("DETAILED PLANNING")
            # Detailed planning using passed through question
            detailed_result = self.detailed_planning_chain.invoke({
                "summaries": summary_result["summaries"]
            })
            console.print("\n[green]Detailed Planning Output:[/green]")
            console.print(detailed_result["query_plan"])

            # Initialize query_plan from detailed planning result
            query_plan = detailed_result["query_plan"]

            print_step("SEARCH AND VALIDATION LOOP")
            # Initialize results storage for validation loop
            all_search_results = []
            iteration = 0
            max_iterations = 10
            
            while iteration < max_iterations:
                iteration += 1
                console.print(f"\n[bold magenta]{'='*10} Iteration {iteration} {'='*10}[/bold magenta]")
                
                console.print("\n[blue]Search Chain Input:[/blue]")
                console.print({"query_plan": query_plan, "question": question})
                
                current_results = self.search_chain.run(
                    query_plan=query_plan,
                    question=question
                )
                console.print("\n[green]Search Chain Output:[/green]")
                console.print(current_results)
                
                all_search_results.append(current_results)
                merged_results = "\n\n".join(all_search_results)
                
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
                
                console.print("\n[yellow]Validation Results:[/yellow]")
                console.print(f"Status: {validation_data['status']}")
                console.print("\nFound Information:")
                console.print(validation_data['found'])
                
                if validation_data['missing'] != "None":
                    console.print("\nMissing Information:")
                    console.print(validation_data['missing'])
                
                if validation_data['status'] == "COMPLETE":
                    console.print("\n[green]All required information has been found![/green]")
                    break
                    
                if validation_data['new_queries'] == "None":
                    console.print("\n[yellow]No new queries to execute.[/yellow]")
                    break
                    
                # Update query plan with new queries
                query_plan = validation_data['new_queries']  # Store as query_plan, not detailed_result
                console.print("\n[blue]New Queries Generated:[/blue]")
                console.print(query_plan)
                
                if iteration == max_iterations:
                    console.print("\n[yellow]Reached maximum iterations.[/yellow]")

            print_step("FINAL ANSWER")
            # Proceed with answer generation
            console.print("\n[blue]Answer Chain Input:[/blue]")
            console.print({
                "search_results": merged_results,
                "question": question
            })
            
            answer = self.answer_chain.run(search_results=merged_results, question=question)
            console.print("\n[green]Answer Chain Output:[/green]")
            console.print(answer)
            
            return {
                "query_plan": query_plan,
                "search_results": merged_results,
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