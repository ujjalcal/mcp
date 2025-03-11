import requests
import json
import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import argparse
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich import print as rprint
from rich.prompt import Prompt, Confirm
import textwrap
from openai import OpenAI

# Load environment variables
load_dotenv()


# Configuration
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

console = Console()

import os
from openai import OpenAI
from dotenv import load_dotenv



# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



class MCPClient:
    """Client for interacting with the Neo4j MCP Server"""
    
    def __init__(self, server_url: str = MCP_SERVER_URL):
        self.server_url = server_url
        self.schema = None
        self.prompts = None
    
    def get_schema(self) -> Dict:
        """Fetch the database schema from the MCP server"""
        try:
            response = requests.get(f"{self.server_url}/schema")
            response.raise_for_status()
            self.schema = response.json()
            return self.schema
        except requests.exceptions.RequestException as e:
            console.print(f"[bold red]Error fetching schema: {str(e)}[/bold red]")
            return None
    
    def get_prompts(self) -> List[Dict]:
        """Fetch the available analysis prompts from the MCP server"""
        try:
            response = requests.get(f"{self.server_url}/prompts")
            response.raise_for_status()
            self.prompts = response.json()
            return self.prompts
        except requests.exceptions.RequestException as e:
            console.print(f"[bold red]Error fetching prompts: {str(e)}[/bold red]")
            return None
    
    def execute_query(self, cypher: str, parameters: Dict = None) -> Dict:
        """Execute a Cypher query against the Neo4j database"""
        if parameters is None:
            parameters = {}
            
        try:
            response = requests.post(
                f"{self.server_url}/query",
                json={"cypher": cypher, "parameters": parameters}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            console.print(f"[bold red]Error executing query: {str(e)}[/bold red]")
            if hasattr(e, 'response') and e.response is not None:
                console.print(f"[bold red]Server response: {e.response.text}[/bold red]")
            return None

    def display_schema(self):
        """Display the database schema in a readable format"""
        if not self.schema:
            self.get_schema()
            
        if not self.schema:
            return
            
        # Display node labels
        node_table = Table(title="Node Labels")
        node_table.add_column("Label", style="cyan")
        node_table.add_column("Count", style="magenta")
        node_table.add_column("Properties", style="green")
        
        for node in self.schema.get("nodes", []):
            node_table.add_row(
                node["label"],
                str(node["count"]),
                ", ".join(node["properties"])
            )
            
        console.print(node_table)
        
        # Display relationship types
        rel_table = Table(title="Relationship Types")
        rel_table.add_column("Type", style="cyan")
        rel_table.add_column("Count", style="magenta")
        rel_table.add_column("Source → Target", style="yellow")
        rel_table.add_column("Properties", style="green")
        
        for rel in self.schema.get("relationships", []):
            rel_table.add_row(
                rel["type"],
                str(rel["count"]),
                f"{' | '.join(rel['source_labels'])} → {' | '.join(rel['target_labels'])}",
                ", ".join(rel["properties"])
            )
            
        console.print(rel_table)
    
    def display_prompts(self):
        """Display available analysis prompts"""
        if not self.prompts:
            self.get_prompts()
            
        if not self.prompts:
            return
            
        for i, prompt in enumerate(self.prompts, 1):
            console.print(f"[bold cyan]{i}. {prompt['name']}[/bold cyan]")
            console.print(f"[italic]{prompt['description']}[/italic]")
            console.print()
    
    def select_prompt(self) -> Dict:
        """Let the user select a prompt and fill in parameters"""
        if not self.prompts:
            self.get_prompts()
            
        if not self.prompts:
            return None
            
        self.display_prompts()
        
        # Select prompt
        prompt_index = Prompt.ask(
            "Select a prompt number", 
            choices=[str(i) for i in range(1, len(self.prompts) + 1)]
        )
        
        selected_prompt = self.prompts[int(prompt_index) - 1]
        console.print(f"\n[bold]Selected: {selected_prompt['name']}[/bold]\n")
        
        # Display prompt details
        prompt_text = selected_prompt["prompt"]
        console.print(Panel(prompt_text, title="Prompt Template"))
        
        # Fill in parameters
        parameters = {}
        example_parameters = selected_prompt.get("example_parameters", {})
        
        if example_parameters:
            console.print("\n[bold]Example parameters:[/bold]")
            for key, value in example_parameters.items():
                console.print(f"  {key}: {value}")
        
        # Extract parameter placeholders from the prompt
        import re
        placeholders = re.findall(r'\{([^{}]+)\}', prompt_text)
        unique_placeholders = set(placeholders)
        
        if unique_placeholders:
            console.print("\n[bold]Enter values for parameters:[/bold]")
            for param in unique_placeholders:
                default = example_parameters.get(param, "")
                value = Prompt.ask(f"  {param}", default=str(default))
                parameters[param] = value
        
        # Extract and modify sample Cypher query
        sample_query_match = re.search(r'```\s*([\s\S]+?)\s*```', prompt_text)
        if sample_query_match:
            sample_query = sample_query_match.group(1).strip()
            
            # Replace placeholders with user values
            for param, value in parameters.items():
                sample_query = sample_query.replace(f"{{{param}}}", value)
            
            console.print("\n[bold]Generated Cypher query:[/bold]")
            syntax = Syntax(sample_query, "cypher", theme="monokai", line_numbers=True)
            console.print(syntax)
            
            if Confirm.ask("Execute this query?", default=True):
                return self.execute_prompt_query(sample_query)
        else:
            console.print("[yellow]No sample query found in the prompt.[/yellow]")
        
        return None
    
    def execute_prompt_query(self, query: str) -> Dict:
        """Execute the query generated from a prompt template"""
        result = self.execute_query(query)
        if result:
            self.display_query_results(result)
        return result
    
    def display_query_results(self, result: Dict):
        """Display query results in a readable format"""
        records = result.get("results", [])
        metadata = result.get("metadata", {})
        
        if not records:
            console.print("[yellow]No results returned.[/yellow]")
            return
            
        # Get all unique keys from all records
        all_keys = set()
        for record in records:
            all_keys.update(record.keys())
        
        # Create a table with all columns
        table = Table(title=f"Query Results ({len(records)} records)")
        for key in all_keys:
            table.add_column(key)
        
        # Add rows to the table
        for record in records:
            row_values = []
            for key in all_keys:
                value = record.get(key, "")
                
                # Handle different data types for display
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, indent=2)
                    # Truncate long values
                    if len(value) > 50:
                        value = value[:47] + "..."
                elif value is None:
                    value = ""
                
                row_values.append(str(value))
            
            table.add_row(*row_values)
        
        console.print(table)
        
        # Display metadata
        if metadata:
            console.print("\n[bold]Query Metadata:[/bold]")
            for key, value in metadata.items():
                console.print(f"  {key}: {value}")
    
    def interactive_query(self):
        """Allow the user to enter a custom Cypher query"""
        console.print("\n[bold]Enter a Cypher query:[/bold]")
        console.print("[italic](Press Enter twice when finished)[/italic]")
        
        lines = []
        while True:
            line = input()
            if not line and lines and not lines[-1]:
                # Empty line after content, break
                break
            lines.append(line)
        
        query = "\n".join(lines).strip()
        
        if not query:
            console.print("[yellow]No query entered.[/yellow]")
            return
        
        syntax = Syntax(query, "cypher", theme="monokai", line_numbers=True)
        console.print("\n[bold]Executing query:[/bold]")
        console.print(syntax)
        
        result = self.execute_query(query)
        if result:
            self.display_query_results(result)


class MCPClientWithLLM(MCPClient):
    """Extended MCP Client with OpenAI LLM integration"""
    
    def __init__(self, server_url=MCP_SERVER_URL, model="gpt-4"):
        super().__init__(server_url)
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model
    
    def generate_query_with_llm(self, user_input, schema=None):
        """Use OpenAI to generate a Cypher query based on user input and schema"""
        if not schema:
            schema = self.get_schema()
            
        # Create a system message with the database schema
        system_message = f"""
        You are a Neo4j database expert. Given the following database schema:
        
        Nodes: {', '.join([node['label'] for node in schema['nodes']])}
        Relationships: {', '.join([rel['type'] for rel in schema['relationships']])}
        
        Generate a Cypher query that answers the user's question. Return ONLY the Cypher query without any explanations.
        """
        
        # Call the OpenAI API
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input}
            ],
            temperature=0.1  # Low temperature for more deterministic outputs
        )
        
        # Extract the generated Cypher query
        cypher_query = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if cypher_query.startswith("``````"):
            cypher_query = cypher_query.strip("```")
            if cypher_query.startswith("cypher"):
                cypher_query = cypher_query[6:].strip()
        
        return cypher_query
    
    def analyze_results_with_llm(self, user_query, results):
        """Use OpenAI to analyze and explain query results"""
        if not results:
            return "No results found."
            
        # Create a prompt for analyzing the results
        prompt = f"""
        The user asked: "{user_query}"
        
        The database returned these results:
        {results}
        
        Please analyze these results and provide a clear, concise explanation.
        """
        
        # Call the OpenAI API
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        # Correctly access the content of the response
        return response.choices[0].message.content

def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(description="Neo4j MCP Client")
    parser.add_argument("--server", help="MCP server URL", default=MCP_SERVER_URL)
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Schema command
    subparsers.add_parser("schema", help="Display database schema")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Execute a Cypher query")
    query_parser.add_argument("--file", help="File containing the Cypher query")
    query_parser.add_argument("--query", help="Cypher query string")
    
    # Prompts command
    prompt_parser = subparsers.add_parser("prompts", help="Work with analysis prompts")
    prompt_parser.add_argument("--list", action="store_true", help="List available prompts")
    prompt_parser.add_argument("--select", action="store_true", help="Select and use a prompt")
    
    # Interactive mode
    subparsers.add_parser("interactive", help="Start interactive mode")
    
    args = parser.parse_args()
    
    client = MCPClientWithLLM(server_url=args.server)
    
    if args.command == "schema":
        client.display_schema()
    
    elif args.command == "query":
        if args.file:
            try:
                with open(args.file, 'r') as f:
                    query = f.read().strip()
            except Exception as e:
                console.print(f"[bold red]Error reading file: {str(e)}[/bold red]")
                return
        elif args.query:
            query = args.query
        else:
            client.interactive_query()
            return
            
        result = client.execute_query(query)
        if result:
            client.display_query_results(result)
    
    elif args.command == "prompts":
        if args.list:
            client.display_prompts()
        elif args.select:
            client.select_prompt()
        else:
            client.display_prompts()
            client.select_prompt()
    
    elif args.command == "interactive" or not args.command:
        llm_interactive_mode(client)
    
    else:
        parser.print_help()

def interactive_mode(client: MCPClient):
    """Run the client in interactive mode"""
    console.print("[bold]Neo4j MCP Client[/bold] - Interactive Mode")
    console.print("Type 'help' for available commands, 'exit' to quit\n")
    
    while True:
        command = Prompt.ask("mcp").lower()
        
        if command == "exit" or command == "quit":
            break
            
        elif command == "help":
            console.print("\n[bold]Available commands:[/bold]")
            console.print("  schema    - Display database schema")
            console.print("  query     - Enter and execute a Cypher query")
            console.print("  prompts   - List and select analysis prompts")
            console.print("  examples  - Show example queries")
            console.print("  clear     - Clear the screen")
            console.print("  exit      - Exit the client\n")
            
        elif command == "schema":
            client.display_schema()
            
        elif command == "query":
            client.interactive_query()
            
        elif command == "prompts":
            client.select_prompt()
            
        elif command == "examples":
            console.print("\n[bold]Example queries:[/bold]")
            examples = [
                ("Get all node labels", "MATCH (n) RETURN DISTINCT labels(n) AS labels, COUNT(*) AS count"),
                ("Get all relationship types", "MATCH ()-[r]->() RETURN DISTINCT type(r) AS type, COUNT(*) AS count"),
                ("Find a specific node", "MATCH (n:Loan {loanId: 105}) RETURN n"),
                ("Find connected nodes", "MATCH (n:Borrower)-[r]-(m) RETURN n.name, type(r), m LIMIT 10"),
                ("Find paths between nodes", "MATCH path = (a:Borrower)-[*1..3]-(b:Borrower) WHERE a.borrowerId <> b.borrowerId RETURN path LIMIT 5"),
            ]
            
            for i, (desc, query) in enumerate(examples, 1):
                console.print(f"\n[bold cyan]{i}. {desc}[/bold cyan]")
                syntax = Syntax(query, "cypher", theme="monokai")
                console.print(syntax)
                
            example_index = Prompt.ask(
                "\nSelect an example to run (or 0 to skip)", 
                choices=["0"] + [str(i) for i in range(1, len(examples) + 1)],
                default="0"
            )
            
            if example_index != "0":
                query = examples[int(example_index) - 1][1]
                result = client.execute_query(query)
                if result:
                    client.display_query_results(result)
            
        elif command == "clear":
            os.system('cls' if os.name == 'nt' else 'clear')
            
        else:
            console.print("[yellow]Unknown command. Type 'help' for available commands.[/yellow]")

def llm_interactive_mode(client: MCPClientWithLLM):
    """Run the client in LLM-assisted interactive mode"""
    console.print("[bold]Neo4j MCP Client with OpenAI[/bold] - Interactive Mode")
    console.print("Type 'help' for available commands, 'exit' to quit\n")
    
    while True:
        command = Prompt.ask("mcp").lower()
        
        if command == "exit" or command == "quit":
            break
            
        elif command == "help":
            console.print("\n[bold]Available commands:[/bold]")
            console.print("  schema    - Display database schema")
            console.print("  query     - Enter and execute a Cypher query")
            console.print("  ask       - Ask a natural language question")
            console.print("  prompts   - List and select analysis prompts")
            console.print("  clear     - Clear the screen")
            console.print("  exit      - Exit the client\n")
            
        elif command == "schema":
            client.display_schema()
            
        elif command == "query":
            client.interactive_query()
            
        elif command == "ask":
            question = Prompt.ask("\n[bold]Enter your question about the database[/bold]")
            console.print("[italic]Generating Cypher query...[/italic]")
            
            # Generate Cypher query using LLM
            cypher_query = client.generate_query_with_llm(question)
            
            # Display and execute the query
            console.print("\n[bold]Generated Cypher query:[/bold]")
            syntax = Syntax(cypher_query, "cypher", theme="monokai", line_numbers=True)
            console.print(syntax)
            
            if Confirm.ask("Execute this query?", default=True):
                result = client.execute_query(cypher_query)
                if result:
                    client.display_query_results(result)
                    
                    # Analyze results with LLM
                    console.print("\n[bold]Analysis:[/bold]")
                    analysis = client.analyze_results_with_llm(question, result)
                    console.print(Panel(analysis, title="AI Analysis"))
            
        elif command == "prompts":
            client.select_prompt()
            
        elif command == "clear":
            os.system('cls' if os.name == 'nt' else 'clear')
            
        else:
            console.print("[yellow]Unknown command. Type 'help' for available commands.[/yellow]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold]Exiting...[/bold]")
        sys.exit(0)