import asyncio
import os
import sys

# Add the project root to sys.path to ensure absolute imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_use import MCPClient
from mcp_use.agents.adapters.langchain_adapter import LangChainAdapter
from src.prompts import DATA_ANALYST_SYSTEM_PROMPT
from src.model import llm
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

console = Console()

async def main():
    """
    Main entry point for the AI Data Analyst agent.
    Connects to the MCP server, loads tools, and initializes a LangChain agent.
    """
    # 1. Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", "dataset.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        return
    
    # Load data for the agent to reference
    df = pd.read_csv(csv_path)
    console.print(Panel(
        f"[bold blue]File:[/bold blue] {csv_path}\n"
        f"[bold blue]Rows:[/bold blue] {len(df)}\n"
        f"[bold blue]Columns:[/bold blue] {len(df.columns)}",
        title="[bold blue]Dataset Loaded[/bold blue]",
        border_style="blue"
    ))

    # 2. Define server configuration for mcp-use
    server_path = os.path.join(base_dir, "src", "server.py")
    
    if not os.path.exists(server_path):
        print(f"Error: Server file not found at {server_path}")
        return

    mcp_config = {
        "mcpServers": {
            "data-analyst": {
                "command": "python",
                "args": [server_path]
            }
        }
    }

    # 3. Initialize the MCP client with the server config
    # mcp-use abstracts the JSON-RPC communication over stdio
    client = MCPClient(config=mcp_config)
    
    # 3. Use the LangChainAdapter to bridge MCP tools to LangChain format
    # This automatically converts MCP Tools, Resources, and Prompts
    adapter = LangChainAdapter()
    
    console.print("\n[bold yellow]--- Connecting to MCP Server ---[/bold yellow]")
    try:
        # Discover and convert tools to LangChain BaseTool objects
        tools = await adapter.create_tools(client)
        console.print(f"[green]✓[/green] Found [bold]{len(tools)}[/bold] tools: [italic]{[t.name for t in tools]}[/italic]")
        
        # 4. Create the LangChain Agent
        # Tool-calling agents are efficient as they use the LLM's native function calling capabilities
        prompt = ChatPromptTemplate.from_messages([
            ("system", DATA_ANALYST_SYSTEM_PROMPT),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # Initialize the tool-calling agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        # 5. Run the analytics workflow
        query = (
            f"I have a dataset loaded from {csv_path}. "
            "Please start the pre-processing workflow by filtering the data and then computing imputation decisions. "
            "Show me the initial diagnostics."
        )
        
        console.print(Panel(f"[bold cyan]User Instruction:[/bold cyan]\n{query}", title="[bold]Workflow Initiated[/bold]", border_style="cyan"))
        
        try:
            with console.status("[bold green]Agent is thinking and analyzing dataset...", spinner="dots"):
                response = await agent_executor.ainvoke({"input": query})
            
            console.print(Panel(Markdown(response['output']), title="[bold green]AI Data Analyst Response[/bold green]", border_style="green", padding=(1, 2)))
            
        except Exception as e:
            console.print(f"[bold red]An error occurred during agent execution:[/bold red] {e}")
            
    finally:
        # Ensure that MCP sessions are closed properly
        await client.close_all_sessions()

if __name__ == "__main__":
    # Entry point for the script
    asyncio.run(main())
