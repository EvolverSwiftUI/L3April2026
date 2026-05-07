"""
LangChain Agent with OpenAI-compatible Ollama model and tool calling.

Uses modern LangChain pattern (0.2+) with tool calling.
Loads configuration from .env file.
"""

import json
import os
from typing import Any
from pathlib import Path

from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI


# ============================================================================
# TOOL 1: Tavily Search (Official LangChain Integration)
# ============================================================================

def setup_tavily_tool():
    """Initialize Tavily search tool."""
    if not os.getenv("TAVILY_API_KEY"):
        raise ValueError("TAVILY_API_KEY environment variable not set")
    return TavilySearch(max_results=5)


# ============================================================================
# TOOL 2: Custom Email Lookup Tool
# ============================================================================

@tool
def lookup_email(name: str) -> str:
    """
    Look up email address for a given person.

    Args:
        name: Full name of the person

    Returns:
        Email address
    """
    # Mock implementation: convert name to email format
    email = f"{name.lower().replace(' ', '.')}@example.com"
    return email


# ============================================================================
# MODEL SETUP: OpenAI-compatible Ollama endpoint
# ============================================================================

def setup_model() -> ChatOpenAI:
    """
    Initialize ChatOpenAI with Ollama's OpenAI-compatible endpoint.

    Ollama must be running:
        ollama run gpt-oss-120b-cloud

    Configuration loaded from .env file.
    """
    model_name = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    api_key = os.getenv("OPENAI_API_KEY", "dummy-key")

    model = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=0.7,
        max_tokens=2048,
        timeout=10.0,
        request_timeout=10.0,
    )
    return model


# ============================================================================
# AGENT SETUP - Modern LangChain Pattern
# ============================================================================

class SimpleAgent:
    """Simple agent executor that handles tool calling."""

    def __init__(self, model: ChatOpenAI, tools: list):
        self.model = model
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}

        # Bind tools to model for tool calling
        self.model_with_tools = model.bind_tools(tools, tool_choice="auto")

    def invoke(self, query: str, max_iterations: int = 10) -> str:
        """Run the agent on a query."""
        messages = [
            HumanMessage(
                content="""You are a helpful assistant with access to two tools:

1. **tavily_search_results_json**: Use this for general knowledge questions, news, web search, or anything requiring internet access.
   - Use for: "latest news", "how to", "what is", "information about"

2. **lookup_email**: Use this when the user asks for someone's email address.
   - Use for: "what is X's email", "email of Y", "contact info for Z"

Always pick the right tool for the task. Be concise in your responses.

User query: """ + query
            )
        ]

        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")

            # Get model response
            response = self.model_with_tools.invoke(messages)
            messages.append(response)

            print(f"Model response: {response.content}")

            # Check if model wants to use tools
            if not response.tool_calls:
                # No more tool calls, return final answer
                return response.content

            # Process tool calls
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_input = tool_call["args"]

                print(f"Calling tool: {tool_name} with args: {tool_input}")

                # Execute tool
                if tool_name in self.tool_map:
                    tool_fn = self.tool_map[tool_name]
                    result = tool_fn.invoke(tool_input)
                    print(f"Tool result: {result}")

                    # Add tool result to messages
                    messages.append(
                        HumanMessage(
                            content=f"Tool '{tool_name}' returned: {result}"
                        )
                    )
                else:
                    messages.append(
                        HumanMessage(
                            content=f"Tool '{tool_name}' not found"
                        )
                    )

        return "Max iterations reached"


def create_email_agent() -> SimpleAgent:
    """Create an agent with tool calling support."""
    model = setup_model()
    tavily_tool = setup_tavily_tool()
    tools = [tavily_tool, lookup_email]

    return SimpleAgent(model, tools)


# ============================================================================
# EXECUTION EXAMPLES
# ============================================================================

def run_example_1(agent: SimpleAgent) -> None:
    """Example 1: Web search for latest AI news."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Search for Latest AI News")
    print("="*70)
    query = "latest news about AI and machine learning"
    print(f"Query: {query}\n")

    try:
        result = agent.invoke(query)
        print(f"\n[OK] Final Answer:\n{result}")
    except Exception as e:
        print(f"Error: {e}")


def run_example_2(agent: SimpleAgent) -> None:
    """Example 2: Email lookup for a person."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Email Lookup")
    print("="*70)
    query = "What is the email address of John Doe?"
    print(f"Query: {query}\n")

    try:
        result = agent.invoke(query)
        print(f"\n[OK] Final Answer:\n{result}")
    except Exception as e:
        print(f"Error: {e}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Load environment variables from .env file
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
        print(f"[OK] Loaded configuration from {env_path}")
        print(f"[OK] TAVILY_API_KEY={'set' if os.getenv('TAVILY_API_KEY') else 'not set'}\n")
    else:
        print(f"Warning: .env file not found at {env_path}\n")

    print("Initializing LangChain Agent with Ollama + Tool Calling...\n")

    try:
        # Create agent
        print("Creating agent...")
        agent = create_email_agent()
        print("[OK] Agent created successfully\n")

        # Run examples
        run_example_1(agent)
        run_example_2(agent)

        print("\n" + "="*70)
        print("Examples completed!")
        print("="*70)

    except ConnectionError as e:
        print(f"ERROR: Cannot connect to Ollama at http://localhost:11434")
        print(f"Details: {e}")
        print("Make sure Ollama is running: ollama run gpt-oss-120b-cloud")
    except TimeoutError as e:
        print(f"ERROR: Request timed out (Ollama not responding)")
        print(f"Details: {e}")
    except ValueError as e:
        print(f"ERROR: {e}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
