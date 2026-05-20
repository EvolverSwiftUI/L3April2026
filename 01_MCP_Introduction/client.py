import asyncio
from fastmcp import Client
from fastmcp.client.transports.stdio import StdioTransport   # Change in the import path for StdioTransport

async def main():
    transport = StdioTransport(
        #"python"
        "D:\\L3April2026\\.venv\\Scripts\\python.exe",           # ✅ command
        ["server.py"]       # ✅ args (separate list)
    )

    async with Client(transport) as client:
        tools = await client.list_tools()
        print("Available tools:", tools)
        
        # Call the "greet" tool with the name "Alice"
        result = await client.call_tool(
            "greet",
            {"name": "Sharad"}
        )
    
        print("Result:", result)

if __name__ == "__main__":
    asyncio.run(main())
    
    