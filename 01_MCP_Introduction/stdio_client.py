import asyncio
from fastmcp import Client
from fastmcp.client.transports.stdio import StdioTransport   # Change in the import path for StdioTransport

async def main():
    transport = StdioTransport(
        #"python"
        "D:\\L3April2026\\.venv\\Scripts\\python.exe",           # ✅ command
        ["stdio_server.py"]       # ✅ args (separate list)
    )

    async with Client(transport) as client:
        tools = await client.list_tools()
        #print("Available tools:", tools)
        
        print(tools[1].inputSchema)
        
        # Call the "greet" tool with the name "Alice"
        result = await client.call_tool(
            "add",
            {"a": "Ram", "b": 3}
        )
    
        print("Result:", result)

if __name__ == "__main__":
    asyncio.run(main())
    
    