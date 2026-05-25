from fastmcp import FastMCP

mcp = FastMCP("My MCP Server")



@mcp.tool
def greet(name: str) -> str:
    ''' Greet a person by name'''
    return f"Hello, {name}!"


@mcp.tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b



if __name__ == "__main__":
    mcp.run()
    #mcp.run(transport="streamable-http",host="127.0.0.1", port=8000)