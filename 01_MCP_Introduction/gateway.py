import asyncio
from fastmcp import Client



class Gateway:
    def __init__(self):
        self.clients = {}
        self.tool_map = {}
        
    
        
        
    async def start(self):
        # Connect to math server and user server
        self.clients["math"] = Client("http://127.0.0.1:8001/mcp")
        self.clients["user"] = Client("http://127.0.0.1:8002/mcp")
            
        for name,client in self.clients.items():
            await client.__aenter__()  # Manually enter the async context
                
            
            tools = await client.list_tools()  # List tools from all clients and build tool map
            
            for t in tools:
                print(f"Registering tool: {t.name} from client: {name}")
                
                self.tool_map[t.name] = {
                    "client": client,
                    "server": name
                }
                
                
    async def call(self,tool,args):
        if tool not in self.tool_map:
            raise Exception(f"Tool {tool} not found")
            
        entry = self.tool_map[tool]

            
        client = entry["client"]
        server_name = entry["server"]                
            
        print(f" [Gateway] Routing '{tool}' -> {server_name}")  
            
        result = await client.call_tool(tool,args)
            
        return result.data
        
    async def stop(self):
        for c in self.clients.values():
            await c.__aexit__(None, None, None)  # Manually exit the async context
                
                
async def main():
    gateway = Gateway()
    
    await gateway.start()
    
    print("\n  ---- TEST CALL ---- ")
    
    #print(await gateway.call("add", {"a": 10, "b": 5}))
    print(await gateway.call("create_user", {"user": {"name": "Alice", "email": "alice@example.com"}}))
    
    await gateway.stop()
    
    
    
if __name__ == "__main__":
    asyncio.run(main())
    
    
    # This is MCP acting like API Gateway 
    
        
        
                    
        



