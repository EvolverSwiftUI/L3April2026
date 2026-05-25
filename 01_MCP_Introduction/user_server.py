from fastmcp import FastMCP

from pydantic import BaseModel, EmailStr

from uuid import uuid4

mcp = FastMCP("user-server")


users ={}

class User(BaseModel):
    name:str
    email:EmailStr
    
@mcp.tool
def create_user(user: User):
    user_id= str(uuid4())
    users[user_id]= user.model_dump()
    return { "user_id": user_id, "user": users[user_id]}

@mcp.tool
def get_user(user_id: str):
    return users.get(user_id, {"error": "User not found"})

@mcp.prompt('user-creation-guide')
def user_creation_prompt():
    return """
        When creating a user:
        - Always provide a name and a valid email address.
        - Email must be a valid email format
        - Do not guess missing values 
    """
    
    
    
@mcp.resource("users://all")
def all_users():
    return users

    


if __name__ == "__main__":
    mcp.run(transport="streamable-http",host="127.0.0.1",port=8002)
    
    


