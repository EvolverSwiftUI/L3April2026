import asyncio
import json

from gateway import Gateway
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)


def convert_tools(mcp_tools):
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema
            }
        }
        for t in mcp_tools
    ]


async def main():

    gateway = Gateway()
    await gateway.start()

    print("\n---- TEST CALL ----")

    try:
        # Collect all MCP tools ONCE
        mcp_tools = []

        for entry in gateway.tool_map.values():
            tools = await entry['client'].list_tools()
            mcp_tools.extend(tools)

        tools_def = convert_tools(mcp_tools)

        # Chat loop
        while True:

            user_input = input("\nEnter your request: ")

            if user_input.lower() in ["exit", "quit"]:
                break

            system_prompt = """
                You are an assistant that:
                - uses tools when required
                - sends resources when required
                - follows strict validation rules
                """

            response = client.chat.completions.create(
                model="gpt-oss:120b-cloud",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                tools=tools_def
            )

            message = response.choices[0].message

            if message.tool_calls:

                tool_call = message.tool_calls[0]

                tool_name = tool_call.function.name

                # FIXED TYPO HERE
                arguments = json.loads(tool_call.function.arguments)

                print("\nLLM Selected:", tool_name)
                print("Arguments:", arguments)

                result = await gateway.call(tool_name, arguments)

                print("\nTool Result:", result)

            else:
                print("\nLLM Response:", message.content)

    finally:
        await gateway.stop()


if __name__ == "__main__":
    asyncio.run(main())