import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def call(text):
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "model_harness_g0.mcp_server"]
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as client:
            await client.initialize()
            listed = await client.list_tools()
            if "echo" not in [t.name for t in listed.tools]:
                raise RuntimeError("MCP fixture did not advertise echo")
            result = await client.call_tool("echo", {"text": text})
            if result.isError:
                raise RuntimeError("MCP tool failed")
            return "".join(x.text for x in result.content if x.type == "text")


if __name__ == "__main__":
    print(json.dumps({"text": asyncio.run(asyncio.wait_for(call(sys.argv[1]), 30))}))
