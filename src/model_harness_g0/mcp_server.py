"""Synthetic, public-data-only MCP fixture."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("g0-fixture")


@mcp.tool()
def echo(text: str) -> str:
    """Echo a public nonce through MCP."""
    return "mcp:" + text


if __name__ == "__main__":
    mcp.run(transport="stdio")
