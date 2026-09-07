import asyncio

from model_harness_g0.mcp_client import call


async def test_actual_mcp_initialize_list_call():
    assert await asyncio.wait_for(call("nonce-with-汉字"), 30) == "mcp:nonce-with-汉字"
