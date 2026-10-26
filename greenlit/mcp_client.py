import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import timedelta

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import ROOT


class Tools:
    def __init__(self, session):
        self.session = session
        self.available = []
        self.trace = []

    async def call(self, name: str, arguments: dict) -> dict:
        result = await self.session.call_tool(name, arguments, read_timeout_seconds=timedelta(minutes=20))
        if result.isError:
            # Avoid forwarding raw MCP tool errors containing provider responses.
            raise RuntimeError(f'MCP tool {name} failed. Check provider configuration and input requirements.')
        payload = getattr(result, 'structuredContent', None)
        if payload is None:
            chunks = [item.text for item in result.content if getattr(item, 'type', None) == 'text']
            payload = json.loads('\n'.join(chunks))
        if isinstance(payload,dict) and payload.get('_greenlit_error'):
            raise ValueError(str(payload['_greenlit_error']))
        self.trace.append({'tool': name, 'transport': 'stdio', 'ok': True})
        return payload


@asynccontextmanager
async def connect_tools():
    params = StdioServerParameters(command=sys.executable, args=['-m', 'greenlit.mcp_server'],
                                  cwd=str(ROOT), env=dict(os.environ))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = Tools(session)
            listing = await session.list_tools()
            tools.available = [tool.name for tool in listing.tools]
            yield tools
