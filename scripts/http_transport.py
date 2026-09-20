"""Bounded asynchronous JSON HTTP transport with reusable connections."""
import asyncio
import json
import sys


def run_async(coroutine, *, winloop=False):
    if winloop and sys.platform=='win32':
        import winloop as loop
        return loop.run(coroutine)
    return asyncio.run(coroutine)


class JsonClient:
    def __init__(self, concurrency, timeout=300, *, force_close=False, trace_configs=None):
        if concurrency<1 or timeout<=0:raise ValueError('Invalid HTTP limits')
        self.concurrency=concurrency;self.timeout=timeout
        self.force_close=force_close;self.trace_configs=trace_configs

    async def __aenter__(self):
        import aiohttp
        self.session=aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=self.concurrency,limit_per_host=self.concurrency,
                                           force_close=self.force_close),
            timeout=aiohttp.ClientTimeout(total=self.timeout),trust_env=False,
            cookie_jar=aiohttp.DummyCookieJar(),trace_configs=self.trace_configs)
        return self

    async def __aexit__(self,*exc):
        await self.session.close()

    async def post(self,url,payload,*,headers=None,trace_request_ctx=None):
        # Match the existing UTF-8 wire payload without blocking OS threads.
        headers={'Content-Type':'application/json','Accept':'application/json',**(headers or {})}
        async with self.session.post(url,data=json.dumps(payload,ensure_ascii=False).encode('utf-8'),
                                     headers=headers,trace_request_ctx=trace_request_ctx) as response:
            data=await response.read()
            if response.status>=400:
                raise RuntimeError(f'HTTP {response.status}: {data.decode("utf-8",errors="replace")[:2000]}')
            value=json.loads(data)
            if not isinstance(value,dict):raise ValueError('Expected JSON object')
            if value.get('error'):raise RuntimeError(str(value['error']))
            return value
