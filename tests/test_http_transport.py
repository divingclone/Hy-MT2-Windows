"""Exercise real pooled HTTP I/O, bounded concurrency, ordering and failures."""
import asyncio
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from aiohttp import web

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from translate import Sampling,translate_many_async
from benchmark_backend import payload_for,run_async_pass


class HttpClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.payloads=[];self.authorization=[];self.connections=set();self.active=0;self.peak=0
        self.entered=asyncio.Event()
        async def handler(request):
            payload=await request.json();self.payloads.append(payload)
            self.authorization.append(request.headers.get('Authorization'))
            self.connections.add(id(request.transport));self.active+=1;self.peak=max(self.peak,self.active)
            self.entered.set()
            try:
                await asyncio.sleep(.03 if payload['seed']%2 else .01)
                text=payload['messages'][0]['content'].split('\n',1)[1]
                if text=='http-error':return web.json_response({'error':'busy'},status=429)
                if text=='malformed':return web.json_response({'choices':[]})
                if text=='slow':await asyncio.sleep(.15)
                return web.json_response({'id':str(payload['seed']),
                    'choices':[{'message':{'content':'译文 '+text},'finish_reason':'length' if text=='truncated' else 'stop'}],
                    'usage':{'completion_tokens':4,'prompt_tokens':10}})
            finally:self.active-=1
        app=web.Application();app.router.add_post('/v1/chat/completions',handler)
        self.runner=web.AppRunner(app);await self.runner.setup()
        self.site=web.TCPSite(self.runner,'127.0.0.1',0);await self.site.start()
        self.url='http://127.0.0.1:'+str(self.site._server.sockets[0].getsockname()[1])

    async def asyncTearDown(self):await self.runner.cleanup()

    def cases(self,texts):return [{'id':str(i),'text':v,'target_lang':'Chinese'} for i,v in enumerate(texts)]

    async def test_reuses_connections_bounds_concurrency_and_preserves_order(self):
        cases=self.cases([str(i) for i in range(8)])
        rows,wall=await translate_many_async(cases,url=self.url,model='hy-mt2',sampling=Sampling(seed=10,repeat_penalty=1.17),concurrency=2)
        self.assertEqual([r['id'] for r in rows],[r['id'] for r in cases])
        self.assertTrue(all(r['ok'] for r in rows));self.assertEqual(self.peak,2)
        self.assertLessEqual(len(self.connections),2)
        self.assertEqual([r['seed'] for r in rows],list(range(10,18)))
        self.assertTrue(all(r['completion_tokens']==4 for r in rows))
        self.assertGreater(wall,0)
        for payload in self.payloads:
            self.assertFalse(payload['stream']);self.assertEqual(payload['repetition_penalty'],1.17)
            self.assertNotIn('repeat_penalty',payload);self.assertNotIn('cache_prompt',payload)

    async def test_failures_and_truncation_are_not_success_and_do_not_drop_rows(self):
        cases=self.cases(['ok','http-error','truncated','malformed'])
        rows,_=await translate_many_async(cases,url=self.url,model='hy-mt2',sampling=Sampling(),concurrency=2)
        self.assertEqual([r['ok'] for r in rows],[True,False,False,False])
        self.assertIn('HTTP 429',rows[1]['error']);self.assertTrue(rows[2]['truncated'])

    async def test_timeout_and_cancellation_release_session(self):
        args=dict(url=self.url,model='hy-mt2',sampling=Sampling(),concurrency=1)
        rows,_=await translate_many_async(self.cases(['slow']),timeout=.04,**args)
        self.assertFalse(rows[0]['ok']);self.assertIn('Timeout',rows[0]['error'])
        self.entered.clear()
        task=asyncio.create_task(translate_many_async(self.cases(['slow']),**args))
        await self.entered.wait();task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task

    async def test_benchmark_uses_same_payload_and_records_full_json_timing(self):
        cases=self.cases(['one','two','three','four'])
        await translate_many_async(cases,url=self.url,model='hy-mt2',sampling=Sampling(),concurrency=2)
        # Independent HTTP connections can arrive out of submission order.
        for i,payload in enumerate(sorted(self.payloads,key=lambda p:p['seed'])):
            self.assertEqual(payload,payload_for(cases[i],i,'vllm',512))
        args=SimpleNamespace(concurrency=2,timeout=5,no_keepalive=False,backend='vllm',max_tokens=512,
                             greedy=False,model='hy-mt2',url=self.url)
        rows,wall,peak=await run_async_pass(cases,args)
        self.assertEqual(peak,2);self.assertGreater(wall,0)
        self.assertTrue(all(r['ok'] and r['ttft_s'] is None and r['completion_tokens']==4 for r in rows))
        self.assertTrue(all(r['client_queue_s']<=r['request_headers_sent_s']<=r['end_to_end_s'] for r in rows))

    async def test_api_key_is_sent_as_header_only(self):
        with patch.dict('os.environ',{'VLLM_API_KEY':'unit-test-key'}):
            rows,_=await translate_many_async(self.cases(['one']),url=self.url,model='hy-mt2',sampling=Sampling(),concurrency=1)
        self.assertEqual(self.authorization,['Bearer unit-test-key'])
        self.assertNotIn('unit-test-key',str(rows)+str(self.payloads))


if __name__=='__main__':unittest.main()
