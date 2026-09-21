"""CPU regression coverage for the production vLLM boundary."""
import hashlib,json,sys,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import gpu_config as gpu
import setup_model as models
import desktop_bridge as bridge
from vllm_runtime import llm_config,server_command

class VllmConfigurationTests(unittest.TestCase):
    def plan(self,**kwargs):
        return gpu.memory_plan(free_mib=10000,model_bytes=1363492720,**kwargs)
    def test_equal_token_capacity_includes_quantization_scales(self):
        bf=self.plan(cache_type_k='bfloat16');q=self.plan(cache_type_k='int8_per_token_head')
        self.assertEqual(bf['budget']['kv_token_capacity'],49152)
        self.assertEqual(q['budget']['kv_token_capacity'],49152)
        self.assertEqual(q['budget']['kv_total_mib']/bf['budget']['kv_total_mib'],132/256)
    def test_concurrency_does_not_promise_full_context_slots(self):
        p=self.plan(parallel=256,context=2048)
        self.assertEqual(p['parallel'],256)
        self.assertLess(p['budget']['full_context_sequences'],256)
        self.assertLessEqual(p['budget']['estimated_total_mib'],10000)
    def test_reject_invalid_or_unaffordable_budget(self):
        for args in ({'parallel':257},{'parallel':True},{'context':0},{'kv_gib':50},{'kv_gib':float('nan')},
                     {'cache_type_k':'q4_0'},{'cache_type_k':'q8_0','cache_type_v':'f16'},{'ubatch':1}):
            with self.subTest(args=args),self.assertRaises(ValueError):self.plan(**args)
        with self.assertRaises(ValueError):gpu.memory_plan(free_mib=1000,model_bytes=1363492720)
    def test_hardware_gate_is_explicit(self):
        for capability in ('8.0','8.6','8.9','12.0'):
            gpu.check_gpu({'compute_capability':capability,'driver_version':'596.36'})
        for device in ({'compute_capability':'7.5','driver_version':'596.36'}, {'compute_capability':'6.1','driver_version':'596.36'}, {'compute_capability':'12.0','driver_version':'580.88'}):
            with self.assertRaises(ValueError):gpu.check_gpu(device)
    def test_auto_uses_integer_int4_on_ampere_and_ada(self):
        for capability in ('8.0','8.6','8.9'):
            device={'compute_capability':capability,'driver_version':'596.36'}
            self.assertEqual(gpu.select_profile(device),'compat')
            for profile in ('fast','quality'):
                with self.assertRaises(ValueError):gpu.select_profile(device,profile)
        blackwell={'compute_capability':'12.0','driver_version':'596.36'}
        self.assertEqual(gpu.select_profile(blackwell),'fast')
        self.assertEqual(gpu.select_profile(blackwell,'compat'),'compat')
    def test_small_card_default_budget_can_fit_shared_int8_pool(self):
        plan=gpu.memory_plan(free_mib=8192*.75,model_bytes=1400000000)
        self.assertGreater(plan['budget']['full_context_sequences'],1)
        self.assertLessEqual(plan['budget']['estimated_total_mib'],6144)
    def test_old_desktop_preferences_migrate_without_losing_connection(self):
        s=bridge.validate_settings({'profile':'official','cache':'q8_0','port':19000,'apiKeyEnabled':False})
        self.assertEqual((s['profile'],s['cache'],s['port'],s['apiKeyEnabled']),('auto','int8_per_token_head',19000,False))
    def test_server_command_uses_python_module_and_no_legacy_backend(self):
        p=self.plan();p.update(model='model-folder',kernel='cutlass')
        with patch('vllm_runtime.python_executable',return_value=Path('runtime/vllm/python.exe')):
            cmd=server_command(p)
        self.assertIn('vllm.entrypoints.cli.main',cmd)
        self.assertIn('int8_per_token_head',cmd)
        self.assertIn('--no-enable-prefix-caching',cmd)
        self.assertNotIn('llama-server.exe',' '.join(cmd))
        self.assertEqual(llm_config(p)['max_num_batched_tokens'],2048)
    def test_desktop_budget_credits_owned_memory_and_respects_percent(self):
        device={'index':0,'uuid':'GPU-test','name':'test','driver_version':'596.36','compute_capability':'12.0','total_memory_mib':32768,'free_memory_mib':24000}
        with tempfile.TemporaryDirectory() as d,patch.object(bridge,'detect_gpus',return_value=[device]):
            b=bridge.Bridge({'root':str(ROOT),'data':d,'output':str(Path(d)/'out.json')})
            p=b.plan()
            self.assertEqual(p['parallel'],32)
            self.assertLessEqual(p['budget']['estimated_total_mib'],9830)
            self.assertEqual(p['backend'],'vllm')

    def test_graph_coverage_handles_mixed_prefill_without_raising_concurrency(self):
        for parallel, batch, expected in ((32,2048,256),(128,2048,256),
                                           (256,2048,512),(32,128,128),
                                           (1,2048,256),(256,256,256)):
            with self.subTest(parallel=parallel,batch=batch):
                plan=self.plan(parallel=parallel,ubatch=batch)
                plan.update(model='model-folder',kernel='cutlass')
                cfg=llm_config(plan)
                self.assertEqual(cfg['compilation_config']['max_cudagraph_capture_size'],expected)
                self.assertEqual(cfg['max_num_seqs'],parallel)
                self.assertEqual(cfg['max_num_batched_tokens'],batch)
                self.assertEqual(cfg['kv_cache_memory_bytes'],int(plan['budget']['kv_total_mib']*1024**2))
                with patch('vllm_runtime.python_executable',return_value=Path('runtime/vllm/python.exe')):
                    command=server_command(plan)
                self.assertEqual(json.loads(command[command.index('--compilation-config')+1]),cfg['compilation_config'])

class CheckpointTests(unittest.TestCase):
    def fixture(self,root):
        data=b'checkpoint fixture';item={'size_bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        manifest={'schema_version':2,'backend':'vllm','checkpoint_dir':'model','checkpoint_files':{'config.json':item},
            'files':{'fast':{'filename':'model.zip',**item}},'checkpoint_size_bytes':len(data)}
        return data,manifest
    def test_zip_extracts_atomically_and_reuses_verified_files(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data,m=self.fixture(root);bundle=root/'model.zip'
            with zipfile.ZipFile(bundle,'w') as z:z.writestr('config.json',data)
            target=models.extract_checkpoint(bundle,root/'model',m)
            self.assertEqual((target/'config.json').read_bytes(),data)
            self.assertEqual(models.extract_checkpoint(bundle,target,m),target)
    def test_zip_traversal_extra_files_and_corruption_never_publish(self):
        for names in ({'../outside':b'bad'},{'config.json':b'bad'}, {'config.json':b'checkpoint fixture','extra':b'bad'}):
            with self.subTest(names=names),tempfile.TemporaryDirectory() as d:
                root=Path(d);data,m=self.fixture(root);bundle=root/'model.zip'
                with zipfile.ZipFile(bundle,'w') as z:
                    for name,value in names.items():z.writestr(name,value)
                with self.assertRaises(ValueError):models.extract_checkpoint(bundle,root/'model',m)
                self.assertFalse((root/'model').exists())
    def test_manifest_rejects_unsafe_names_and_unpinned_files(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data,m=self.fixture(root);path=root/'manifest.json'
            path.write_text(json.dumps(m));models.load_manifest(path)
            for change in ('../model.zip','model.zip:stream'):
                m['files']['fast']['filename']=change;path.write_text(json.dumps(m))
                with self.assertRaises(ValueError):models.load_manifest(path)
    def test_compat_checkpoint_has_independent_pinned_contents(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,m=self.fixture(root);path=root/'manifest.json'
            m['files']['compat']={**m['files']['fast'],'filename':'compat.zip'}
            m['checkpoints']={'compat':{k:m[k] for k in ('checkpoint_dir','checkpoint_files','checkpoint_size_bytes')}}
            m['checkpoints']['compat']['checkpoint_dir']='compat'
            path.write_text(json.dumps(m))
            loaded=models.load_manifest(path)
            self.assertEqual(models.checkpoint_manifest(loaded,'compat')['checkpoint_dir'],'compat')
            self.assertEqual(models.checkpoint_manifest(loaded,'fast')['checkpoint_dir'],'model')
            m['checkpoints']['compat']['checkpoint_dir']='../escape'
            path.write_text(json.dumps(m))
            with self.assertRaises(ValueError):models.load_manifest(path)

class PortableEnvironmentTests(unittest.TestCase):
    def test_host_toolchains_cannot_leak_into_inference(self):
        import os
        from vllm_runtime import environment
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);runtime=root/'runtime/vllm'
            for name in ('python.exe','Lib/site-packages/triton/runtime/tcc/tcc.exe',
                         'Lib/site-packages/triton/backends/nvidia/bin/ptxas.exe',
                         'Lib/site-packages/triton/backends/nvidia/bin/cudart64_13.dll',
                         'Lib/site-packages/flashinfer/data/aot/sampling/sampling.dll'):
                p=runtime/name;p.parent.mkdir(parents=True,exist_ok=True);p.touch()
            with patch.dict(os.environ,{'PATH':'host-compiler','CC':'cl.exe','CUDA_PATH':'host-cuda','INCLUDE':'host-sdk','PYTHONPATH':'host-python','FLASHINFER_DISABLE_JIT':'0','CUDNN_LIB_CONFIG':'FULL'}):
                env=environment(root)
            self.assertEqual(env['CUDNN_LIB_CONFIG'],'GRAPH_JIT_ONLY')
            self.assertNotIn('host-',env['PATH'])
            self.assertNotIn('INCLUDE',env)
            self.assertNotIn('PYTHONPATH',env)
            self.assertEqual(env['FLASHINFER_DISABLE_JIT'],'1')
            self.assertTrue(env['CC'].endswith('tcc.exe'))

class DesktopRuntimePathTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'win32', 'Windows canonical paths')
    def test_extended_rust_path_becomes_native_toolchain_path(self):
        from desktop_bridge import runtime_path
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(runtime_path('\\\\?\\'+str(root)), root)


if __name__=='__main__':unittest.main()
