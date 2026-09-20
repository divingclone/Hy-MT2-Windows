"""Smoke-test a relocated native vLLM package using its own Python and model.

Stop other GPU services first. No deletion or model downloads. Writes a fresh
report outside the package and stops only the owned validation service.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--port', type=int, default=18084)
    args = parser.parse_args()
    root = args.directory.resolve()
    if (root/'payload').is_dir():
        root /= 'payload'
    report_path = args.report.resolve()
    logs = report_path.with_suffix('')
    if report_path.exists() or logs.exists() or report_path.is_relative_to(root):
        raise ValueError('Choose a fresh report path outside the package')
    logs.mkdir(parents=True)
    python = root/'runtime/vllm/python.exe'
    report = {'backend':'vllm','root':str(root),'ok':False,'commands':[]}
    env = {k:v for k,v in os.environ.items() if k not in ('PYTHONHOME','PYTHONPATH','VLLM_API_KEY')}
    env.update(PYTHONUTF8='1',PYTHONNOUSERSITE='1')
    label = 'portable-vllm-smoke'
    pid_path = root/'results'/f'{label}.pid'
    if pid_path.exists():
        raise ValueError('Validation label already exists; stop that service first')

    def run(name, command):
        started = time.monotonic()
        with (logs/(name+'.launcher.log')).open('xb') as stream:
            process = subprocess.Popen(command,cwd=root,env=env,stdout=stream,stderr=stream,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                code = process.wait(timeout=900)
            except BaseException:
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
                process.wait(timeout=30)
                raise
        report['commands'].append({'name':name,'exit_code':code,'wall_s':time.monotonic()-started})
        if code:
            raise RuntimeError(f'{name} failed: {logs/(name+".launcher.log")}')

    prefix = [str(python),'-E','-s','-X','utf8']
    try:
        run('relocation',prefix+['-c',
            'import sys,pathlib,torch,vllm,hymt_vllm_model; p=pathlib.Path(sys.executable).parent; '
            'assert pathlib.Path(sys.prefix)==p; '
            'assert all(pathlib.Path(m.__file__).is_relative_to(p) for m in (torch,vllm,hymt_vllm_model)); '
            'print(sys.executable,torch.__version__,vllm.__version__)'])
        run('model',prefix+[str(root/'scripts/setup_model.py')])
        output = logs/'batch.jsonl'
        run('batch',prefix+[str(root/'scripts/translate_batch.py'),str(root/'examples/input.jsonl'),str(output),
                            '--parallel','32','--greedy','--max-tokens','128'])
        rows = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
        assert rows and all(row.get('ok') and row.get('translation') and not row.get('truncated') for row in rows)
        report['batch_requests'] = len(rows)
        run('server',prefix+[str(root/'scripts/serve.py'),'--background','--label',label,'--port',str(args.port)])
        config = json.loads((root/'results'/f'{label}.config.json').read_text(encoding='utf-8'))
        assert Path(config['executable']).resolve() == python
        assert Path(config['model']).resolve().is_relative_to(root/'models')
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        body = {'model':'hy-mt2','messages':[{'role':'user','content':'Translate into Chinese: Hello world.'}],
                'temperature':0,'max_tokens':64}
        request = urllib.request.Request(f'http://127.0.0.1:{args.port}/v1/chat/completions',
            data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with opener.open(request,timeout=60) as response:
            result = json.load(response)
        assert result['choices'][0]['message']['content'].strip()
        report['api'] = result
        report['ok'] = True
    except Exception as error:
        report['error'] = str(error)
    finally:
        if pid_path.exists():
            try:
                run('stop',['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',
                    str(root/'scripts/stop-server.ps1'),'-Label',label])
                report['owned_service_stopped'] = not pid_path.exists()
            except Exception as error:
                report['ok'] = False
                report['cleanup_error'] = str(error)
        report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'ok':report['ok'],'report':str(report_path)},ensure_ascii=False))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
