"""Opt-in local diagnostics; never changes model inputs or predictions."""
import os,json,time,hashlib,math
from pathlib import Path
from threading import Lock

def clean(v):
    if isinstance(v,float) and not math.isfinite(v):return None
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    return v if v is None or isinstance(v,(str,int,float,bool)) else type(v).__name__
class Trace:
    def __init__(self,task):
        self.enabled=bool(os.environ.get('AGENTHON_DIAGNOSTICS_DIR'));self.task_id=str(task.get('task_id','task'))
        self.events=[];self.lock=Lock();self.started=time.monotonic()
    def add(self,kind,data):
        if not self.enabled:return
        try:
            with self.lock:
                if len(self.events)<2000:self.events.append({'event':kind,'elapsed_sec':round(time.monotonic()-self.started,4),'data':clean(data)})
        except Exception:pass
    def save(self):
        if not self.enabled:return
        try:
            folder=Path(os.environ['AGENTHON_DIAGNOSTICS_DIR']);folder.mkdir(parents=True,exist_ok=True)
            name=hashlib.sha256(self.task_id.encode()).hexdigest()[:16]+'.json'
            (folder/name).write_text(json.dumps({'task_id':self.task_id,'version':'s1.9','events':self.events},ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        except Exception as e:print('Local diagnostics unavailable:',type(e).__name__,flush=True)

def emit(task,kind,data):
    trace=task.get('_trace')
    if trace:trace.add(kind,data)
def transport(task,stage,payload,body,start,expected):
    choices=body.get('choices',[]) if isinstance(body,dict) else []
    first=choices[0] if choices and isinstance(choices[0],dict) else {}
    emit(task,'transport',{'stage':stage,'request_sha256':hashlib.sha256(json.dumps(payload).encode('utf-8')).hexdigest(),
        'expected_rows':expected,'finish_reason':first.get('finish_reason'),'usage':{k:v for k,v in body.get('usage',{}).items() if k in ('prompt_tokens','completion_tokens','total_tokens') and isinstance(v,int)} if isinstance(body.get('usage'),dict) else None,
        'latency_sec':round(time.monotonic()-start,4)})
def forecast(raw):
    if not isinstance(raw,dict):return {}
    return {k:raw[k] for k in ('entity_id','point_forecast','point_unit','projected_level','label','interval','evidence_ids','_unit_conflict','_numeric_disagreement','_house_disagreement') if k in raw}
