"""Counterfactual diagnosis checks with no live model calls or outcome labels."""
import io,json,os,sys,tempfile,copy
from pathlib import Path
from unittest.mock import patch
import agent_s16 as control
import agent_s18 as candidate
from s18_retrieval import supplement

class Reply(io.BytesIO):
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

def main(official):
    units=official/'units';count=0
    for unit in sorted(units.iterdir()):
        if not (unit/'task.json').exists():continue
        requests=[]
        def fake_open(req,timeout):
            payload=json.loads(req.data);requests.append(req.data)
            user=json.loads(payload['messages'][1]['content'])
            if 'facts_by_entity' in payload['messages'][0]['content']:
                data={'facts_by_entity':[]}
            else:
                data={'predictions':[]}
                labels=user['target']['labels']
                for row in user['rows']:
                    quote=row['evidence'][0]['text'][:80]
                    data['predictions'].append({'entity_id':row['entity_id'],'point_forecast':.1,'point_unit':user['target']['numeric_unit'],
                        'label':labels[0] if labels else '', 'interval':{'lo':-.2,'hi':.5},'evidence_ids':['E0'],'evidence_quote':quote})
            body={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(data)}}],'usage':{'prompt_tokens':100,'completion_tokens':50,'total_tokens':150}}
            return Reply(json.dumps(body).encode())
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);outputs=[];payloads=[]
            for module,diagnosis in ((control,False),(candidate,False),(candidate,True)):
                requests.clear()
                env={'MODEL_ENDPOINT':'http://mock.test','MODEL_NAME':'mock','MODEL_TOKEN':'sentinel-secret','AGENTHON_SUPPLEMENT_RETRIEVAL':'0','AGENTHON_DIAGNOSTICS_DIR':str(root/'diag') if diagnosis else ''}
                with patch.dict(os.environ,env),patch('urllib.request.urlopen',fake_open):
                    a=module.run(unit/'task.json',unit/'corpus',root/'answer.json')
                outputs.append(a['entity_predictions']);payloads.append(sorted(requests))
            assert outputs[0]==outputs[1]==outputs[2],unit.name
            assert payloads[0]==payloads[1]==payloads[2],unit.name
            files=list((root/'diag').glob('*.json'));assert len(files)==1
            text=files[0].read_text(encoding='utf-8');assert 'sentinel-secret' not in text and 'http://mock.test' not in text
            events=json.loads(text)['events'];assert any(x['event']=='transport' for x in events)
            assert sum(x['event']=='final_prediction' for x in events)==len(outputs[0])
        count+=1
    # A scoped artificial corpus checks missing-driver coverage and ownership.
    task={'target':{'name':'operating_margin_pct'},'prompt':'Predict operating margin using demand and guidance.'}
    entity={'entity_id':'X','name':'Example'}
    p0=candidate.Passage('A',0,36,'Operating margin was 10 percent.', '2024-01-01')
    text='Demand guidance supports higher operating margin next quarter.'
    p1=candidate.Passage('B',0,len(text),text,'2024-01-01')
    wrong=candidate.Passage('OTHER',0,len(text),text,'2024-01-01')
    class Corpus:
        passages=[p0,p1,wrong]
        def admits_passage(self,p,eid):return p.doc_id!='OTHER'
    corpus=Corpus()
    expanded,metadata=supplement(task,entity,[p0],corpus)
    assert p1 in expanded and wrong not in expanded and expanded[0] is p0
    assert len(metadata['added'])<=2
    same,meta=supplement(task,entity,expanded,corpus)
    assert same==expanded and not meta['added']
    # New deterministic claim rules run on Linux without production NLI.
    if os.name!='nt':
        sys.path.insert(0,str(official.resolve()))
        from baselines.guardrails_example.citation_rail import check_claim_rules
        for unit in sorted(units.iterdir()):
            if not (unit/'task.json').exists():continue
            with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'MODEL_ENDPOINT':'','MODEL_NAME':'','AGENTHON_DIAGNOSTICS_DIR':''}):
                a=candidate.run(unit/'task.json',unit/'corpus',Path(d)/'answer.json')
            assert not check_claim_rules(a,unit,token_counter=lambda t:len(t.encode('utf-8'))),unit.name
    Path('test-output').mkdir(exist_ok=True)
    Path('test-output/s18-diagnostics-verification.json').write_text(json.dumps({'units':count,'identical_control_payloads':True,'identical_control_predictions':True,'diagnostics_noninterference':True,'secret_exclusion':True,'bounded_owned_supplementation':True,'house_mode':'mock','production_nli':False},indent=2)+'\n',encoding='utf-8')
    print(f'PASS: identical S1.6 requests and predictions with retrieval disabled on {count} units; diagnostics noninterference, secret exclusion and bounded owned supplementation')

if __name__=='__main__':main(Path(sys.argv[1]))
