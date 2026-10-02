"""Fault injection through real HTTP serialization/parser, not an accuracy eval."""
import json,os,tempfile,importlib.resources as resources
from pathlib import Path
from unittest.mock import patch
import jsonschema
import agent_s19 as agent

class Reply:
    def __init__(self,data):self.data=json.dumps({'choices':[{'message':{'content':json.dumps(data)}}]}).encode()
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self):return self.data

def main(units):
    unit=units/'t4-auction-btc-202411-us7'
    task=json.loads((unit/'task.json').read_text(encoding='utf-8'))
    schema=json.loads((resources.files('qfbench2_common')/'schemas/analysis.schema.json').read_text())
    cases=[]
    for mode in ['review_array_kind','null_review_ids','object_unit','batch_exception']:
        def responder(request,timeout):
            payload=json.loads(request.data)
            prompt=json.loads(payload['messages'][1]['content'])
            if 'facts_by_entity' in payload['messages'][0]['content']:
                return Reply({'facts_by_entity':[]})
            predictions=[]
            for row in prompt['rows']:
                is_review=row.get('review_draft') is not None
                quote=row['evidence'][0]['text'][:80]
                prediction={'entity_id':row['entity_id'],'point_forecast':2.5 if is_review else 2.0,
                            'point_unit':prompt['target']['numeric_unit'],
                            'evidence_ids':['E0'],'evidence_quote':quote,
                            'reason':{'evidence_id':'E0','premise_quote':quote,
                                      'mechanism':'The observed historical pattern informs the forecast with uncertainty about future demand.'},
                            'change_reason':'The review considered the historical pattern as an omitted forecast driver.',
                            'correction_kind':'omitted_driver'}
                if is_review and mode=='review_array_kind':prediction['correction_kind']=['arithmetic']
                if is_review and mode=='null_review_ids':prediction['evidence_ids']=None
                if mode=='object_unit':prediction['point_unit']={'unit':'ratio'}
                predictions.append(prediction)
            return Reply({'predictions':predictions})
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'MODEL_ENDPOINT':'http://house.test','MODEL_NAME':'test'}),patch('urllib.request.urlopen',responder):
            if mode=='batch_exception':
                with patch.object(agent,'house_extract',side_effect=RuntimeError('injected local batch fault')):
                    answer=agent.run(unit/'task.json',unit/'corpus',Path(d)/'answer.json')
            else:answer=agent.run(unit/'task.json',unit/'corpus',Path(d)/'answer.json')
        jsonschema.validate(answer,schema)
        assert [r['entity_id'] for r in answer['entity_predictions']]==[r['entity_id'] for r in task['entities']]
        assert all(r['claims'] and r['interval']['lo']<=r['point_forecast']<=r['interval']['hi'] for r in answer['entity_predictions'])
        assert answer['notes']['house_requests']<=25
        cases.append({'case':mode,'valid_rows':len(answer['entity_predictions']),'status':'PASS'})
    Path('test-output').mkdir(exist_ok=True)
    Path('test-output/s19-runtime-faults.json').write_text(json.dumps({'scope':'fault injection only; zero real House requests','cases':cases},indent=2),encoding='utf-8')
    print('PASS: malformed House fields and per-batch exceptions produce complete valid answers')

if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
