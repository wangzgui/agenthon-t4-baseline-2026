"""Targeted V1.7 decision and official deterministic/Final reason checks."""
import json,sys,tempfile,os,copy
import jsonschema
from importlib import resources
from pathlib import Path
from unittest.mock import patch
from s17_decision import enrich,arithmetic,assess,comparisons
from s17_facts import validate_facts
from agent_s17 import Passage,Corpus,run,canonical_prediction,review_or_medoid
from s17_reasoning import build

def main(official):
    sys.path.insert(0,str(official.resolve()))
    from baselines.guardrails_example.citation_rail import check_claim_rules,check_submitted_reasons,load_corpus
    quote='Diluted EPS was $2.00 in Q1 2024; diluted EPS was $1.00 in Q1 2023.'
    p=Passage('D',0,len(quote),quote,'2024-04-01')
    raw=[{'evidence_id':'E0','quote':quote,'value_text':'$2.00','metric':'Diluted EPS','period':'Q1 2024'},
         {'evidence_id':'E0','quote':quote+'x','value_text':'$1.00','metric':'Diluted EPS','period':'Q1 2023'}]
    # Use distinct source slices so quantities cannot overwrite one another by quote key.
    raw[0]['quote']='Diluted EPS was $2.00 in Q1 2024'
    raw[1]['quote']='diluted EPS was $1.00 in Q1 2023'
    facts=enrich(validate_facts(raw,[p],'X'),raw)
    result=arithmetic({'operation':'growth_pct','left_fact':'F0','right_fact':'F1'},facts)
    assert result['status']=='verified' and result['value']==100 and result['unit']=='pct'
    assert comparisons(facts)
    repeated=copy.deepcopy(raw)
    for f in repeated:f['quote']=quote
    same=enrich(validate_facts(repeated,[p],'X'),repeated)
    assert [f['quantity']['value'] for f in same]==[2,1]
    forged=copy.deepcopy(raw);forged[0]['value_text']='$200.00'
    assert 'quantity' not in enrich(validate_facts(forged,[p],'X'),forged)[0]
    mismatch=copy.deepcopy(facts);mismatch[1]['quantity']['basis']='adjusted'
    assert arithmetic({'operation':'growth_pct','left_fact':'F0','right_fact':'F1'},mismatch)['status']=='rejected'
    unknown=copy.deepcopy(facts);unknown[1]['quantity']['period']='unknown'
    assert arithmetic({'operation':'growth_pct','left_fact':'F0','right_fact':'F1'},unknown)['status']=='rejected'
    task={'target':{'type':'regression','name':'change_bps'}}
    entity={'entity_id':'X','_source_facts':facts}
    invalid=assess(task,entity,{'point_forecast':10,'forecast_components':{'baseline':4,'event_adjustment':7,'unit':'bps'}})
    assert invalid['_decision_conflict']
    observed=assess(task,entity,{'point_forecast':100,'calculation':{'operation':'growth_pct','left_fact':'F0','right_fact':'F1','result_is_forecast':True}})
    assert observed['_decision_conflict']
    converted=canonical_prediction(task,entity,{'point_forecast':.2,'point_unit':'pct','interval':{'lo':.1,'hi':.3}})
    assert converted['interval']=={'lo':10,'hi':30} and canonical_prediction(task,entity,converted)==converted
    unit=official/'units/t4-EXAMPLE-eps-beat'
    task=json.loads((unit/'task.json').read_text(encoding='utf-8'));corpus=Corpus(unit/'corpus',task['cutoff_date'],task)
    def fake_batch(t,batch,replicate=0,numeric_prior=None):
        output=[]
        for _,e,ps in batch:
            idx=next((i for i,p in enumerate(ps) if p.doc_id!='task'),None)
            r={'entity_id':e['entity_id'],'point_forecast':1.57,'point_unit':'usd','label':t['target']['labels'][0],'evidence_ids':['E0']}
            if idx is not None:
                q=ps[idx].text[:100]
                r['reason']={'evidence_id':f'E{idx}','premise_quote':q,'mechanism':'The reported operating trend informs the near-term earnings outlook; uncertainty limits the weight assigned to this driver.','answer_implication':'intentionally wrong number 999'}
            output.append(r)
        return output
    with tempfile.TemporaryDirectory() as d,patch('agent_s17.house_url',return_value='http://house.test'),patch.dict(os.environ,{'MODEL_NAME':'test'}),patch('agent_s17.house_batch',side_effect=fake_batch),patch('agent_s17.house_extract',return_value=[]):
        answer=run(unit/'task.json',unit/'corpus',Path(d)/'answer.json')
    schema=json.loads((resources.files('qfbench2_common')/'schemas/analysis.schema.json').read_text())
    jsonschema.validate(answer,schema)
    assert answer.get('submitted_reasons')
    assert all('999' not in r['answer_implication'] for r in answer['submitted_reasons'])
    findings=check_submitted_reasons(answer,load_corpus(unit/'corpus'),task['cutoff_date'])
    assert not findings,findings
    # New rules checked with the actual official scorer. Byte count is a conservative
    # claim-token upper bound for this ASCII fixture; this is not a production NLI run.
    count=0
    if os.name=='nt':
        print('PASS: source quantities, arithmetic rejection, decomposition, canonical intervals and official Final reason checker; POSIX-only scorer claim checks deferred to Linux CI')
        return
    for u in sorted((official/'units').iterdir()):
        if not (u/'task.json').exists():continue
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'MODEL_ENDPOINT':'','MODEL_NAME':''}):
            answer=run(u/'task.json',u/'corpus',Path(d)/'answer.json')
        findings=check_claim_rules(answer,u,token_counter=lambda t:len(t.encode('utf-8')))
        assert not findings,(u.name,findings)
        count+=1
    print(f'PASS: anchored quantities, arithmetic rejection, decomposition, unit/interval invariance, Final reason checker and scorer 5.2.2 deterministic claims on {count} units')

if __name__=='__main__':main(Path(sys.argv[1]))
