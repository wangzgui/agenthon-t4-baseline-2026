"""Contracts, source ownership, exact quotes, and causal numerical invariants."""
import importlib.resources as resources
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import jsonschema
from agent_s18 import Corpus, Passage, canonical_prediction, house_medoid, normalize, retrieve, run, review_or_medoid
from s18_calibration import interval
from s18_fundamentals import extract_facts
from s18_numerics import DatedSeries, fit_prior, residuals_for_entity
import datetime as dt


def check_invariants():
    task={'target':{'type':'regression','name':'yield_change_bps'},'interval_level':.9,
          'prompt':'Predict change in yield in bp','entities':[]}
    entity={'entity_id':'X','start_yield_pct':4.0}
    raw={'entity_id':'X','point_forecast':.1,'projected_level':4.1}
    canonical=canonical_prediction(task,entity,raw)
    assert abs(canonical['point_forecast']-10)<1e-8
    assert canonical['_unit_conflict']
    # Converted-only and direct-only representations must fuse identically.
    with tempfile.TemporaryDirectory() as directory:
        corpus=Corpus(Path(directory),'2024-01-01',dict(task,entities=[entity]))
        a=normalize(task,entity,dict(raw,point_forecast=10.0),[corpus.task_rows['X']],corpus,2.0,.6)
        b=normalize(task,entity,{'point_forecast':10.0},[corpus.task_rows['X']],corpus,2.0,.6)
        assert abs(a['point_forecast']-b['point_forecast'])<1e-8
        assert abs(a['point_forecast']-5.2)<1e-8
        only=normalize(task,entity,{'projected_level':4.1},[corpus.task_rows['X']],corpus,2.0,.6)
        assert abs(only['point_forecast']-5.2)<1e-8
        conflicting=normalize(task,entity,raw,[corpus.task_rows['X']],corpus,2.0,.6)
        assert abs(conflicting['point_forecast']-4.8)<1e-8
    band,source=interval(task,0,{'lo':-999,'hi':999},[-2,-1,0,1,2]*10,20,0,0,1)
    assert band['hi']-band['lo']<20 and 'residual' in source
    taskp=dict(task,target={'type':'regression','name':'default_probability'})
    band,_=interval(taskp,.99)
    assert 0<=band['lo']<=.99<=band['hi']<=1
    good='Three months ended June 30\n2024 2023\nDiluted earnings per share $2.50 $2.00'
    assert len(extract_facts(good))==1
    assert extract_facts(good)[0].period.endswith(':2024')
    assert not extract_facts('Diluted earnings per share $2.50 $2.00')
    assert not extract_facts(good.replace('Three months','Six months'))
    assert not extract_facts('Three months ended June 30 March 31\n2024 2024\nDiluted earnings per share $2.50 $2.00')
    mixed='In second quarter 2024, net income was $4.9 billion and diluted earnings per common share of $1.33, compared with $4.9 billion of net income and diluted EPS of $1.25 in the same period a year ago.'
    fact=extract_facts(mixed)[0]
    assert (fact.current,fact.previous)==(1.33,1.25)
    # Calibration labels cannot change model selection or coefficients.
    dates=tuple(dt.date(2020,1,1)+dt.timedelta(days=i*7) for i in range(100))
    series=[DatedSeries('D'+str(j),dates,tuple(i*.1+j for i in range(100)),'X'+str(j)) for j in range(4)]
    t=dict(task,target={'type':'regression','name':'value_change_pct'},cutoff_date='2021-12-01',resolution_date='2021-12-08')
    fitted=fit_prior(t,series)
    boundary=dt.date.fromisoformat(fitted.calibration_start)
    altered=[DatedSeries(s.doc_id,s.dates,tuple(v+(100 if day>=boundary else 0) for day,v in zip(s.dates,s.values)),s.heading) for s in series]
    changed=fit_prior(t,altered)
    assert (fitted.model,fitted.coefficients)==(changed.model,changed.coefficients)
    assert fitted.calibration_origins>0 and fitted.pooled_residuals!=changed.pooled_residuals


def main(units):
    check_invariants()
    from dataclasses import replace
    from s18_calibration import fusion_weight
    from s18_facts import validate_facts
    from s18_numerics import FittedPrior
    fit=FittedPrior('ridge',(0,0),1,-.5,-1,100,validation_origins=6)
    assert fusion_weight(fit)==fusion_weight(replace(fit,validation_score=-50,validation_baseline=-100))
    sys.path.insert(0,str(units.parent))
    from qfbench2_track_analysis.numeric import verbatim_quote, claim_number_status
    p=Passage('D',0,50,'Revenue increased by 12 percent in the quarter.','2024-01-01')
    assert len(validate_facts([{'evidence_id':'E0','quote':'Revenue increased by 12 percent'}],[p],'X'))==1
    assert not validate_facts([{'evidence_id':'E0','quote':'Revenue increased by 99 percent'}],[p],'X')
    assert not validate_facts([{'evidence_id':'E999','quote':'Revenue increased by 12 percent'}],[p],'X')
    t={'target':{'type':'regression','name':'change_bps'}}
    canonical=canonical_prediction(t,{'entity_id':'X'}, {'point_forecast':.2,'point_unit':'pct'})
    assert canonical['point_forecast']==20.0
    assert canonical_prediction(t,{'entity_id':'X'},canonical)['point_forecast']==20.0
    batch=[(0,{'entity_id':'X'},[p])]
    draft={'entity_id':'X','point_forecast':1.0}
    good=dict(draft,point_forecast=2.0,evidence_ids=['E0'],evidence_quote='Revenue increased by 12 percent')
    bad=dict(good,evidence_quote='Revenue increased by 99 percent')
    assert review_or_medoid(t,batch,[[draft],[good]])[0]['point_forecast']==2.0
    assert review_or_medoid(t,batch,[[draft],[bad]])[0]['point_forecast']==1.0
    schema=json.loads((resources.files('qfbench2_common')/'schemas'/'analysis.schema.json').read_text())
    count=rows=claims=0
    results=[]
    for unit in sorted(units.iterdir()):
        if not (unit/'task.json').is_file():
            continue
        task=json.loads((unit/'task.json').read_text(encoding='utf-8'))
        corpus=Corpus(unit/'corpus',task['cutoff_date'],task)
        with tempfile.TemporaryDirectory() as directory:
            a=run(unit/'task.json',unit/'corpus',Path(directory)/'answer.json')
        jsonschema.validate(a,schema)
        assert [r['entity_id'] for r in a['entity_predictions']]==[r['entity_id'] for r in task['entities']]
        for row in a['entity_predictions']:
            assert row['interval']['lo']<=row['point_forecast']<=row['interval']['hi']
            for claim in row['claims']:
                assert corpus.admits(claim['doc_id'],row['entity_id'])
                text=corpus.texts[claim['doc_id']]
                start,end=claim['span_start'],claim['span_end']
                assert 0<=start<end<=len(text)
                span=text[start:end]
                assert verbatim_quote(claim['claim'],[span])
                if claim['doc_id']=='task':
                    own=corpus.task_rows[row['entity_id']]
                    assert own.start<=start<end<=own.end
                claims+=1
            rows+=1
        # Exercise ensemble plus House-only representations and missing rows.
        def fake_house(_,batch,replicate=0,numeric_prior=None):
            return [{'entity_id':e['entity_id'],'point_forecast':.1+replicate*.01,
                     'label':task.get('target',{}).get('labels',[''])[0],
                     'evidence_ids':['E0'],'interval':{'lo':-.3,'hi':.5}} for _,e,_ in batch]
        with tempfile.TemporaryDirectory() as directory,patch('agent_s18.house_url',return_value='http://house.test'),patch.dict('os.environ',{'MODEL_NAME':'test'}),patch('agent_s18.house_batch',side_effect=fake_house),patch('agent_s18.house_extract',return_value=[]):
            mocked=run(unit/'task.json',unit/'corpus',Path(directory)/'answer.json')
            jsonschema.validate(mocked,schema)
            assert mocked['notes']['house_requests']<=25
        count+=1
        results.append({'unit':unit.name,'rows':len(a['entity_predictions']),'notes':a['notes']})
        print('OK',unit.name)
    summary={'units':count,'rows':rows,'claims':claims,'ownership_and_exact_quotes':'pass','numerical_invariants':'pass','house_mock':'pass','actual_house_score':'unavailable','units_detail':results}
    Path('test-output').mkdir(exist_ok=True)
    Path('test-output/s18-verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('PASS',count,'units',rows,'rows',claims,'owned exact citations')


if __name__=='__main__':
    main(Path(sys.argv[1]))
