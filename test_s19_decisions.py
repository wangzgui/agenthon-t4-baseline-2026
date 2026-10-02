"""Named regression tests for the audited failure mechanisms; no score mocks."""
import json,os,tempfile
from pathlib import Path
from unittest.mock import patch
from agent_s19 import Passage,review_or_medoid
from s19_decisions import merge_facts
from s19_reasoning import build

def main():
    quote='Revenue increased by 12 percent in the quarter.'
    p=Passage('D',0,len(quote),quote,'2024-01-01')
    original=[{'evidence_id':'E0','quote':quote,'status':'original'}]
    added=[{'evidence_id':'E0','quote':quote},{'evidence_id':'E0','quote':'12 percent in the quarter'}]
    facts=merge_facts(original,added)
    assert facts[0]==original[0] and len(facts)==2 and original[0]['status']=='original'
    t={'target':{'type':'regression','name':'revenue_growth_pct'}}
    batch=[(0,{'entity_id':'X'},[p])]
    draft={'entity_id':'X','point_forecast':1.0}
    review=dict(draft,point_forecast=2.0,evidence_ids=['E0'],evidence_quote=quote)
    assert review_or_medoid(t,batch,[[draft],[review]])[0]['point_forecast']==1.0
    review.update(correction_kind='omitted_driver',change_reason='The draft omitted stronger revenue demand, which raises the projected growth.',reason={'evidence_id':'E0','premise_quote':quote,'mechanism':'Reported demand expansion provides support for future revenue growth, subject to persistence uncertainty.'})
    assert review_or_medoid(t,batch,[[draft],[review]])[0]['point_forecast']==2.0
    for key,value in [('premise_quote','Revenue increased by 99 percent.'),('evidence_id','E999')]:
        broken=dict(review,reason=dict(review['reason'],**{key:value}))
        assert review_or_medoid(t,batch,[[draft],[broken]])[0]['point_forecast']==1.0
    # Whole-roster coherence: partial corrections cannot splice ranking scales.
    rank=dict(t,target={'type':'ranking','name':'revenue_growth_pct'})
    batch2=batch+[(1,{'entity_id':'Y'},[p])]
    first=[draft,dict(draft,entity_id='Y',point_forecast=3.0)]
    partial=[review,dict(review,entity_id='Y',correction_kind='invented')]
    assert review_or_medoid(rank,batch2,[first,partial])==first
    class Corpus:
        def admits(self,doc,eid):return doc=='D' and eid=='X'
    answer={'entity_predictions':[{'entity_id':'X','point_forecast':2.5,'interval':{'lo':0,'hi':4}}]}
    reasons=build(t,batch,{'X':review},answer,Corpus())
    assert len(reasons)==1 and '2.5' in reasons[0]['answer_implication']
    assert reasons[0]['citations'][0]['span_end']==len(quote)
    reversed_answer={'entity_predictions':[dict(answer['entity_predictions'][0],point_forecast=-2.5)]}
    assert not build(t,batch,{'X':review},reversed_answer,Corpus())
    print('PASS: preserve original facts; quote-only correction rejected; explicit correction; fabricated premises; coherent ranking; normalized Final reasoning')

if __name__=='__main__':main()
