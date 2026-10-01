"""A bounded second retrieval pass for uncovered task-derived information needs."""
import re,math
from s18_facts import queries,terms,index

# Metadata/grammar terms do not identify a missing financial driver.
GENERIC=set('company companies market financial finance predict prediction forecast target next future value change growth rate year quarter month day days numeric percent percentage basis points bps pct recent latest prior historical entity entities table features information evidence determine estimate expected direction return returns result results'.split())
def needs(task,entity):
    qs=queries(task,entity);out=[]
    # S1.6 queries already derive driver clauses from the prompt. Do not use IDs.
    for q in qs:
        key=set(q)-GENERIC-set(terms(entity.get('name','')))-set(terms(entity.get('entity_id','')))
        if key and key not in out:out.append(key)
    return out[:10]
def support(p,need):
    present=set(terms(p.text))
    hits=len(present&need)
    return hits>=min(2,len(need)) and hits/max(1,len(need))>=.25

def supplement(task,entity,passages,corpus,max_extra=2):
    requirements=needs(task,entity);source=[p for p in passages if p.doc_id!='task']
    missing=[q for q in requirements if not any(support(p,q) for p in source)]
    if not missing:return list(passages),{'missing_before':[],'added':[],'missing_after':[]}
    counts,df,avg=index(corpus);n=len(counts);eid=str(entity['entity_id']);added=[]
    for need in missing:
        if len(added)>=max_extra:break
        if any(support(p,need) for p in added):continue
        candidates=[]
        for i,p in enumerate(corpus.passages):
            if p.doc_id=='task' or not corpus.admits_passage(p,eid) or not support(p,need):continue
            if any(p.doc_id==x.doc_id and max(p.start,x.start)<min(p.end,x.end) for x in passages+added):continue
            c=counts[i];length=sum(c.values());score=0
            for w in need:
                tf=c[w]
                if tf:score+=math.log(1+(n-df[w]+.5)/(df[w]+.5))*tf*2.2/(tf+1.2*(.25+.75*length/avg))
            # Require target relevance OR a narrowly specified driver clause.
            target=set(terms(task.get('target',{}).get('name','')))-GENERIC
            if target and not (set(terms(p.text))&target) and len(need)>8:continue
            candidates.append((score,-i,p))
        if candidates:added.append(max(candidates,key=lambda x:x[:2])[2])
    # Append so existing E-index citations and statement-first context stay stable.
    after=[q for q in missing if not any(support(p,q) for p in added)]
    metadata={'missing_before':[sorted(q) for q in missing],'added':[{'doc_id':p.doc_id,'start':p.start,'end':p.end} for p in added],
              'missing_after':[sorted(q) for q in after]}
    return list(passages)+added,metadata
