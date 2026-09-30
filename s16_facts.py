"""Target-driven sparse retrieval and verifiable facts; official corpus only."""
from __future__ import annotations
import math
import re
from collections import Counter

WORD=re.compile(r'[a-z]+|\d+(?:\.\d+)?')
NUMBER=re.compile(r'(?<!\w)[-+]?\$?\(?\d+(?:,\d{3})*(?:\.\d+)?\)?%?')
STOP=set('the and of to a in for is it on as by with from that this an or be at are was were will its each per only using predict forecast value point interval entity task'.split())


def terms(text):
    return [w for w in WORD.findall(str(text).lower()) if w not in STOP]


def queries(task,entity):
    target=str(task.get('target',{}).get('name','')).replace('_',' ')
    prompt=str(task.get('prompt',''))
    # Derive drivers from instructions rather than dispatching on task IDs.
    spans=re.findall(r'(?:reason from|drivers?\s*[—:\-]|sensitivity to|using|based on|consider)\s*(.{12,700}?)(?:\.\s|$)',prompt,re.I)
    result=[target+' '+str(entity.get('name',''))]
    result.extend(spans[:3])
    result.extend(re.split(r'[;,]|\s+and\s+',spans[0])[:6] if spans else [])
    result.append(prompt[:1000])
    return [Counter(terms(s)) for s in result if s]


def index(corpus):
    if hasattr(corpus,'_fact_index'):
        return corpus._fact_index
    counts=[Counter(terms(p.text)) for p in corpus.passages]
    df=Counter(w for c in counts for w in c)
    avg=sum(sum(c.values()) for c in counts)/max(1,len(counts))
    corpus._fact_index=(counts,df,max(avg,1.0))
    return corpus._fact_index


def select(task,entity,corpus,limit=6):
    counts,df,avg=index(corpus)
    n=len(counts)
    query=queries(task,entity)
    candidates=[]
    eid=str(entity['entity_id'])
    for i,p in enumerate(corpus.passages):
        if p.doc_id=='task' or not corpus.admits_passage(p,eid):
            continue
        c=counts[i]; length=sum(c.values())
        scores=[]
        for q in query:
            score=0.0
            for word in q:
                tf=c[word]
                if tf:
                    idf=math.log(1+(n-df[word]+.5)/(df[word]+.5))
                    score+=idf*tf*2.2/(tf+1.2*(.25+.75*length/avg))
            scores.append(score)
        candidates.append((i,p,scores))
    # Round-robin query coverage with overlap suppression: all slots need not
    # repeat the same target noun in the same filing section.
    chosen=[]; used=set()
    for qindex in range(len(query)):
        ranking=sorted(candidates,key=lambda item:item[2][qindex],reverse=True)
        for i,p,scores in ranking:
            if i in used or scores[qindex]<=0:
                continue
            if any(p.doc_id==old.doc_id and max(p.start,old.start)<min(p.end,old.end) for old in chosen):
                continue
            chosen.append(p); used.add(i); break
        if len(chosen)>=limit:
            break
    if not chosen and candidates:
        chosen=[max(candidates,key=lambda item:max(item[2]))[1]]
    own=corpus.task_rows.get(eid)
    if own:
        chosen=chosen[:limit-1]+[own]
    return chosen


def local_facts(passages,max_facts=10):
    """Extract source sentences, not guessed values from ambiguous table cells."""
    facts=[]
    for i,p in enumerate(passages):
        candidates=re.split(r'(?<=[.!?])\s+|\n',p.text)
        for sentence in candidates:
            sentence=sentence.strip()
            if not 20<=len(sentence)<=380 or not NUMBER.search(sentence):
                continue
            if not re.search(r'\d',sentence):
                continue
            years=re.findall(r'\b(?:19|20)\d{2}\b',sentence)
            facts.append({'evidence_id':f'E{i}','quote':sentence,
                          'period_context':years,'status':'source_observation_not_target_outcome'})
            if len(facts)>=max_facts:
                return facts
    return facts


def validate_facts(raw,passages,entity_id):
    """Keep only exact observed quotes; reject invented numeric paraphrases."""
    kept=[]
    for fact in raw if isinstance(raw,list) else []:
        if not isinstance(fact,dict):
            continue
        eid=fact.get('evidence_id',''); quote=fact.get('quote','')
        if not isinstance(eid,str) or not re.fullmatch(r'E\d+',eid) or not isinstance(quote,str):
            continue
        index=int(eid[1:])
        if index>=len(passages) or not 12<=len(quote)<=380 or quote not in passages[index].text:
            continue
        record={'evidence_id':eid,'quote':quote,'entity_id':entity_id,
                'doc_id':passages[index].doc_id,'doc_date':passages[index].date,
                'span_start':passages[index].start+passages[index].text.index(quote),
                'span_end':passages[index].start+passages[index].text.index(quote)+len(quote)}
        # Model-provided period/basis interpretation is explicitly provisional;
        # it never becomes an asserted claim or automatic numeric training label.
        for key in ('metric','period','comparison_period','unit','basis','driver_role'):
            if isinstance(fact.get(key),str):
                record[key]=fact[key][:140]
        kept.append(record)
        if len(kept)>=5:
            break
    return kept
