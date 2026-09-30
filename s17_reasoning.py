"""Build up to three bounded Final reasons against the actual normalized answers."""
import json,re,unicodedata
DENY=('leaderboard','canary','/home/','units/','reference/','outcome.json','team_id','team name','participant_id','participant name','submission_id','other submission', '://')
def safe_text(text,limit):
    if not isinstance(text,str) or not 12<=len(text)<=limit:return False
    t=unicodedata.normalize('NFKC',text).casefold()
    return not any(x in t for x in DENY) and not re.search(r'(?i)ignore.{0,20}instructions|award.{0,20}score|give.{0,20}full credit',t)

def build(task,prepared,raw_by_id,answer,corpus):
    rows={str(r['entity_id']):r for r in answer['entity_predictions']};candidates=[]
    for _,entity,passages in prepared:
        eid=str(entity['entity_id']);raw=raw_by_id.get(eid,{})
        reason=raw.get('reason')
        if not isinstance(reason,dict):continue
        quote=reason.get('premise_quote');mechanism=reason.get('mechanism');evidence_id=reason.get('evidence_id')
        if not isinstance(quote,str) or not 20<=len(quote)<=600 or len(quote.split())<3 or not safe_text(mechanism,900):continue
        if not isinstance(evidence_id,str) or not re.fullmatch(r'E\d+',evidence_id):continue
        index=int(evidence_id[1:])
        if index>=len(passages):continue
        p=passages[index]
        if p.doc_id=='task' or quote not in p.text or not corpus.admits(p.doc_id,eid):continue
        row=rows[eid];point=row['point_forecast'];target=task.get('target',{}).get('name','target')
        old=raw.get('point_forecast');changed=not isinstance(old,(float,int)) or abs(point-old)>1e-6*max(1,abs(point))
        # Rebuild the implication from the final normalized answer, always.
        implication=f"For {eid}, the submitted {target} forecast is {point!r}"
        if 'label' in row:implication+=f" with label {row['label']}"
        implication+='; this mechanism is a forecast driver, not proof of a known future outcome.'
        if changed:
            if isinstance(old,(float,int)) and old*point<0:continue
            mechanism+=' The final value also reflects the available historical prior and forecast uncertainty.'
        if not safe_text(implication,900):continue
        start=p.start+p.text.index(quote)
        item={'reason_id':f'r{len(candidates)+1}','premise':quote,'mechanism':mechanism,'answer_implication':implication,
              'scope':{'entities':[eid]},'citations':[{'doc_id':p.doc_id,'span_start':start,'span_end':start+len(quote)}]}
        # Prefer distinct mechanisms, not the same boilerplate repeated per company.
        signature=re.sub(r'\W+',' ',mechanism.casefold()).strip()
        if any(signature==s for _,s,_ in candidates):continue
        relevance=len(set(re.findall(r'[a-z]{3,}',str(task.get('prompt','')).lower())) & set(re.findall(r'[a-z]{3,}',quote.lower())))
        candidates.append((relevance,signature,item))
    candidates.sort(key=lambda x:x[0],reverse=True);out=[]
    for _,_,item in candidates:
        if len(out)>=3:break
        trial=out+[item]
        projected=[{k:r[k] for k in ('entity_id','label','point_forecast','interval') if k in r} for r in rows.values()]
        if len(json.dumps(projected,ensure_ascii=False,separators=(',',':')).encode())>3000:break
        judged=[{k:r[k] for k in ('reason_id','premise','mechanism','answer_implication')} for r in trial]
        if len(json.dumps(judged,ensure_ascii=False,separators=(',',':')).encode())<=6000:
            out.append(item)
    return out
