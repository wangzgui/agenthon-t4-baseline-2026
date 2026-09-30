"""Source-anchored quantities and checked arithmetic; no outcome lookup."""
from __future__ import annotations
import math,re
from s17_semantics import convert,infer
NUM=r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
QUANTITY=re.compile(r'(?P<currency>\$|USD\s*)?(?P<number>'+NUM+r')\s*(?P<suffix>percentage points?|percent|basis points?|bps|bp|%|billion|million|thousand|bn|mm|[xX])?',re.I)

def amount(text):
    if not isinstance(text,str): return None
    m=QUANTITY.fullmatch(text.strip())
    if not m:return None
    n=float(m['number'].replace(',',''));suffix=(m['suffix'] or '').lower()
    unit='usd' if m['currency'] else 'unknown'
    scale=1.0
    if suffix in ('%','percent','percentage point','percentage points'):unit='pct'
    elif suffix in ('bp','bps','basis point','basis points'):unit='bps'
    elif suffix in ('x',):unit='ratio'
    elif suffix in ('billion','bn'):scale=1e9
    elif suffix in ('million','mm'):scale=1e6
    elif suffix=='thousand':scale=1e3
    if not math.isfinite(n*scale):return None
    return {'value':n*scale,'unit':unit,'scale':scale}

def enrich(facts,raw_facts=()):
    """Interpretations require literal supporting text before enabling arithmetic."""
    raw_by_quote={}
    for r in raw_facts if isinstance(raw_facts,list) else []:
        if isinstance(r,dict) and isinstance(r.get('quote'),str):raw_by_quote.setdefault(r['quote'],[]).append(r)
    out=[]
    for i,fact in enumerate(facts):
        f=dict(fact);f['fact_id']=f'F{i}';quote=f['quote'];options=raw_by_quote.get(quote,[]);r=options.pop(0) if options else {}
        text=r.get('value_text');metric=r.get('metric');period=r.get('period')
        q=amount(text) if isinstance(text,str) and text in quote else None
        if q and q['unit']!='unknown' and isinstance(metric,str) and len(metric.strip())>=2 and metric.casefold() in quote.casefold():
            f['quantity']={**q,'value_text':text,'metric':metric.casefold(),'basis':('adjusted' if re.search(r'adjusted|non[- ]gaap',quote,re.I) else 'gaap' if re.search(r'\bgaap\b',quote,re.I) else 'unspecified')}
            # A model-parsed period without a literal anchor remains unknown.
            if isinstance(period,str) and re.search(r'\d',period) and period.casefold() in quote.casefold():
                f['quantity']['period']=period.casefold()
            else:f['quantity']['period']='unknown'
        out.append(f)
    return out

def arithmetic(spec,facts):
    if not isinstance(spec,dict):return {'status':'absent'}
    by_id={f.get('fact_id'):f for f in facts if isinstance(f,dict)}
    a=by_id.get(spec.get('left_fact'),{}).get('quantity');b=by_id.get(spec.get('right_fact'),{}).get('quantity')
    if not a or not b:return {'status':'rejected','cause':'unverified_operand'}
    if a['metric']!=b['metric'] or a['basis']!=b['basis'] or a['period']=='unknown' or b['period']=='unknown':
        return {'status':'rejected','cause':'incomparable_metric_basis_or_period'}
    # Period-over-period calculations need distinct, non-accumulated periods.
    if a['period']==b['period'] or re.search(r'ytd|year.to.date|six months|nine months',a['period']+' '+b['period']):
        return {'status':'rejected','cause':'ambiguous_period_comparison'}
    bv=convert(b['value'],b['unit'],a['unit'])
    if bv is None:return {'status':'rejected','cause':'incompatible_units'}
    op=spec.get('operation');value=None;unit=a['unit']
    if op=='difference':value=a['value']-bv
    elif op=='growth_pct' and abs(bv)>1e-12:value=100*(a['value']/bv-1);unit='pct'
    elif op=='ratio' and abs(bv)>1e-12:value=a['value']/bv;unit='ratio'
    if value is None or not math.isfinite(value):return {'status':'rejected','cause':'invalid_operation_or_denominator'}
    return {'status':'verified','value':value,'unit':unit,'operation':op,'left_fact':spec['left_fact'],'right_fact':spec['right_fact'],'use':'observed_comparison_not_future_outcome'}

def assess(task,entity,raw):
    raw=dict(raw or {});facts=entity.get('_source_facts',[])
    result=arithmetic(raw.get('calculation'),facts)
    raw['_calculation_check']=result
    if result['status']=='rejected':raw['_decision_conflict']=True
    if result['status']=='verified' and raw.get('calculation',{}).get('result_is_forecast') is True:
        # Observed comparisons are context, never silently promoted into future truth.
        raw['_decision_conflict']=True
    components=raw.get('forecast_components')
    if isinstance(components,dict):
        try:
            a=float(components['baseline']);b=float(components['event_adjustment']);point=float(raw['point_forecast'])
            u=components.get('unit');m=infer(task)
            expected=convert(a+b,u,m.unit)
            if not all(math.isfinite(v) for v in (a,b,point)) or expected is None or abs(expected-point)>1e-6*max(1,abs(point),abs(expected)):
                raw['_decision_conflict']=True
            else:raw['_decomposition_checked']=True
        except (KeyError,TypeError,ValueError,OverflowError):raw['_decision_conflict']=True
    return raw


def comparisons(facts):
    out=[]
    for a in facts:
        for b in facts:
            if a.get('fact_id')==b.get('fact_id'):continue
            for op in ('difference','growth_pct'):
                r=arithmetic({'operation':op,'left_fact':a.get('fact_id'),'right_fact':b.get('fact_id')},facts)
                if r['status']=='verified':out.append(r)
                if len(out)>=8:return out
    return out
