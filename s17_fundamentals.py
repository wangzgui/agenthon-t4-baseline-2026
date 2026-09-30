"""Period-labelled financial facts from the current entity's eligible filings."""
from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from s17_semantics import base_field, infer

EPS = re.compile(r'\bdiluted\s+(?:EPS|earnings?(?:\s+per\s+(?:common\s+)?share)?|income(?:\s+from\s+continuing\s+operations)?)', re.I)
NUM = re.compile(r'(?<![\w.])\$?\(?-?(?:\d+(?:,\d{3})*\.\d+|\.\d+)\)?')
YEAR = re.compile(r'\b(?:19|20)\d{2}\b')
MONTH = re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', re.I)


@dataclass(frozen=True)
class StatementFact:
    metric: str
    current: float
    previous: float
    period: str
    comparison_period: str
    basis: str
    unit: str
    doc_id: str
    span_start: int
    span_end: int

    def prompt_record(self):
        return asdict(self)


def _number(token):
    return float(token.replace('$','').replace(',','').replace('(','-').replace(')',''))


def extract_facts(text, doc_id=''):
    facts = []
    for match in EPS.finditer(text):
        after = text[match.end():match.end()+260]
        before = text[max(0,match.start()-1600):match.start()]
        comparison = re.search(r'compared\s+(?:with|to)', after, re.I)
        start, end = max(0,match.start()-1600), min(len(text),match.end()+260)
        pair = None
        years = []
        period = ''
        if comparison:
            first = NUM.search(after[:comparison.start()])
            counterpart = after[comparison.end():]
            metric = EPS.search(counterpart[:170])
            second = NUM.search(counterpart[metric.end():]) if metric else NUM.search(counterpart)
            if not metric and re.search(r'\b(?:billion|million)\b',counterpart[:80],re.I):
                continue
            context = before[-350:] + after
            # A narrative pair must explicitly establish year-over-year and a
            # quarterly (not accumulated) period. QoQ comparisons are refused.
            years = [int(x) for x in YEAR.findall(context)]
            annual = bool(re.search(r'year.over.year|same (?:quarter|period).*?(?:prior|previous|last) year|year.ago', context,re.I))
            quarter = bool(re.search(r'three months|\bquarter\b', context,re.I))
            if first and second and quarter and (annual or len(set(years))==2 and max(years)-min(years)==1):
                if re.search(r'\b(?:billion|million)\b',after[:comparison.start()],re.I):
                    continue
                pair = (_number(first.group()),_number(second.group()))
                if years:
                    years = [max(years), max(years)-1]
                month = MONTH.findall(context)
                period = month[-1] if month else 'quarter'
                start = max(0,match.start()-350)
        else:
            # Read the header immediately preceding the EPS row, retaining its
            # period labels. Never infer a comparator from two adjacent figures.
            headers = list(re.finditer(r'(three|six|nine|twelve)\s+months(?:\s+ended)?', before,re.I))
            if not headers:
                continue
            header = before[headers[0].start():]
            if headers[0].group(1).lower() != 'three':
                continue
            years = [int(x) for x in YEAR.findall(header)]
            numbers = list(NUM.finditer(after[:110]))
            if len(years)>=2 and years[0]-years[1]==1 and len(numbers)>=2:
                months = MONTH.findall(header)
                # Two different end-months in the first comparison indicate QoQ.
                if len(months)>=2 and months[0].lower()!=months[1].lower():
                    continue
                pair = (_number(numbers[0].group()), _number(numbers[1].group()))
                period = months[0] if months else 'quarter'
                start = max(0,match.start()-1600)+headers[0].start()
        if pair is None or len(years)<2 or abs(pair[1])<0.01:
            continue
        if re.search(r'adjusted|non.GAAP',text[max(0,match.start()-50):match.end()+40],re.I):
            continue
        if not all(abs(value)<10000 for value in pair):
            continue
        facts.append(StatementFact('diluted_eps', *pair,
                     f'three_months:{period}:{years[0]}',
                     f'three_months:{period}:{years[1]}', 'diluted_reported',
                     'usd_per_share', doc_id, start, end))
    return facts


def _eps_pair(text):
    facts = extract_facts(text)
    return (facts[0].current, facts[0].previous) if facts else None


def statement_priors(task, corpus):
    meaning = infer(task)
    if meaning.kind!='growth' or 'eps' not in meaning.target_name.lower():
        return {}, {}
    priors, records = {}, {}
    for entity in task.get('entities',[]):
        eid = str(entity['entity_id'])
        candidates=[]
        for doc_id,text in corpus.texts.items():
            if doc_id=='task' or not corpus.admits(doc_id,eid) or not corpus.is_entity_specific(doc_id,eid):
                continue
            for fact in extract_facts(text,doc_id):
                candidates.append((corpus.dates.get(doc_id,''),fact))
        if not candidates:
            continue
        fact=max(candidates,key=lambda item:(item[0],item[1].period))[1]
        records[eid]=fact
        # Transfer growth between comparable reported quarters. Do not compare
        # the reported quarter with the target's different prior-year quarter.
        growth=100*(fact.current/fact.previous-1)
        if -300<=growth<=500:
            priors[eid]=growth
    return priors, records


def eps_growth_priors(task, corpus):
    return statement_priors(task,corpus)[0]
