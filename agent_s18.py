"""Agenthon Track 4 S1.8: cutoff-safe statistical priors plus House reasoning.

No family names or public unit identifiers drive predictions.  The only remote
dependency is the organizer-provided House model.  When it is unavailable, a
deterministic statistical fallback still emits a valid answer for every row.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import time
import re
import statistics
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from s18_semantics import (base_field, convert, derived_point, historical_candidate,
                           horizon_days, infer, robust_scale, unit_of)
from s18_numerics import collect_series, fit_prior, prior_for_entity, revision_prior, residuals_for_entity
from s18_fundamentals import _eps_pair, statement_priors
from s18_calibration import interval as calibrated_interval, fusion_weight
from s18_facts import select as fact_retrieve, local_facts, validate_facts
from s18_diagnostics import Trace, emit, transport, forecast as diagnostic_forecast
from s18_retrieval import supplement

STOP = set("the and for from with into using only each this that their will after before about your give whose when which must some are its per then than have has was were through between above below table task corpus frozen date report prediction forecast predict future support evidence claims label value point interval row rows name unit type doc document notes public private".split())
WORD = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?")
NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
BAD_KEYS = {"date", "year", "month", "quarter", "id", "cik", "amount", "size", "volume", "count", "open_interest"}


def tokens(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower()) if (len(w) > 1 or w.isdigit()) and w not in STOP]


def finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        x = float(str(value).replace(",", "").strip())
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def is_scale_or_identifier(key: str) -> bool:
    words = set(key.lower().split("_"))
    return bool(words & (BAD_KEYS - {"open_interest"})) or "open_interest" in key.lower()


@dataclass(frozen=True)
class Passage:
    doc_id: str
    start: int
    end: int
    text: str
    date: str


class _LegacyCorpus:
    def __init__(self, directory: Path, cutoff: str):
        self.cutoff = cutoff
        self.texts: dict[str, str] = {}
        self.passages: list[Passage] = []
        for path in sorted(directory.glob("*.json")):
            if path.name == "manifest.json":
                continue
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(doc, dict):
                continue
            doc_id = str(doc.get("doc_id") or path.stem)
            doc_date = str(doc.get("doc_date") or "")
            if not DATE.match(doc_date) or doc_date[:10] > cutoff[:10]:
                continue
            if isinstance(doc.get("text"), str):
                parts = [doc["text"]]
            else:
                parts = [str(s.get("text", "")) for s in doc.get("spans", []) if isinstance(s, dict)]
            whole = " ".join(parts)
            if not whole.strip() or doc_id in self.texts:
                continue
            self.texts[doc_id] = whole
            # The official offset convention joins span texts with one space.
            # Passage boundaries may differ, but each [start:end] is exact.
            for a, b in self._windows(whole):
                excerpt = whole[a:b]
                if excerpt.strip():
                    self.passages.append(Passage(doc_id, a, b, excerpt, doc_date))

    @staticmethod
    def _windows(text: str):
        n = len(text)
        i = 0
        while i < n:
            j = min(i + 700, n)
            if j < n:
                candidates = [text.rfind(mark, i + 220, j) for mark in ("\n", ". ", "; ", " | ")]
                k = max(candidates)
                if k > i + 220:
                    j = k + (2 if text[k:k+2] in (". ", "; ", " | ") else 1)
            if j <= i:
                j = min(i + 700, n)
            yield i, j
            i = j

    def valid(self, p: Passage) -> bool:
        return self.texts.get(p.doc_id, "")[p.start:p.end] == p.text


class Corpus(_LegacyCorpus):
    """Eligible documents plus the officially defined, row-scoped task source."""
    def __init__(self,directory,cutoff,task=None):
        super().__init__(directory,cutoff)
        self.dates={p.doc_id:p.date for p in self.passages}
        self.labels={}
        self.task_rows={}
        for path in (directory.parent/'manifest.json',directory/'manifest.json'):
            if not path.is_file():
                continue
            try:
                manifest=json.loads(path.read_text(encoding='utf-8'))
            except (ValueError,OSError):
                continue
            for entry in manifest.get('files',[]):
                if entry.get('role')!='corpus' or not str(entry.get('path','')).endswith('.json'):
                    continue
                doc_id=Path(entry['path']).stem
                if doc_id=='manifest':
                    continue
                ids=entry.get('entity_ids')
                if isinstance(ids,list) or entry.get('shared') is True:
                    self.labels[doc_id]=(set(ids or []),entry.get('shared') is True)
        if task:
            lines=[]
            offset=0
            for entity in task.get('entities',[]):
                line=json.dumps(entity,ensure_ascii=False,separators=(', ',': '))
                passage=Passage('task',offset,offset+len(line),line,cutoff)
                self.task_rows[str(entity['entity_id'])]=passage
                lines.append(line)
                offset+=len(line)+1
            self.texts['task']='\n'.join(lines)
            self.dates['task']=cutoff
            self.passages.extend(self.task_rows.values())

    def admits(self,doc_id,entity_id):
        if doc_id=='task':
            return entity_id in self.task_rows
        ids,shared=self.labels.get(doc_id,(set(),False))
        return shared or entity_id in ids

    def is_entity_specific(self,doc_id,entity_id):
        ids,shared=self.labels.get(doc_id,(set(),False))
        return entity_id in ids and not shared

    def admits_passage(self,p,entity_id):
        if not self.admits(p.doc_id,entity_id):
            return False
        if p.doc_id=='task':
            own=self.task_rows[entity_id]
            return own.start<=p.start<p.end<=own.end
        return True


def task_spec(task: dict) -> tuple[str, str, list[str]]:
    target = task.get("target") if isinstance(task.get("target"), dict) else {}
    target_type = str(target.get("type") or task.get("target_type") or "regression").lower()
    if target_type not in {"classification", "regression", "ranking"}:
        target_type = "regression"
    name = str(target.get("name") or task.get("target_name") or "target")
    labels = [str(x) for x in target.get("labels", [])] if isinstance(target.get("labels"), list) else []
    return target_type, name, labels


def entity_terms(entity: dict) -> tuple[set[str], set[str]]:
    identity, detail = set(), set()
    for key, value in entity.items():
        if not isinstance(value, str) or key in {"corpus_ref", "unit", "units", "currency"}:
            continue
        words = set(tokens(value))
        if key in {"entity_id", "name", "ticker", "symbol", "series_id", "series_name", "tenor", "cik", "reference_month", "ref_month"}:
            identity |= words
        else:
            detail |= words
    return identity, detail


def retrieve(task,entity,corpus,limit=6):
    return fact_retrieve(task,entity,corpus,limit)


def feature_hint(task: dict, entity: dict) -> dict:
    """Distinguish a historical target proxy from a starting level."""
    meaning = infer(task)
    result = {"target_kind": meaning.kind, "target_unit": meaning.unit,
              "numeric_quantity": meaning.numeric_description}
    proxy = historical_candidate(entity, meaning)
    if proxy is not None:
        result["recent_same_quantity"] = proxy
    base = base_field(entity, meaning)
    if base is not None:
        result["arithmetic_base"] = {"field": base[0], "value": base[1], "unit": base[2]}
        if meaning.kind == "change":
            result["formula"] = "(projected_level - arithmetic_base.value), converted from base unit to target unit"
        elif meaning.kind == "growth":
            result["formula"] = "100 * (projected_level / arithmetic_base.value - 1)"
    return result


def fallback_point(task: dict, entity: dict, all_entities: list[dict]) -> float:
    target_type, _, _ = task_spec(task)
    meaning = infer(task)
    if meaning.kind == "probability":
        return 0.5
    if target_type == "classification" and meaning.kind == "level":
        revised_base = finite(entity.get("latest_precutoff_estimate"))
        if revised_base is not None:
            return revised_base
    proxy = historical_candidate(entity, meaning)
    if proxy is not None:
        return proxy
    # For a change or growth target, a starting level is *not* a forecast.
    if meaning.kind in {"change", "growth"}:
        return 0.0
    peer_proxies = [historical_candidate(row, meaning) for row in all_entities]
    peer_proxies = [value for value in peer_proxies if value is not None]
    if peer_proxies:
        return statistics.median(peer_proxies)
    return 0.0


def tabular_evidence_values(task: dict, passages: list[Passage]) -> list[float]:
    """Find a historical column with matching quantity AND convertible units."""
    meaning = infer(task)
    target_words = set(tokens(meaning.target_name)) - {"next", "future", "predicted", "rank"}
    best: tuple[float, list[float]] | None = None
    for passage in passages:
        lines = passage.text.splitlines()
        for i, line in enumerate(lines):
            if "|" not in line or i + 2 >= len(lines):
                continue
            headings = [set(tokens(cell)) for cell in line.split("|")]
            if len(headings) < 3:
                continue
            overlap = [len(h & target_words) for h in headings]
            column = max(range(len(overlap)), key=overlap.__getitem__)
            if overlap[column] < 2:
                continue
            heading = line.split("|")[column].strip().lower()
            heading_words = set(tokens(heading))
            if meaning.kind == "change" and not heading_words & {"change", "delta", "mom", "yoy", "growth", "return"}:
                continue
            if meaning.kind == "growth" and not heading_words & {"growth", "yoy", "change"}:
                continue
            field_unit = unit_of(heading)
            if field_unit == "level" and overlap[column] >= 2:
                field_unit = meaning.unit
            if meaning.unit != "level" and field_unit != meaning.unit and convert(1.0, field_unit, meaning.unit) is None:
                continue
            observations = []
            for row in lines[i+1:i+32]:
                cells = row.split("|")
                if len(cells) != len(headings):
                    if observations:
                        break
                    continue
                x = finite(cells[column].strip().rstrip("%"))
                if x is not None:
                    if meaning.unit != "level" and field_unit != meaning.unit:
                        x = convert(x, field_unit, meaning.unit)
                    if x is None:
                        continue
                    observations.append(x)
            if observations:
                candidate = (float(overlap[column]) + min(len(observations), 5) / 10, observations)
                if best is None or candidate[0] > best[0]:
                    best = candidate
    return best[1] if best else []


def interval_for(task: dict, point: float, suggested: dict | None, entities: list[dict], history: list[float] | None = None) -> dict:
    level = finite(task.get("interval_level")) or 0.9
    lower = finite((suggested or {}).get("lo"))
    upper = finite((suggested or {}).get("hi"))
    target_type, target_name, _ = task_spec(task)
    if target_type == "classification" and ("probab" in str(task.get("prompt", "")).lower() or "probab" in target_name.lower()):
        point = min(max(point, 0), 1)
        return {"level": level, "lo": max(0.0, min(point - 0.35, lower if lower is not None else point - 0.35)), "hi": min(1.0, max(point + 0.35, upper if upper is not None else point + 0.35))}
    meaning = infer(task)
    floor = {"bps": 17.0 * math.sqrt(max(1, horizon_days(task))), "pct": 4.0,
             "ratio": 0.5, "probability": 0.35}.get(meaning.unit, 0.5)
    if meaning.unit == "pct" and meaning.kind == "growth":
        floor = 20.0
    if meaning.unit == "pct" and meaning.kind == "change" and "return" in str(task.get("prompt", "")).lower():
        floor = 12.0
    scale = max(abs(point) * (0.25 if meaning.kind == "level" else 0.45), floor)
    sigma = robust_scale(history or [])
    if sigma is not None:
        scale = max(scale, 1.65 * sigma * math.sqrt(max(1.0, horizon_days(task) / 30.0)))
    if lower is None or upper is None or lower > upper:
        lower, upper = point - scale, point + scale
    else:
        lower, upper = min(lower, point - scale), max(upper, point + scale)
    return {"level": level, "lo": lower, "hi": upper}


def cross_section(task: dict, entity: dict) -> dict:
    """Small robust tabular summary; no fitted model or labeled data assumed."""
    roster = task.get("entities", [])
    if not isinstance(roster, list):
        return {}
    _, target, _ = task_spec(task)
    target_words = set(tokens(target))
    candidates = []
    for key, value in entity.items():
        x = finite(value)
        if x is None or is_scale_or_identifier(key):
            continue
        peers = [finite(row.get(key)) for row in roster if isinstance(row, dict)]
        peers = [v for v in peers if v is not None]
        if len(peers) < 2:
            continue
        median = statistics.median(peers)
        mad = statistics.median([abs(v - median) for v in peers])
        relevance = len(set(tokens(key)) & target_words)
        candidates.append((relevance, key, {"value": x, "peer_median": median,
                                            "robust_z": max(-8.0, min(8.0, (x - median) / max(1.4826 * mad, 1e-8))) }))
    candidates.sort(reverse=True)
    return {key: summary for _, key, summary in candidates[:8]}


def house_url() -> str | None:
    endpoint = os.environ.get("MODEL_ENDPOINT", "").rstrip("/")
    return (endpoint + ("" if endpoint.endswith("/v1") else "/v1") + "/chat/completions") if endpoint else None


def house_batch(task: dict, batch: list[tuple[int, dict, list[Passage]]],
                replicate: int = 0, numeric_prior: dict[str, float] | None = None) -> list[dict] | None:
    url = house_url()
    model = os.environ.get("MODEL_NAME")
    if not url or not model:
        return None
    target_type, target_name, labels = task_spec(task)
    meaning = infer(task)
    rows = []
    for _, entity, passages in batch:
        rows.append({
            "entity_id": entity["entity_id"],
            "fields": {k: v for k, v in entity.items() if k != "corpus_ref"},
            "numeric_feature_hint": feature_hint(task, entity),
            "rolling_origin_prior": (numeric_prior or {}).get(str(entity["entity_id"])),
            "cross_section": cross_section(task, entity),
            "source_facts": entity.get("_source_facts") or local_facts(passages,8),
            "review_draft": entity.get("_draft_forecast") if replicate else None,
            "prior_status": entity.get("_prior_status", "no validated local prior"),
            "evidence": [{"id": f"E{i}", "text": p.text} for i, p in enumerate(passages)],
        })
    roster_overview = [{
        "entity_id": entity.get("entity_id"),
        "name": entity.get("name"),
        "numeric_fields": {k: v for k, v in entity.items() if finite(v) is not None and not k.endswith("date")},
    } for entity in task.get("entities", [])]
    system = (
        "You are a forecasting analyst using ONLY the supplied pre-cutoff corpus passages. "
        "Return one strict JSON object, no markdown or prose. Do not use your memory of realized events. "
        "Forecast the unknown future, not a current value. Preserve target units exactly. "
        "Give a genuine numerical forecast when the target is numeric: distinguish the starting "
        "level, the change, and a growth rate. Zero change is a substantive prediction, not a "
        "default for uncertainty. A rolling-origin prior, when supplied, is only a pre-cutoff "
        "statistical estimate; weigh it against event evidence, never treat it as a known outcome. "
        "For a roster of maturities, reason about one shared curve "
        "shock with maturity-dependent sensitivity. "
        "Compare every requested row with the entire roster, using the roster overview even when "
        "only a subset of rows is requested in this call. Each row must cite evidence IDs that "
        "actually support the forecast rationale. If uncertain, be conservative."
    )
    user = json.dumps({
        "instruction": (
            ("Review the supplied draft independently: check quantities, units, periods, peer ordering and omitted drivers; correct only with a reason grounded in the provided facts. " if replicate else "") +
            "Return {predictions:[{entity_id,label,point_forecast,point_unit,projected_level,interval:{lo,hi},"
            "evidence_ids:[E0,...],evidence_quote}]} exactly once per row. "
            "For classification, label must be one allowed label. If target is a probability, point is in [0,1]; "
            "otherwise point is the numeric forecast named in the prompt. For ranking, point must "
            "be the underlying metric (larger is ranked higher), not rank number. For RANKING, "
            "compare all roster rows jointly, decide a full ordering, then assign distinct "
            "metric forecasts consistent with that ordering. Use the same unit and horizon "
            "for every row; do not simply order rows by their starting level. "
            "For a CHANGE target, point_forecast is the change in the target unit, never the starting level; "
            "if an arithmetic_base is supplied, also return projected_level in its base unit. "
            "For GROWTH, projected_level is a future comparable level and point_forecast is percent growth. "
            "For a revision target, compare the forecast revised level with the supplied "
            "pre-cutoff estimate before assigning an up/down label. Distinguish historical "
            "same-stage revisions from revisions already incorporated in that estimate. "
            "For directional classification, first decide the direction, then provide a numeric "
            "forecast on the corresponding side of any stated threshold. "
            "Use evidence_quote as a short VERBATIM substring of a cited passage; don't invent quotes. "
            "Source facts with model-parsed period/unit metadata are provisional; verify those interpretations against the quote and surrounding evidence. "
            "Use drivers to forecast a future level or change rather than copying an observed fact as the answer. "
            "Explain the event-relative component mentally: expected event effects already priced in must not be counted again. "
            "Set point_unit to the target numeric_unit. Do not claim future outcomes are known."
        ),
        "target": {"type": target_type, "name": target_name, "labels": labels,
                   "numeric_kind": meaning.kind, "numeric_unit": meaning.unit,
                   "numeric_description": meaning.numeric_description},
        "cutoff_date": task.get("cutoff_date"),
        "resolution_date": task.get("resolution_date"),
        "prompt": str(task.get("prompt", ""))[:3500],
        "roster_overview": roster_overview,
        "rows": rows,
    }, ensure_ascii=False, separators=(",", ":"))
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0 if replicate == 0 else 0.2,
        "seed": 2609 + replicate,
        "max_tokens": 4000,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("MODEL_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    started=time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=max(1,min(65,task.get("_deadline",time.monotonic()+65)-time.monotonic()))) as response:
            body = json.loads(response.read().decode("utf-8"))
        transport(task,"review" if replicate else "forecast",payload,body,started,len(batch))
        content = body["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        match = re.search(r"\{.*\}", content, re.S)
        parsed = json.loads(match.group(0) if match else content)
        return parsed.get("predictions") if isinstance(parsed.get("predictions"), list) else None
    except Exception as exc:
        emit(task,"transport_failure",{"stage":"review" if replicate else "forecast","exception_type":type(exc).__name__})
        # Never write endpoint, credentials, or raw response into the answer/log.
        print(f"House batch unavailable: {type(exc).__name__}", flush=True)
        return None


def house_extract(task,batch):
    url=house_url(); model=os.environ.get('MODEL_NAME')
    if not url or not model or time.monotonic()>=task['_deadline']-2:
        return []
    user={'target':task.get('target'),'prompt':task.get('prompt'),
          'cutoff_date':task.get('cutoff_date'),
          'rows':[{'entity_id':e['entity_id'],'features':{k:v for k,v in e.items() if not k.startswith('_')},
                   'evidence':[{'id':f'E{i}','text':p.text,'doc_date':p.date} for i,p in enumerate(passages)]}
                  for _,e,passages in batch]}
    instruction=('Extract facts relevant to the requested future target, never forecast known outcomes. '
                 'Return strict JSON {facts_by_entity:[{entity_id,facts:[{evidence_id,quote,metric,period,comparison_period,unit,basis,driver_role}]}]}. '
                 'At most 3 facts per entity; quote must be an exact substring (12-250 characters) of that row evidence. '
                 'Prefer informative current observations, comparable historical observations, and forward guidance or event expectations. '
                 'Distinguish quarterly vs year-to-date, level vs change, GAAP vs adjusted, and expected vs surprise effects. '
                 'If the quote does not establish a period, unit or comparison, mark it unknown; do not invent metadata. '
                 'Read ONLY supplied evidence and features. Every interpretation is provisional and must follow its quote.')
    payload={'model':model,'messages':[{'role':'system','content':instruction},
              {'role':'user','content':json.dumps(user,ensure_ascii=False)}],
             'temperature':0,'seed':2616,'max_tokens':4000,
             'chat_template_kwargs':{'enable_thinking':False}}
    headers={'Content-Type':'application/json'}
    if os.environ.get('MODEL_TOKEN'):
        headers['Authorization']='Bearer '+os.environ['MODEL_TOKEN']
    request=urllib.request.Request(url,data=json.dumps(payload).encode(),headers=headers,method='POST')
    started=time.monotonic()
    try:
        with urllib.request.urlopen(request,timeout=max(1,min(65,task['_deadline']-time.monotonic()))) as response:
            body=json.loads(response.read().decode())
        transport(task,'facts',payload,body,started,len(batch))
        content=body['choices'][0]['message']['content']
        if isinstance(content,list):
            content=''.join(p.get('text','') for p in content if isinstance(p,dict))
        match=re.search(r'\{.*\}',content,re.S)
        data=json.loads(match.group() if match else content)
        return data.get('facts_by_entity',[]) if isinstance(data,dict) else []
    except Exception as exc:
        emit(task,'transport_failure',{'stage':'facts','exception_type':type(exc).__name__})
        print('House fact extraction unavailable:',type(exc).__name__,flush=True)
        return []


def review_or_medoid(task,batch,options):
    fallback=house_medoid(options,task_spec(task)[0])
    first={str(p.get('entity_id')):p for p in options[0]}
    review={str(p.get('entity_id')):p for p in options[1]}
    chosen=[]
    accepted=[]
    for _,entity,passages in batch:
        eid=str(entity['entity_id']); raw=review.get(eid)
        quote=raw.get('evidence_quote','') if raw else ''
        ids=raw.get('evidence_ids',[]) if raw else []
        valid=bool(raw and finite(raw.get('point_forecast')) is not None and not raw.get('_unit_conflict')
                   and isinstance(quote,str) and 12<=len(quote)<=280)
        anchored=False
        for evidence_id in ids if isinstance(ids,list) else []:
            if isinstance(evidence_id,str) and re.fullmatch(r'E\d+',evidence_id):
                index=int(evidence_id[1:])
                if index<len(passages) and quote and quote in passages[index].text:
                    anchored=True
        accepted.append(valid and anchored)
        emit(task,'review_check',{'entity_id':eid,'finite_forecast':bool(raw and finite(raw.get('point_forecast')) is not None),'unit_conflict':bool(raw and raw.get('_unit_conflict')),'valid_quote_length':isinstance(quote,str) and 12<=len(quote)<=280,'quote_anchored':anchored,'tentative_accept':valid and anchored,'ranking_group_can_override':task_spec(task)[0]=='ranking'})
        selected=raw if valid and anchored else first.get(eid)
        if selected is not None and eid in first and eid in review:
            selected=dict(selected)
            a=finite(first[eid].get('point_forecast')); b=finite(review[eid].get('point_forecast'))
            if a is not None and b is not None:
                selected['_house_disagreement']=abs(a-b)*.5
        chosen.append(selected)
    if task_spec(task)[0]=='ranking':
        # Keep an entire coherent forecast set, never a mixture of independent
        # rank scales. Partial or unanchored review falls back to first draft.
        if chosen and all(accepted):
            return chosen
        return options[0] or fallback
    return [p for p in chosen if p is not None] or fallback


def canonical_prediction(task,entity,raw):
    raw=dict(raw or {})
    meaning=infer(task)
    point=finite(raw.get('point_forecast'))
    declared=raw.get('point_unit')
    aliases={'%':'pct','percent':'pct','percentage_points':'pct','bp':'bps','basis_points':'bps','USD':'usd'}
    unit=aliases.get(declared,declared)
    direct_explicit=unit==meaning.unit
    if point is not None and unit and unit!=meaning.unit:
        converted=convert(point,unit,meaning.unit)
        if converted is None and unit=='pct' and meaning.unit=='probability':
            converted=point/100
        if converted is None:
            point=None
            raw['_unit_conflict']=True
        else:
            point=converted
            direct_explicit=True
    derived=derived_point(entity,meaning,raw.get('projected_level'))
    if derived is not None and math.isfinite(derived):
        conflict=point is not None and abs(point-derived)>max(1e-6,.05*max(abs(point),abs(derived),1.0))
        raw['_unit_conflict']=bool(raw.get('_unit_conflict')) or conflict
        if conflict:
            raw['_numeric_disagreement']=abs(point-derived)
        # An explicitly typed direct forecast is authoritative over a
        # conflicting auxiliary level; otherwise transparent arithmetic wins.
        if not conflict or not direct_explicit:
            point=derived
    if point is not None:
        raw['point_forecast']=point
    else:
        raw.pop('point_forecast',None)
    raw.pop('projected_level',None)
    # point_forecast is now canonical; do not convert it again in normalize.
    raw['point_unit']=meaning.unit
    return raw


def house_medoid(responses: list[list[dict]], target_type: str) -> list[dict]:
    """Choose an observed forecast near the ensemble center, retaining its citations."""
    valid = [response for response in responses if isinstance(response, list) and response]
    if not valid:
        return []
    if target_type == "ranking":
        largest = max(len(response) for response in valid)
        valid = [response for response in valid if len(response) == largest]
    if len(valid) == 1:
        return valid[0]
    indexed = [{str(row.get("entity_id")): row for row in response
                if isinstance(row, dict) and isinstance(row.get("entity_id"), str)}
               for response in valid]
    shared = set.intersection(*(set(rows) for rows in indexed))
    if not shared:
        return valid[0]
    if target_type == "ranking":
        ranks = []
        for rows in indexed:
            ordered = sorted(shared, key=lambda entity_id: finite(rows[entity_id].get("point_forecast")) or 0.0)
            ranks.append({entity_id: rank for rank, entity_id in enumerate(ordered)})
        centers = {entity_id: statistics.median(run[entity_id] for run in ranks)
                   for entity_id in shared}
        chosen = min(range(len(indexed)), key=lambda i:
                     sum(abs(ranks[i][entity_id] - centers[entity_id]) for entity_id in shared))
        return valid[chosen]
    output = []
    for entity_id in dict.fromkeys(entity_id for rows in indexed for entity_id in rows):
        options = [rows[entity_id] for rows in indexed if entity_id in rows]
        numbers = [finite(row.get("point_forecast")) for row in options]
        available = [(row, value) for row, value in zip(options, numbers) if value is not None]
        if not available:
            output.append(options[0])
            continue
        center = statistics.median(value for _, value in available)
        selected=dict(min(available, key=lambda pair: abs(pair[1] - center))[0])
        selected["_house_disagreement"]=statistics.median(abs(value-center) for _,value in available)*1.4826
        output.append(selected)
    return output


def make_claim(corpus: Corpus, passages: list[Passage], raw: dict, entity_id: str) -> list[dict]:
    ids = raw.get("evidence_ids") if isinstance(raw, dict) else []
    if not isinstance(ids, list):
        ids = []
    selected: list[Passage] = []
    for item in ids[:2]:
        if isinstance(item, str) and re.fullmatch(r"E\d+", item):
            index = int(item[1:])
            if index < len(passages) and passages[index] not in selected:
                selected.append(passages[index])
    if not selected and passages:
        selected=[passages[0]]
    # One or two relevant extractive claims; no blind padding with other firms.
    selected=[p for p in selected if corpus.admits_passage(p,entity_id)][:2]
    if not selected:
        own=corpus.task_rows.get(entity_id)
        selected=[own] if own else []
    quote = str(raw.get("evidence_quote") or "") if isinstance(raw, dict) else ""
    claims = []
    for p in selected[:3]:
        if not corpus.valid(p):
            continue
        if 12 <= len(quote) <= 280 and quote in p.text:
            quote_at = p.text.index(quote)
            # Keep neighboring facts as NLI premise context; the quote only
            # anchors the source passage and must occur verbatim within it.
            start = p.start + max(0, quote_at - 120)
            end = p.start + min(len(p.text), quote_at + len(quote) + 240)
            claim_text = quote
        else:
            start, end = p.start, p.end
            claim_text = p.text[:350]
            if claim_text.rfind(" ")>100:
                claim_text=claim_text[:claim_text.rfind(" ")]
        claims.append({"doc_id": p.doc_id, "span_start": start, "span_end": end, "claim": claim_text})
    return claims


def threshold_label(task: dict, entity: dict, point: float) -> str | None:
    """Apply explicit directional thresholds when labels and quantity agree."""
    _, _, labels = task_spec(task)
    meaning = infer(task)
    if meaning.unit not in {"pct", "bps"}:
        return None
    positive = next((label for label in labels if any(w in label.lower() for w in ("positive", "up", "increase"))), None)
    negative = next((label for label in labels if any(w in label.lower() for w in ("negative", "down", "decrease"))), None)
    middle = next((label for label in labels if any(w in label.lower() for w in ("flat", "neutral", "inline"))), None)
    if not all((positive, negative, middle)):
        return None
    thresholds = [(key, finite(value)) for key, value in entity.items() if "threshold" in key.lower()]
    thresholds = [(key, value) for key, value in thresholds if value is not None and unit_of(key) == meaning.unit]
    if not thresholds:
        return None
    threshold = min(abs(value) for _, value in thresholds)
    if point > threshold:
        return positive
    if point < -threshold:
        return negative
    return middle


def _directional_threshold(task: dict, entity: dict, raw_label: str | None,
                           point: float) -> tuple[str | None, float]:
    """Preserve a valid model direction and reconcile the point to its band."""
    _, _, labels = task_spec(task)
    derived = threshold_label(task, entity, point)
    if derived is None:
        return None, point
    if raw_label not in labels:
        return derived, point
    positive = next((x for x in labels if any(w in x.lower() for w in ("positive", "up", "increase"))), None)
    negative = next((x for x in labels if any(w in x.lower() for w in ("negative", "down", "decrease"))), None)
    middle = next((x for x in labels if any(w in x.lower() for w in ("flat", "neutral", "inline"))), None)
    thresholds = [abs(value) for key, raw in entity.items() if "threshold" in key.lower()
                  and unit_of(key) == infer(task).unit if (value := finite(raw)) is not None]
    if not thresholds:
        return raw_label, point
    threshold = min(thresholds)
    margin = max(0.05 * threshold, 1e-6)
    if raw_label == positive:
        point = max(point, threshold + margin)
    elif raw_label == negative:
        point = min(point, -threshold - margin)
    elif raw_label == middle:
        point = max(-threshold, min(threshold, point))
    return raw_label, point


def _uncertainty_direction(task: dict, entity: dict, label: str, point: float,
                           interval: dict) -> tuple[str, float]:
    """Use a numeric predictive interval to resolve thresholded ternary labels."""
    _, _, labels = task_spec(task)
    positive = next((x for x in labels if "positive" in x.lower()), None)
    negative = next((x for x in labels if "negative" in x.lower()), None)
    middle = next((x for x in labels if "flat" in x.lower()), None)
    if not all((positive, negative, middle)) or label != middle:
        return label, point
    meaning = infer(task)
    thresholds = [abs(value) for key, raw in entity.items() if "threshold" in key.lower()
                  and unit_of(key) == meaning.unit if (value := finite(raw)) is not None]
    if not thresholds:
        return label, point
    threshold = min(thresholds)
    sigma = (interval["hi"] - interval["lo"]) / (2 * 1.645)
    if sigma <= 0:
        return label, point
    def cdf(z: float) -> float:
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))
    p_negative = cdf((-threshold - point) / sigma)
    p_positive = 1 - cdf((threshold - point) / sigma)
    p_flat = max(0.0, 1 - p_negative - p_positive)
    if max(p_negative, p_positive) > p_flat + 0.05:
        if p_positive >= p_negative:
            return positive, max(point, threshold * 1.05)
        return negative, min(point, -threshold * 1.05)
    return label, point


def normalize(task: dict, entity: dict, raw: dict | None, passages: list[Passage],
              corpus: Corpus, learned_prior: float | None = None,
              prior_weight: float | None = None, residuals=(), calibration_origins=0) -> dict:
    raw = canonical_prediction(task,entity,raw)
    target_type, _, labels = task_spec(task)
    entities = task.get("entities", [])
    meaning = infer(task)
    fallback = fallback_point(task, entity, entities)
    history = tabular_evidence_values(task, passages)
    if target_type != "classification" and history:
        fallback = statistics.median(history[-min(5, len(history)):])
    point = finite(raw.get("point_forecast"))
    weight=prior_weight if prior_weight is not None else 0.0
    if point is None:
        point=learned_prior if learned_prior is not None and target_type!='classification' else fallback
        if learned_prior is not None:
            weight=1.0
    elif learned_prior is not None and target_type in {'regression','ranking'}:
        if raw.get("_unit_conflict"):
            weight=max(weight,0.65)
        point=weight*learned_prior+(1-weight)*point
    # Do not blend untyped numbers harvested from arbitrary passages into a
    # typed forecast. Statistical and statement priors are already canonical.
    if target_type == "classification":
        label = raw.get("label")
        if label not in labels:
            # Label/probability semantics cannot be inferred reliably for
            # arbitrary unseen tasks; the prompt and House model decide first.
            label = labels[0] if labels else "unknown"
        if meaning.unit == "probability":
            point = min(max(point, 0.0), 1.0)
        reconciled, point = _directional_threshold(task, entity, raw.get("label"), point)
        if reconciled is not None:
            label = reconciled
        elif set(labels) == {"up", "down"} and "revis" in meaning.target_name.lower():
            base = finite(entity.get("latest_precutoff_estimate"))
            if base is not None:
                prior = revision_prior(entity, corpus)
                if prior is not None:
                    delta, positive_fraction, count = prior
                    # Use a vintage-derived sign only when the historical
                    # same-stage revisions strongly agree.  Otherwise House
                    # retains control of the classification label.
                    if count >= 4 and (positive_fraction >= 0.75 or positive_fraction <= 0.25):
                        point = 0.65 * point + 0.35 * (base + delta)
                        if abs(point - base) > max(1e-8, 0.1 * abs(delta)):
                            label = "up" if point > base else "down"
    predicted_interval = interval_for(task, point, raw.get("interval") if isinstance(raw.get("interval"), dict) else None, entities, history)
    if target_type == "classification":
        label, adjusted = _uncertainty_direction(task, entity, label, point, predicted_interval)
        if adjusted != point:
            point = adjusted
            predicted_interval = interval_for(task, point, raw.get("interval") if isinstance(raw.get("interval"), dict) else None, entities, history)
    predicted_interval, interval_source = calibrated_interval(
        task,point,raw.get("interval"),residuals,calibration_origins,
        max(finite(raw.get("_house_disagreement")) or 0.0,
            min(abs(point)*.25+1.0,finite(raw.get("_numeric_disagreement")) or 0.0)*.25),learned_prior,weight)
    result = {
        "entity_id": entity["entity_id"],
        "point_forecast": point,
        "interval": predicted_interval,
        "claims": make_claim(corpus, passages, raw,str(entity["entity_id"])),
    }
    if target_type == "classification":
        result["label"] = label
    emit(task,'final_prediction',{'entity_id':entity['entity_id'],'fallback':fallback,'statistical_prior':learned_prior,'actual_weight':weight,'house':diagnostic_forecast(raw),'final_point':point,'final_label':result.get('label'),'interval':predicted_interval,'interval_source':interval_source,'calibration_origins':calibration_origins})
    return result


def run(task_path: Path, corpus_path: Path, out_path: Path) -> dict:
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["_deadline"]=time.monotonic()+480
    task["_trace"]=Trace(task)
    entities = task.get("entities", [])
    if not isinstance(entities, list):
        raise ValueError("task entities must be a list")
    corpus = Corpus(corpus_path, str(task.get("cutoff_date", "9999-12-31")),task)
    if not corpus.passages:
        raise ValueError("no eligible pre-cutoff corpus passages")
    prepared = [(i, e, retrieve(task, e, corpus)) for i, e in enumerate(entities)]
    target_type, _, _ = task_spec(task)
    meaning = infer(task)
    series = collect_series(corpus, str(task.get("cutoff_date", "9999-12-31")), meaning) \
        if target_type in {"regression", "ranking"} and meaning.kind in {"change", "level"} else []
    fitted = fit_prior(task, series) if series else None
    learned = {str(e["entity_id"]): prior_for_entity(e, series, fitted, meaning)
               for e in entities} if fitted is not None else {}
    fundamentals, statement_records = statement_priors(task, corpus)
    for entity_id, value in fundamentals.items():
        learned.setdefault(entity_id, value)
    # Put validated statement facts next to their precise source slices.
    for index,entity,passages in prepared:
        fact=statement_records.get(str(entity['entity_id']))
        if fact:
            entity['_statement_fact']=fact.prompt_record()
            text=corpus.texts[fact.doc_id]
            start=max(fact.span_start,fact.span_end-1800)
            premise=Passage(fact.doc_id,start,fact.span_end,text[start:fact.span_end],corpus.dates[fact.doc_id])
            passages.insert(0,premise)
            del passages[6:]
    supplementation_enabled=os.environ.get('AGENTHON_SUPPLEMENT_RETRIEVAL','1')!='0'
    for i,(index,entity,passages) in enumerate(prepared):
        original=[{'doc_id':p.doc_id,'start':p.start,'end':p.end} for p in passages]
        if supplementation_enabled:
            expanded,coverage=supplement(task,entity,passages,corpus)
            passages[:]=expanded
        else:coverage={'disabled':True,'added':[]}
        emit(task,'retrieval',{'entity_id':entity['entity_id'],'initial':original,'supplement':coverage,'passage_count':len(passages)})
    emit(task,'prior_validation',{'model':fitted.model if fitted else None,'validation_score':fitted.validation_score if fitted else None,
         'validation_baseline':fitted.validation_baseline if fitted else None,'fusion_weight':fusion_weight(fitted) if fitted else None})
    for _,entity,passages in prepared:
        entity['_source_facts']=local_facts(passages,8)
        entity['_prior_status']=('selected_on_pre_cutoff_history' if fitted and fitted.model!='zero'
                                else 'reported_quarter_transfer_proxy' if str(entity['entity_id']) in fundamentals
                                else 'persistence_baseline_or_no_fitted_signal')
    # Budget all phases together. Every scheduled batch receives a forecast
    # call; extraction and draft review spend only the remaining request budget.
    batch_size=(len(prepared) if target_type=='ranking' and 0<len(prepared)<=12 else
                max(6,math.ceil(len(prepared)/25)))
    batches=[prepared[i:i+batch_size] for i in range(0,len(prepared),batch_size)]
    house_results={}; house_calls=0; fact_calls=0; verified_facts=0
    available=bool(house_url() and os.environ.get('MODEL_NAME'))
    count=len(batches)
    if available and count:
        extract_count=count if 2*count<=25 else max(0,25-count)
        review_count=count if target_type in {'regression','ranking'} and 3*count<=25 else 0
        def phase(function,indices):
            nonlocal house_calls
            result={}
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                futures={pool.submit(function,i):i for i in indices if time.monotonic()<task['_deadline']-2}
                house_calls+=len(futures)
                for future in concurrent.futures.as_completed(futures):
                    result[futures[future]]=future.result() or []
            return result
        extracted=phase(lambda i:house_extract(task,batches[i]),range(extract_count))
        fact_calls=len(extracted)
        for i,fact_rows in extracted.items():
            by_id={str(e['entity_id']):(e,p) for _,e,p in batches[i]}
            for record in fact_rows:
                if not isinstance(record,dict) or str(record.get('entity_id')) not in by_id:
                    continue
                entity,passages=by_id[str(record['entity_id'])]
                facts=validate_facts(record.get('facts'),passages,str(entity['entity_id']))
                emit(task,'fact_validation',{'entity_id':entity['entity_id'],'proposed':len(record.get('facts',[])) if isinstance(record.get('facts'),list) else 0,'accepted':len(facts),'facts':[{'doc_id':f['doc_id'],'span_start':f['span_start'],'span_end':f['span_end']} for f in facts]})
                if facts:
                    entity['_source_facts']=facts
                    verified_facts+=len(facts)
        first=phase(lambda i:house_batch(task,batches[i],0,learned),range(count))
        for i,predictions in first.items():
            by_id={str(e['entity_id']):e for _,e,_ in batches[i]}
            for raw in predictions:
                if isinstance(raw,dict) and str(raw.get('entity_id')) in by_id:
                    by_id[str(raw['entity_id'])]['_draft_forecast']=canonical_prediction(task,by_id[str(raw['entity_id'])],raw)
        second=phase(lambda i:house_batch(task,batches[i],1,learned),range(review_count))
        for i in range(count):
            by_id={str(e['entity_id']):e for _,e,_ in batches[i]}
            options=[[canonical_prediction(task,by_id[str(p.get('entity_id'))],p)
                      for p in response if isinstance(p,dict) and str(p.get('entity_id')) in by_id]
                     for response in (first.get(i,[]),second.get(i,[]))]
            # A checked draft is preferred only if its cited quote is real;
            # otherwise retain an observed robust forecast with its own citations.
            selected=review_or_medoid(task,batches[i],options)
            emit(task,'candidate_selection',{'batch':i,'expected_entities':list(by_id),'first':[diagnostic_forecast(p) for p in options[0]],
                 'review':[diagnostic_forecast(p) for p in options[1]],'selected':[diagnostic_forecast(p) for p in selected],
                 'missing_first':[eid for eid in by_id if eid not in {str(p.get('entity_id')) for p in options[0]}],
                 'missing_review':[eid for eid in by_id if eid not in {str(p.get('entity_id')) for p in options[1]}]})
            for pred in selected:
                house_results[str(pred['entity_id'])]=pred
    rows = [normalize(task, entity, house_results.get(str(entity["entity_id"])), passages, corpus,
                      learned.get(str(entity["entity_id"])),
                      fusion_weight(fitted) if fitted and learned.get(str(entity["entity_id"])) is not None
                      and str(entity["entity_id"]) not in fundamentals else fusion_weight(None,True),
                      *(residuals_for_entity(entity,series,fitted,meaning) if fitted else ((),0)))
            for _, entity, passages in prepared]
    answer = {
        "task_id": task["task_id"],
        "schema_version": "3",
        "target_type": task_spec(task)[0],
        "entity_predictions": rows,
        "notes": {"agent": "agenthon-t4-baseline-s1.8", "house_requests": house_calls,
                  "model_rows": len(house_results), "historical_prior": fitted.model if fitted else "none",
                  "prior_examples": fitted.examples if fitted else 0,
                  "statement_prior_rows": len(fundamentals),
                  "calibration_origins": fitted.calibration_origins if fitted else 0,
                  "calibration_start": fitted.calibration_start if fitted else None,
                  "entity_ownership_checked": True,
                  "fact_requests":fact_calls,"verified_house_facts":verified_facts,
                  "fallback_rows":len(entities)-len(house_results),
                  "numeric_conflicts":sum(bool(r.get("_unit_conflict")) for r in house_results.values())},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(answer, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    task["_trace"].save()
    return answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("verb", choices=["analyze"])
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    answer = run(args.task, args.corpus, args.out)
    print(f"wrote {args.out}: {len(answer['entity_predictions'])} rows", flush=True)


if __name__ == "__main__":
    main()
