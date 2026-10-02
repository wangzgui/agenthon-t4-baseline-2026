"""Conservative fact retention and explicit, quote-bound review corrections.

These checks establish structure and provenance, not semantic entailment.
No observed quote is treated as a future label or a fitted model coefficient.
"""
import math,re

def merge_facts(original, extracted):
    """Preserve every baseline fact; append unique bounded extraction records."""
    kept=[dict(f) for f in original if isinstance(f,dict)]
    seen={(f.get('evidence_id'),f.get('quote')) for f in kept}
    for fact in extracted[:3]:
        signature=(fact.get('evidence_id'),fact.get('quote'))
        if signature not in seen:
            kept.append(dict(fact,status='exact_quote_provisional_interpretation'))
            seen.add(signature)
    return kept

def point(row):
    value=row.get('point_forecast') if isinstance(row,dict) else None
    return float(value) if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) else None

def review_admission(draft, review, passages):
    a,b=point(draft),point(review)
    if b is None:return False,'missing_finite_review'
    if a is None:return True,'recover_missing_draft_with_quote_check'
    if abs(a-b)<=1e-6*max(1,abs(a),abs(b)):return True,'unchanged_numeric_point'
    if review.get('correction_kind') not in {'unit','period','arithmetic','consistency','omitted_driver'}:
        return False,'missing_correction_kind'
    explanation=review.get('change_reason')
    if not isinstance(explanation,str) or not 40<=len(explanation)<=500:
        return False,'missing_specific_change_reason'
    reason=review.get('reason')
    if not isinstance(reason,dict):return False,'missing_forecast_mechanism'
    quote=reason.get('premise_quote'); mechanism=reason.get('mechanism');eid=reason.get('evidence_id')
    if not isinstance(mechanism,str) or not 40<=len(mechanism)<=900:
        return False,'missing_forecast_mechanism'
    if not isinstance(quote,str) or not 20<=len(quote)<=280 or not isinstance(eid,str) or not re.fullmatch(r'E\d+',eid):
        return False,'invalid_mechanism_premise'
    i=int(eid[1:])
    if i>=len(passages) or quote not in passages[i].text or eid not in review.get('evidence_ids',[]):
        return False,'unanchored_mechanism_premise'
    if mechanism.strip()==quote.strip() or explanation.strip()==quote.strip():
        return False,'observation_without_decision_explanation'
    return True,'explicit_quote_bound_correction_semantics_provisional'
