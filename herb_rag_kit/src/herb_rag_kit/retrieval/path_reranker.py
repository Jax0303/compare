from __future__ import annotations
import json, math
from typing import List, Dict, Any, Tuple


def _feature_vector(item: Dict[str, Any], schema: Dict[str, Any], paths: List[Dict[str, Any]]) -> List[float]:
    """Heuristic features for an item (graph row or doc hit)."""
    score = float(item.get("_score", 0.0))
    p = (item.get("p") or item.get("predicate") or "").strip()
    s = (item.get("s") or item.get("e1") or item.get("name") or "").strip()
    o = (item.get("o") or item.get("e2") or "").strip()
    # schema-based features
    sc = schema.get(p) or {}
    support = float(sc.get("support", 0.0))
    reciprocity = float(sc.get("reciprocity", 0.0))
    avg_obj = float(sc.get("avg_obj_per_sp", 1.0))
    func_like = 1.0 / max(1.0, avg_obj)
    # path features
    has_pair = 1.0 if s and o and any((s == pp.get("s") and o == pp.get("t")) for pp in paths) else 0.0
    num_paths = float(len(paths))
    return [score, support, reciprocity, func_like, has_pair, num_paths]


def rerank_with_lgbm(candidates: List[Dict[str, Any]], schema: Dict[str, Any], paths: List[Dict[str, Any]], model_path: str | None) -> List[Dict[str, Any]]:
    """
    LightGBM 점수를 사용해 재랭크. 모델이 없으면 휴리스틱 점수에 의존.
    입력 후보는 `rerank_results` 전 단계 결합 결과를 기대.
    """
    feats = [_feature_vector(it, schema, paths) for it in candidates]
    scores = []
    booster = None
    if model_path and model_path.strip():
        try:
            import lightgbm as lgb
            booster = lgb.Booster(model_file=model_path)
        except Exception:
            booster = None
    if booster is not None and feats:
        import numpy as np
        X = np.array(feats, dtype="float32")
        s = booster.predict(X)
        scores = [float(x) for x in s]
    else:
        # backoff: 간단 가중합
        for f in feats:
            score = 0.5*f[0] + 0.2*math.log1p(f[1]) + 0.2*f[2] + 0.1*f[3] + 0.3*f[4]
            scores.append(float(score))
    out: List[Dict[str, Any]] = []
    for it, sc in sorted(zip(candidates, scores), key=lambda z: z[1], reverse=True):
        it2 = dict(it)
        it2["_lgbm"] = sc
        out.append(it2)
    return out


