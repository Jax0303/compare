#!/usr/bin/env python3
from __future__ import annotations
import os, json, time, random, argparse, subprocess, sys, pathlib
from tqdm import tqdm

def run_query(q: str, k: int, out_path: str, explain: bool = True) -> dict:
    env = os.environ.copy()
    cmd = [sys.executable, 'herb_rag_kit/scripts/lg_pipeline.py', 'query', '--q', q, '--k', str(k), '--out', out_path]
    if explain:
        cmd.append('--explain')
    t0 = time.time()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    dt = int((time.time() - t0) * 1000)
    ok = (p.returncode == 0)
    try:
        obj = json.loads(p.stdout) if p.stdout.strip().startswith('{') else {}
    except Exception:
        obj = {}
    return {
        'ok': ok,
        'latency_ms': dt,
        'stdout': obj,
        'stderr': p.stderr[-2000:],
    }

def main() -> None:
    ap = argparse.ArgumentParser(description='Run N natural language queries and log metrics')
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--out', default='runs/session.jsonl')
    ap.add_argument('--log', default='runs/bench_results.jsonl')
    ap.add_argument('--summary', default='runs/bench_summary.json', help='write summary JSON here')
    ap.add_argument('--no-progress', action='store_true', help='disable tqdm progress bar')
    args = ap.parse_args()

    # RELATES-safe query pool to avoid Document mode
    queries = [
        '그래프의 RELATES 기준 상위 predicate 5개',
        '그래프에서 outdegree 기준 상위 엔티티 5개',
        'predicate film_film_genre 상위 5쌍',
        'predicate people_person_profession 상위 5쌍',
        'predicate award_award_nominee_award_nominations__award_award_nomination_award 상위 5쌍',
    ]
    # fill to args.n by sampling
    pool = queries if len(queries) >= 1 else ['상위 predicate 5개']
    chosen = [random.choice(pool) for _ in range(args.n)]

    os.makedirs(os.path.dirname(args.log) or '.', exist_ok=True)
    it = enumerate(chosen, 1)
    bar = None if args.no_progress else tqdm(total=args.n, ncols=80, desc='bench')
    all_rows = []
    for i, q in it:
        rec = {'i': i, 'q': q}
        res = run_query(q, args.k, args.out, explain=True)
        rec.update({'ok': res['ok'], 'latency_ms': res['latency_ms']})
        if isinstance(res['stdout'], dict):
            rec['rows'] = len(res['stdout'].get('graph_rows', [])) + len(res['stdout'].get('hybrid_rows', []))
            rec['cypher'] = res['stdout'].get('cypher')
            rec['consistent'] = res['stdout'].get('consistent')
            rec['confidence'] = res['stdout'].get('confidence')
            rec['entities'] = len(res['stdout'].get('entities', []))
            rec['paths'] = len(res['stdout'].get('paths', []))
        with open(args.log, 'a', encoding='utf-8') as wf:
            wf.write(json.dumps(rec, ensure_ascii=False) + '\n')
        all_rows.append(rec)
        if bar is not None:
            bar.set_postfix(ok=rec['ok'], ms=rec['latency_ms'], rows=rec.get('rows', 0))
            bar.update(1)
        else:
            print(f"[{i}/{args.n}] ok={rec['ok']} latency={rec['latency_ms']}ms rows={rec.get('rows')}")

    # summary
    if all_rows:
        import statistics as st
        n = len(all_rows)
        oks = [1 if r.get('ok') else 0 for r in all_rows]
        lat = [r.get('latency_ms', 0) for r in all_rows]
        rowsv = [r.get('rows', 0) for r in all_rows]
        summary = {
            'n': n,
            'ok_rate': round(sum(oks)/n, 3),
            'latency_ms': {
                'avg': round(sum(lat)/n, 1),
                'p50': int(st.median(lat)),
                'p95': int(st.quantiles(lat, n=20)[18]) if n >= 20 else None,
            },
            'rows': {
                'avg': round(sum(rowsv)/n, 2),
                'p50': int(st.median(rowsv)),
            },
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.summary:
            pathlib.Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    # default STRICT_GRAPH_MODE for stability
    os.environ.setdefault('STRICT_GRAPH_MODE', '1')
    main()


