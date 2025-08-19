# scripts/viz_inline.py
import os, re
import networkx as nx
from pyvis.network import Network

# ========= 입출력 경로 =========
GEXF_IN        = "runs/graphrag_from_index_1755508514.gexf"   # 필요 시 수정
OUT_MAIN_HTML  = "runs/graphrag_viz_inline.html"
OUT_COMM_HTML  = "runs/community_graph.html"
OUT_FILTERED_G = "runs/graphrag_filtered.gexf"
os.makedirs("runs", exist_ok=True)

# ========= 파라미터(읽기 쉬운 기본값) =========
TOPK_HUBS        = 100   # 허브(top degree) 기준 포함할 중심 노드 수
RADIUS           = 1     # 허브로부터 r-hop (2로 올리면 급격히 복잡)
BACKBONE_TOPK    = 3     # 노드당 유지할 상위 가중치 엣지 수(2~3 추천)
KCORE_K          = 2     # k-core 필터(1~3 권장, 0이면 스킵)
TARGET_MAX_NODES = 700   # 최종 노드 상한(초과 시 degree로 자동 컷)
LABEL_MIN_DEG    = 10    # 라벨 붙일 최소 차수
LABEL_TOP        = 30    # 라벨 표시 최대 개수
MAX_LABEL_LEN    = 22    # 라벨 길이 제한(말줄임표)
COMM_EDGE_W_MIN  = 2     # 메타그래프 약한 엣지 제거(커뮤니티 간 weight < 이 값 제거)
TOPM_COMM_EDGES  = 5     # 커뮤니티당 가장 강한 연결 m개만 유지(메타그래프 간선 상한)

# ========= 유틸 =========
def short(s: str, m: int = MAX_LABEL_LEN) -> str:
    s = str(s)
    return s if len(s) <= m else s[: m - 1] + "…"

def to_simple_weighted(G: nx.Graph) -> nx.Graph:
    """Multi/Directed → Simple Undirected Weighted 집계."""
    U = G.to_undirected() if G.is_directed() else G
    H = nx.Graph()
    H.add_nodes_from(U.nodes(data=True))
    for u, v, data in U.edges(data=True):
        w = float(data.get("weight", 1.0))
        if H.has_edge(u, v):
            H[u][v]["weight"] += w
        else:
            H.add_edge(u, v, weight=w)
    return H

def subgraph_by_hubs(G: nx.Graph, topk: int, radius: int) -> nx.Graph:
    hubs = sorted(G.degree, key=lambda x: x[1], reverse=True)[:topk]
    keep = set()
    for n, _ in hubs:
        keep |= set(nx.single_source_shortest_path_length(G, n, cutoff=radius).keys())
    return G.subgraph(keep).copy()

def prune_topk_edges(G: nx.Graph, k: int = 2) -> nx.Graph:
    """노드당 상위 k개의 가중치 큰 엣지만 남기는 백본 추출."""
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for u in G.nodes():
        pairs = []
        for v, data in G[u].items():
            w = float(data.get("weight", 1.0))
            pairs.append((w, v))
        pairs.sort(reverse=True)
        for w, v in pairs[:k]:
            if H.has_edge(u, v):
                H[u][v]["weight"] = max(H[u][v].get("weight", 0.0), w)
            else:
                H.add_edge(u, v, weight=w)
    H.remove_nodes_from([n for n, d in H.degree if d == 0])
    return H

def community_partition(G: nx.Graph) -> dict:
    """커뮤니티(모듈러리티) 파티션: louvain → greedy fallback."""
    try:
        import community as community_louvain  # python-louvain
        print("[INFO] community: louvain")
        return community_louvain.best_partition(G)
    except Exception:
        print("[INFO] community: greedy_modularity (fallback)")
        from networkx.algorithms.community import greedy_modularity_communities
        part = {}
        for gid, com in enumerate(greedy_modularity_communities(G)):
            for n in com:
                part[n] = gid
        return part

def compute_cluster_labels(groups: dict[int, list[str]], topk: int = 4) -> dict[int, str]:
    """커뮤니티 자동 라벨(TF-IDF 상위 키워드). scikit-learn 없으면 Fallback(빈도 상위)."""
    def norm(s: str) -> str:
        return re.sub(r"[^0-9A-Za-z가-힣·\s]+", " ", s).lower()

    docs = []
    idx2cid = []
    for cid in sorted(groups):
        docs.append(" ".join(groups[cid]))
        idx2cid.append(cid)

    labels = {}
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
        X = vec.fit_transform([norm(t) for t in docs])
        vocab = vec.get_feature_names_out()
        for i, cid in enumerate(idx2cid):
            row = X[i].toarray().ravel()
            ids = row.argsort()[-topk:][::-1]
            toks = [vocab[j] for j in ids if row[j] > 0]
            labels[cid] = " / ".join(toks) if toks else f"Cluster {cid}"
    except Exception:
        # Fallback: 공백 토큰 빈도 상위
        for i, cid in enumerate(idx2cid):
            toks = [t for t in norm(docs[i]).split() if len(t) >= 2]
            freq = {}
            for t in toks:
                freq[t] = freq.get(t, 0) + 1
            top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:topk]
            labels[cid] = " / ".join([t for t, _ in top]) if top else f"Cluster {cid}"
    return labels

# ========= 로드 =========
print("[INFO] load:", GEXF_IN)
G_raw = nx.read_gexf(GEXF_IN)
G = to_simple_weighted(G_raw)
print("[INFO] full graph:", G.number_of_nodes(), "nodes /", G.number_of_edges(), "edges")

# ========= 축소 파이프라인 (자동 완화 포함) =========
H_hubs = subgraph_by_hubs(G, TOPK_HUBS, RADIUS)
print("[INFO] subgraph:", H_hubs.number_of_nodes(), "nodes /", H_hubs.number_of_edges(), "edges")

def try_build(H_base, bbk, kc):
    H1 = prune_topk_edges(H_base, k=bbk)
    print(f"[INFO] backbone(k={bbk}):", H1.number_of_nodes(), "nodes /", H1.number_of_edges(), "edges")
    if kc >= 1 and H1.number_of_nodes() and H1.number_of_edges():
        try:
            H2 = nx.k_core(H1, k=kc)
            print(f"[INFO] k-core(k={kc}):", H2.number_of_nodes(), "nodes /", H2.number_of_edges(), "edges")
            return H2
        except nx.NetworkXError:
            pass
    return H1

candidates = [
    (BACKBONE_TOPK, KCORE_K),
    (BACKBONE_TOPK + 1, KCORE_K),
    (BACKBONE_TOPK + 2, KCORE_K),
    (BACKBONE_TOPK, max(1, KCORE_K - 1)),
    (BACKBONE_TOPK + 1, max(1, KCORE_K - 1)),
    (max(3, BACKBONE_TOPK + 2), 1),
    (max(3, BACKBONE_TOPK + 2), 0),  # k-core 스킵
]

H = None
used = None
for bbk, kc in candidates:
    H_try = try_build(H_hubs, bbk, kc)
    n = H_try.number_of_nodes()
    if n == 0:
        continue
    if n > TARGET_MAX_NODES:
        degs = sorted([d for _, d in H_try.degree()], reverse=True)
        thr = degs[int(len(degs) * 0.35)]  # 상위 65%만 남기기
        keep_nodes = [u for u, d in H_try.degree() if d >= thr]
        H_try = H_try.subgraph(keep_nodes).copy()
        print(f"[INFO] auto-trim (deg >= {thr}):", H_try.number_of_nodes(), "nodes /", H_try.number_of_edges(), "edges")
    used = (bbk, kc)
    H = H_try
    break

if H is None or H.number_of_nodes() == 0:
    raise SystemExit("[ERR] Empty graph — BACKBONE_TOPK/KCORE_K/TOPK_HUBS 값을 더 완화하세요.")

print(f"[INFO] selected: backbone_topk={used[0]}, kcore={used[1]}, "
      f"final: {H.number_of_nodes()} nodes / {H.number_of_edges()} edges")

# ========= 커뮤니티 / 라벨 =========
part = community_partition(H)           # node -> cluster id
deg  = dict(H.degree())

# 표시 라벨(최소 차수 + 상위 N개만)
candidates_nodes = [n for n, d in H.degree() if d >= LABEL_MIN_DEG]
label_nodes = set(
    n for n, _ in sorted(((n, deg[n]) for n in candidates_nodes),
                         key=lambda x: x[1], reverse=True)[:LABEL_TOP]
)

# ========= 필터된 그래프 GEXF로 저장 (Gephi용) =========
nx.write_gexf(H, OUT_FILTERED_G)
print("[OK] saved filtered GEXF:", OUT_FILTERED_G)

# ========= PyVis(메인 그래프) =========
net = Network(height="100vh", width="100%", directed=False,
              notebook=False, cdn_resources="in_line")  # 외부 CDN X
net.set_options(
    '{"physics":{"enabled":true,"solver":"forceAtlas2Based",'
    '"forceAtlas2Based":{"gravitationalConstant":-80,"centralGravity":0.015,'
    '"springLength":130,"springConstant":0.08,"avoidOverlap":1},'
    '"minVelocity":0.75,"stabilization":{"iterations":150}},'
    '"nodes":{"shape":"dot","scaling":{"min":5,"max":60}},'
    '"edges":{"smooth":false}}'
)
for n in H.nodes():
    net.add_node(
        n,
        label=short(n) if n in label_nodes else "",
        title=str(n),
        value=deg[n],                 # 노드 크기 = 차수
        group=int(part.get(n, 0)),    # 커뮤니티 색상
    )
for u, v, d in H.edges(data=True):
    net.add_edge(u, v, value=float(d.get("weight", 1)))  # 엣지 두께 = 가중치

net.write_html(OUT_MAIN_HTML)
print("[OK] saved:", OUT_MAIN_HTML, "| bytes:", os.path.getsize(OUT_MAIN_HTML))

# ========= 메타그래프(커뮤니티 간 연결) =========
CG = nx.Graph()

# 노드(커뮤니티) 집계: size = 커뮤니티 내 degree 합, count = 노드 수
for n in H.nodes():
    g = int(part.get(n, 0))
    if not CG.has_node(g):
        CG.add_node(g, size=0.0, count=0)
    CG.nodes[g]["size"]  += float(deg.get(n, 0))
    CG.nodes[g]["count"] += 1

# 커뮤니티 간 엣지 가중치 누적
for u, v, d in H.edges(data=True):
    gu, gv = int(part.get(u, 0)), int(part.get(v, 0))
    if gu == gv:
        continue
    w = float(d.get("weight", 1))
    if CG.has_edge(gu, gv):
        CG[gu][gv]["weight"] += w
    else:
        CG.add_edge(gu, gv, weight=w)

# 약한 inter-community 엣지 제거
if COMM_EDGE_W_MIN > 1:
    CG.remove_edges_from([(u, v) for u, v, d in CG.edges(data=True)
                          if float(d.get("weight", 1)) < COMM_EDGE_W_MIN])

# 커뮤니티별 가장 강한 연결 TOPM만 유지 (양방향 일관성 있게)
keep_pairs = set()
for g in list(CG.nodes()):
    nbrs = []
    for nb in CG[g]:
        w = float(CG[g][nb].get("weight", 1))
        nbrs.append((w, nb))
    nbrs.sort(reverse=True)
    for _, nb in nbrs[:TOPM_COMM_EDGES]:
        keep_pairs.add(tuple(sorted((g, nb))))
drop_edges = [e for e in CG.edges() if tuple(sorted(e)) not in keep_pairs]
CG.remove_edges_from(drop_edges)

# 커뮤니티 자동 라벨
groups = {}
for n, g in part.items():
    if n in H:  # H에 포함된 노드만
        groups.setdefault(int(g), []).append(str(n))
cid_to_label = compute_cluster_labels(groups, topk=4)

# PyVis 메타그래프
net2 = Network(height="90vh", width="100%", directed=False, notebook=False, cdn_resources="in_line")
net2.set_options(
    '{"physics":{"enabled":true,"solver":"forceAtlas2Based",'
    '"forceAtlas2Based":{"gravitationalConstant":-60,"centralGravity":0.02,'
    '"springLength":160,"springConstant":0.06,"avoidOverlap":1},'
    '"stabilization":{"iterations":120}},'
    '"nodes":{"shape":"dot","scaling":{"min":10,"max":120}},'
    '"edges":{"smooth":false}}'
)
for g_id, data in CG.nodes(data=True):
    label = cid_to_label.get(int(g_id), f"C{g_id}")
    label = f"{label} (n={int(data.get('count', 0))})"
    net2.add_node(int(g_id), label=label, title=label, value=float(data.get("size", 1)))
for u, v, d in CG.edges(data=True):
    net2.add_edge(int(u), int(v), value=float(d.get("weight", 1)))

net2.write_html(OUT_COMM_HTML)
print("[OK] saved:", OUT_COMM_HTML, "| bytes:", os.path.getsize(OUT_COMM_HTML),
      "| meta nodes/edges:", CG.number_of_nodes(), CG.number_of_edges())
