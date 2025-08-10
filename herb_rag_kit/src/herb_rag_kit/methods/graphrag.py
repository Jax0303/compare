from __future__ import annotations
from typing import Dict, Any, List, Set
import networkx as nx
import re

class GraphRAG:
    def __init__(self):
        self.G = nx.Graph()
        self.doc_map: Dict[str, str] = {}

    def add_document(self, doc_id: str, text: str):
        self.doc_map[doc_id] = text
        ents = self._extract_entities(text or "")
        for i, e in enumerate(ents):
            self.G.add_node(e)
            if i > 0:
                self.G.add_edge(ents[i-1], e, doc=doc_id)

    def _extract_entities(self, text: str) -> List[str]:
        toks = re.findall(r"\b([A-Z][A-Za-z0-9_-]+(?:\s+[A-Z][A-Za-z0-9_-]+)*)\b", text)
        # 유사 중복 제거 후 상위 50개
        out, seen = [], set()
        for t in toks:
            if t not in seen:
                seen.add(t); out.append(t)
            if len(out) >= 50: break
        return out

    def retrieve(self, question: str, hops: int = 2, k: int = 10) -> List[str]:
        q_ents = self._extract_entities(question or "")
        if not q_ents:
            return list(self.doc_map.keys())[:k]
        reached_docs: List[str] = []
        seen: Set[str] = set()
        for qe in q_ents:
            if qe not in self.G:
                continue
            nodes = list(nx.single_source_shortest_path_length(self.G, qe, cutoff=hops).keys())
            sub = self.G.subgraph(nodes)
            for _, _, data in sub.edges(data=True):
                did = data.get("doc")
                if did and did in self.doc_map and did not in seen:
                    seen.add(did); reached_docs.append(did)
                    if len(reached_docs) >= k:
                        return reached_docs
        return reached_docs[:k]
