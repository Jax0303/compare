from __future__ import annotations
from typing import Dict, Any, List, Tuple, Set
import networkx as nx
import re

class GraphRAG:
    def __init__(self):
        self.G = nx.Graph()
        self.doc_map: Dict[str, str] = {}  # doc_id -> text

    def add_document(self, doc_id: str, text: str):
        self.doc_map[doc_id] = text
        ents = self._extract_entities(text)
        # connect consecutive entities
        for i, e in enumerate(ents):
            self.G.add_node(e)
            if i > 0:
                self.G.add_edge(ents[i-1], e, doc=doc_id)

    def _extract_entities(self, text: str) -> List[str]:
        # simple capitalized words/phrases as entity proxies
        toks = re.findall(r"\b([A-Z][a-zA-Z0-9_-]+(?:\s+[A-Z][a-zA-Z0-9_-]+)*)\b", text)
        return list(dict.fromkeys(toks))[:50]

    def retrieve(self, question: str, hops: int = 2, k: int = 10) -> List[str]:
        q_ents = self._extract_entities(question)
        if not q_ents:
            return list(self.doc_map.keys())[:k]
        reached_docs: List[str] = []
        seen: Set[str] = set()
        for qe in q_ents:
            if qe not in self.G: 
                continue
            # BFS over entity graph; collect edge docs
            for nbr in nx.single_source_shortest_path_length(self.G, qe, cutoff=hops):
                pass
            for u, v, data in self.G.edges(data=True):
                if u == qe or v == qe:
                    doc_id = data.get("doc")
                    if doc_id and doc_id in self.doc_map and doc_id not in seen:
                        seen.add(doc_id)
                        reached_docs.append(doc_id)
                        if len(reached_docs) >= k:
                            return reached_docs
        return reached_docs[:k]
