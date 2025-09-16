from __future__ import annotations
import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Neo4jConfig:
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "neo4j")
    database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    emb_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))

class Neo4jStore:
    def __init__(self, cfg: Optional[Neo4jConfig] = None):
        self.cfg = cfg or Neo4jConfig()
        self.driver = GraphDatabase.driver(self.cfg.uri, auth=(self.cfg.user, self.cfg.password))

    def close(self):
        try:
            self.driver.close()
        except Exception:
            pass

    # ---------- schema (Community-safe) ----------
    def create_schema(self):
        statements = [
            # Document id unique
            "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            # Entity key unique (key = name+'|'+type)
            "CREATE CONSTRAINT ent_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE",
            # Vector index (Neo4j 5.x)
            """
            CREATE VECTOR INDEX doc_embedding IF NOT EXISTS
            FOR (d:Document) ON (d.embedding)
            OPTIONS { indexConfig: { `vector.dimensions`: $dim, `vector.similarity_function`: 'cosine' } }
            """,
            # Full-text index (5.x 권장 구문)
            "CREATE FULLTEXT INDEX idx_doc_fulltext IF NOT EXISTS FOR (n:Document) ON EACH [n.title, n.text]"
        ]
        with self.driver.session(database=self.cfg.database) as s:
            for st in statements:
                s.run(st, dim=self.cfg.emb_dim)

    # ---------- upserts ----------
    def upsert_document(self, doc_id: str, text: str, title: Optional[str] = None,
                        url: Optional[str] = None, embedding: Optional[List[float]] = None):
        with self.driver.session(database=self.cfg.database) as sess:
            sess.run(
                """
                MERGE (d:Document {id:$id})
                ON CREATE SET d.title=$title, d.text=$text, d.url=$url, d.createdAt=timestamp()
                ON MATCH  SET d.title=coalesce($title,d.title), d.text=coalesce($text,d.text), d.url=coalesce($url,d.url)
                """,
                id=doc_id, title=title, text=text, url=url
            )
            if embedding is not None:
                sess.run("MATCH (d:Document {id:$id}) SET d.embedding=$emb", id=doc_id, emb=embedding)

    def upsert_triple(self, s_name: str, predicate: str, o_name: str,
                      s_type: str = "THING", o_type: str = "THING",
                      doc_id: Optional[str] = None, conf: float = 1.0):
        s_key = f"{s_name}|{s_type}"
        o_key = f"{o_name}|{o_type}"
        with self.driver.session(database=self.cfg.database) as sess:
            sess.run(
                """
                MERGE (s:Entity {name:$s_name, type:$s_type})
                ON CREATE SET s.key=$s_key
                SET s.key = coalesce(s.key, $s_key)
                MERGE (o:Entity {name:$o_name, type:$o_type})
                ON CREATE SET o.key=$o_key
                SET o.key = coalesce(o.key, $o_key)
                MERGE (s)-[r:RELATES {predicate:$p}]->(o)
                ON CREATE SET r.firstSeen=timestamp(), r.confidence=$conf
                ON MATCH  SET r.confidence=max(coalesce(r.confidence,0.0), $conf)
                """,
                s_name=s_name, s_type=s_type, s_key=s_key,
                o_name=o_name, o_type=o_type, o_key=o_key,
                p=predicate, conf=float(conf)
            )
            if doc_id:
                sess.run(
                    """
                    MATCH (d:Document {id:$doc_id})
                    MATCH (s:Entity {name:$s_name, type:$s_type})
                    MATCH (o:Entity {name:$o_name, type:$o_type})
                    MERGE (s)-[:APPEARS_IN]->(d)
                    MERGE (o)-[:APPEARS_IN]->(d)
                    """,
                    doc_id=doc_id, s_name=s_name, s_type=s_type, o_name=o_name, o_type=o_type
                )

    # ---------- queries ----------
    def run_cypher(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        with self.driver.session(database=self.cfg.database) as s:
            res = s.run(cypher, params or {})
            return [r.data() for r in res]

    def fulltext(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        cy = """
        CALL db.index.fulltext.queryNodes('idx_doc_fulltext', $q)
        YIELD node, score
        RETURN node.id AS id, node.title AS title, score
        LIMIT $k
        """
        return self.run_cypher(cy, {"q": query, "k": k})

    def knn(self, qvec: List[float], k: int = 5) -> List[Dict[str, Any]]:
        cy = """
        CALL db.index.vector.queryNodes('doc_embedding', $k, $q)
        YIELD node, score
        RETURN node.id AS id, node.title AS title, score
        """
        return self.run_cypher(cy, {"k": k, "q": qvec})

    # --------- PPR / PageRank utils (optional) ---------
    def ppr(self, source_name: str, k: int = 10) -> List[Dict[str, Any]]:
        cy = """
        MATCH (s:Entity {name:$name})
        CALL gds.pageRank.stream({
          nodeProjection: 'Entity',
          relationshipProjection: {RELATES: {type: 'RELATES', orientation: 'NATURAL'}},
          maxIterations: 20, dampingFactor: 0.85, sourceNodes: [s]
        }) YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS n, score
        RETURN n.name AS name, score
        ORDER BY score DESC LIMIT $k
        """
        try:
            return self.run_cypher(cy, {"name": source_name, "k": k})
        except Exception:
            return []

    # ---------- stats ----------
    def graph_size(self) -> Dict[str,int]:
        q = [
            ("docs", "MATCH (d:Document) RETURN count(d) AS n"),
            ("entities", "MATCH (e:Entity) RETURN count(e) AS n"),
            ("relations", "MATCH ()-[r:RELATES]->() RETURN count(r) AS n"),
        ]
        out = {}
        with self.driver.session(database=self.cfg.database) as s:
            for k, cy in q:
                out[k] = s.run(cy).single()["n"]
        return out
