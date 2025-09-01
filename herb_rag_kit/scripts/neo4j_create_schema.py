from herb_rag_kit.graphdb.neo4j_store import Neo4jStore

if __name__ == "__main__":
    store = Neo4jStore()
    store.create_schema()
    print("[OK] Neo4j schema created (constraints, vector index, fulltext index).")
