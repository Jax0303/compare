SHELL := /bin/bash

# ---------- Config ----------
VENV ?= .venv
PYW := $(VENV)/bin/python
PIPW := $(VENV)/bin/pip

NEO4J_CONTAINER ?= neo4j-herb
NEO4J_IMAGE ?= neo4j:5
NEO4J_PASSWORD ?= Neo4j-1717!

# LLM
GEMINI_MODEL ?= models/gemini-2.5-pro

# Extract params
CONF ?= 0.6
MAX_CHUNKS ?= 8
MAX_OBJECTS_PER_SP ?= 5
MAX_PER_ENTITY ?= 200
DOCSTORE ?= indexes/txt/docstore.jsonl
STORE_DIR ?= herb_rag_kit/src/herb_rag_kit/store

.PHONY: help venv install docstore extract neo4j-up eval viz export-graph freebase-load

help:
	@echo "Targets:"
	@echo "  venv            - Create virtualenv"
	@echo "  install         - Install requirements"
	@echo "  neo4j-up        - Start Neo4j docker"
	@echo "  docstore        - Build docstore.jsonl from $(STORE_DIR)"
	@echo "  extract         - Run LLM extract pipeline"
	@echo "  freebase-load   - Load Freebase triples to Neo4j (FILE=...)"
	@echo "  eval            - Compute KG quality metrics"
	@echo "  viz             - Save HTML subgraph visualization"
	@echo "  export-graph    - Export graph as GEXF/GraphML/CSV/PNG (FMT= gexf|graphml|csv|png, OUT=path)"

venv:
	python3 -m venv $(VENV)
	$(PIPW) install --upgrade pip

install: venv
	$(PIPW) install -r requirements.txt

neo4j-up:
	@docker start $(NEO4J_CONTAINER) >/dev/null 2>&1 || \
		docker run -d --name $(NEO4J_CONTAINER) \
		  -p 7474:7474 -p 7687:7687 \
		  -e NEO4J_AUTH="neo4j/$(NEO4J_PASSWORD)" $(NEO4J_IMAGE)
	@bash -c 'until (echo > /dev/tcp/127.0.0.1/7687) >/dev/null 2>&1; do sleep 1; done'
	@echo "Neo4j is ready on bolt://localhost:7687"

docstore: | venv
	$(PYW) herb_rag_kit/scripts/convert_txt_to_docstore.py \
		--store $(STORE_DIR) \
		--out $(DOCSTORE) \
		--ext txt --ext TXT --ext md

extract: | venv
	GEMINI_MODEL=$(GEMINI_MODEL) $(PYW) herb_rag_kit/scripts/lg_pipeline.py extract \
		--docstore $(DOCSTORE) \
		--conf-threshold $(CONF) \
		--max-chunks $(MAX_CHUNKS) \
		--max-objects-per-sp $(MAX_OBJECTS_PER_SP) \
		--max-per-entity $(MAX_PER_ENTITY)

freebase-load: | venv
	@if [ -z "$(FILE)" ]; then echo "Usage: make freebase-load FILE=path [LIMIT=0 CLEAR=1]"; exit 1; fi
	@if [ "$(CLEAR)" = "1" ]; then CLEAR_FLAG=--clear-db; else CLEAR_FLAG=; fi; \
	$(PYW) herb_rag_kit/scripts/freebase_loader.py --input "$(FILE)" --limit ${LIMIT-0} $$CLEAR_FLAG

eval: | venv
	$(PYW) herb_rag_kit/scripts/kg_quality_eval.py

viz: | venv
	$(PYW) herb_rag_kit/scripts/neo4j_export_viz.py --out runs/kg_subgraph.html
	@echo "Saved: runs/kg_subgraph.html"

export-graph: | venv
	@if [ -z "$(FMT)" ] || [ -z "$(OUT)" ]; then echo "Usage: make export-graph FMT=gexf OUT=runs/kg.gexf [CY=...]"; exit 1; fi
	$(PYW) herb_rag_kit/scripts/neo4j_export_graph.py --format $(FMT) --out $(OUT) $${CY:+--cypher "$(CY)"}



