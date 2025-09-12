import networkx as nx


def main() -> None:
    G = nx.DiGraph()
    G.add_node("A", type="Entity")
    G.add_node("B", type="Entity")
    G.add_edge("A", "B", predicate="relates")
    print({
        "nodes": list(G.nodes(data=True)),
        "edges": list(G.edges(data=True)),
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
    })


if __name__ == "__main__":
    main()



