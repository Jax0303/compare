from langgraph.graph import StateGraph, START, END


def incr(state: dict) -> dict:
    state["x"] = state.get("x", 0) + 1
    return state


def main() -> None:
    g = StateGraph(dict)
    g.add_node("incr", incr)
    g.add_edge(START, "incr")
    g.add_edge("incr", END)
    app = g.compile()
    out = app.invoke({"x": 0})
    print({"x": out.get("x")})


if __name__ == "__main__":
    main()



