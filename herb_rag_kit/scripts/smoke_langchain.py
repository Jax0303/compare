from langchain_core.runnables import RunnableLambda


def strip_text(inp: dict) -> dict:
    return {"text": (inp.get("text") or "").strip()}


def to_length(inp: dict) -> dict:
    txt = inp.get("text") or ""
    return {"length": len(txt)}


def main() -> None:
    chain = RunnableLambda(strip_text) | RunnableLambda(to_length)
    print(chain.invoke({"text": "  hello langchain  "}))


if __name__ == "__main__":
    main()



