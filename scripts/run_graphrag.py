import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from herb_rag_kit.pipelines.run_method import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--method", "graphrag"] + sys.argv[1:]
    main()
