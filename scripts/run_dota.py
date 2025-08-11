import sys
from pathlib import Path

# Ensure the package is importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from herb_rag_kit.pipelines.run_method import main

if __name__ == "__main__":
    # Inject the fixed method argument and delegate to the shared runner
    sys.argv = [sys.argv[0], "--method", "dota"] + sys.argv[1:]
    main()
