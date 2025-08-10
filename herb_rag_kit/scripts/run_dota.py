import os, sys
from herb_rag_kit.pipelines.run_method import main
if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--method", "dota"] + sys.argv[1:]
    main()
