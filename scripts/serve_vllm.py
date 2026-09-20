"""Compatibility entry point for the production vLLM server."""
from serve import main
if __name__=="__main__":
    raise SystemExit(main())
