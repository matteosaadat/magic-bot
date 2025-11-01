# run_dev.py
"""
Local development launcher for FastAPI.
Equivalent to: `uvicorn app:app --reload --host 0.0.0.0 --port 8000`
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
