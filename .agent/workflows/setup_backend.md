---
description: Setup and run SBSCR backend server
---

1. Install required Python packages
```bash
pip install -r requirements.txt
```
// turbo
2. Run the FastAPI server using uvicorn
```bash
uvicorn serve:app --host 0.0.0.0 --port 8000
```
// turbo
