### Endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | `/resume` | Pipeline 1: Upload PDF/DOCX/ODT + optional target job |
| POST | `/resume/domains` | Pipeline 2: Upload PDF/DOCX/ODT → ISCO domain grouping |
| POST | `/text` | Pipeline 3: Comma-separated skills → suggestions |
| POST | `/text/goal` | Pipeline 4: Skills + target job → top match + target match |
| POST | `/query` | Pipeline 5: Free text query → entity results |