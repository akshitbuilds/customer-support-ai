Get a free API key at [console.groq.com](https://console.groq.com).

### 4. Build the knowledge base (first-time setup only)
The FAISS index is already built and included (`backend/rag/faiss_index.pkl`). To rebuild it from the PDFs in `knowledge_base/`:
```bash
python backend/rag/rag_pipeline.py
```

### 5. Start the backend
```bash
uvicorn backend.main:app --reload --port 8000
```
Confirm it's running at `http://127.0.0.1:8000/docs`.

### 6. Start the frontend (in a second terminal)
```bash
streamlit run frontend/app.py
```
This opens the chat interface at `http://localhost:8501`.

## Sample Interactions

**Single-agent query:**
> "What is the price of the iPhone 15?" → routed to **Product** agent

**Multi-agent query:**
> "What is refund policy?" → routed to **Billing** and **FAQ** agents, responses combined

**Complaint/escalation:**
> "I am extremely disappointed with your service" → routed to **Complaint** agent, offers escalation contact

## Evaluation

Tested routing accuracy against 25 labeled queries covering all 5 agents, multi-intent cases, and edge cases. Achieved **76% exact-match accuracy**.

Found and fixed a real bug during this process: migrating the underlying model from LLaMA 3.3 to GPT-OSS-120B (a reasoning model) caused the intent parser's exact-match logic to silently fail and fall back to the default agent on the majority of queries, since reasoning tokens interfered with clean parsing. Switching to substring-based category matching resolved it.

Remaining imprecision clusters around distinguishing product-defect language (e.g., "the charger doesn't work") from technical-support intent — a direction for future improvement.

## Known Limitations

- Authentication is a lightweight session-name gate rather than full registration/login (documented as a scoping decision — see report)
- Rapid sequential message submission can occasionally cause a display race condition in the frontend (backend always processes correctly; this is a Streamlit session-state timing issue, not a backend/routing bug)
- Deployment to Render's free tier was attempted but blocked by the platform's 512MB memory limit, due to the combined footprint of the embedding model and FAISS index — see report for details
- Routing occasionally confuses product-defect complaints with pure product or complaint intent (see Evaluation above)

## Author

Akshit — 3rd year CS (AI & Data Science), M S Ramaiah Institute of Technology, Bengaluru
