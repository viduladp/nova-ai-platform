# NOVA AI Support & Personalization Platform

AI Engineer Assessment — Multi-Agent AI System for NOVA, a D2C Fashion & Beauty Brand.

---

## 🔗 Shareable Links

| Resource | Link |
|---|---|
| 📓 Task 1 — Prompt Engineering (Colab) | [Open Notebook](https://colab.research.google.com/drive/117LX_5tH2yzgDkJ1pJpiWyVSZLw6SgBR?usp=sharing) |
| 📓 Task 2 — MCP Server (Colab) | [Open Notebook](https://colab.research.google.com/drive/1dExWNfhPqz_CD0q-JFlBM6nG8r9HSE8z?usp=sharing) |
| 📓 Task 3 — RAG Pipeline (Colab) | [Open Notebook](https://colab.research.google.com/drive/18Hm3QJaFb1OD45wqgPGj1aB9JWZ0toMp?usp=sharing) |
| 📓 Task 4 — Fine-Tuning (Colab) | [Open Notebook](https://colab.research.google.com/drive/1pw_BrzElPZrbQT3OT3aLxB3wNVdZaDVF?usp=sharing) |
| 📓 Task 5 — Multi-Agent Platform (Colab) | [Open Notebook](https://colab.research.google.com/drive/1SmiHdC2OoOQHUeGe-fVvG6IOKeHE-2iR?usp=sharing) |
| 🤗 Fine-tuned Model (HuggingFace) | [nova-brand-voice-tinyllama](https://huggingface.co/Viddhula/nova-brand-voice-tinyllama) |
| 📊 W&B Training Dashboard | [nova-brand-voice run](https://wandb.ai/viduladp-sonata-software/nova-brand-voice) |
| 💻 GitHub Repository | [nova-ai-platform](https://github.com/viduladp/nova-ai-platform) |

---

## 📁 Repository Structure

```
nova-ai-platform/
├── README.md
├── requirements.txt
├── .env.example
├── nova_mock_db.json
├── prompts/
│   ├── system_prompt_v1.txt
│   └── system_prompt_v2.txt
├── data/
│   ├── sample_tickets.json
│   └── brand_voice_train.jsonl
├── task1_prompt_engineering.ipynb
├── task2_mcp/
│   ├── server.py
├── task2_mcp_notebook.ipynb
├── task3_rag_pipeline.ipynb
├── rag_module.py
├── task4_finetune.ipynb
├── task5_nova_platform.py
├── task5_demo.py
├── task5_notebook.ipynb
├── nova_agent_graph.png
├── evaluation_report.json
├── audit_log.jsonl
└── nova_traces.json
```

---

## 🛠️ Setup

### Requirements
- Python 3.10+
- Google Colab Free Tier
- Groq API key — [console.groq.com](https://console.groq.com)
- W&B API key — [wandb.ai](https://wandb.ai) (Task 4 only)
- HuggingFace token — [huggingface.co](https://huggingface.co) (Task 4 only)

### Environment Variables
Copy `.env` and fill in your keys:
```bash
GROQ_API_KEY=your_groq_api_key_here
WANDB_API_KEY=your_wandb_api_key_here
HF_TOKEN=your_huggingface_token_here
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📋 Tasks

### Task 1 — Prompt Engineering
COSTAR system prompt with CoT intent classification, frustration detection, escalation logic, and prompt injection defense. Tested across 8 scenarios using Groq `llama-3.3-70b-versatile`.

### Task 2 — MCP Server
FastMCP server with 5 backend tools (`get_order_status`, `initiate_return`, `get_product_info`, `get_customer_profile`, `search_faqs`). Synthetic database of 500 orders, 200 customers, 100 products generated using Faker. Full audit logging to `audit_log.jsonl`.

### Task 3 — RAG Pipeline
Hybrid search RAG using ChromaDB (dense) + BM25 (sparse) + cross-encoder re-ranking. Embeddings via `BAAI/bge-small-en-v1.5`. Evaluated with RAGAS — Faithfulness: 0.55, Answer Relevancy: 0.615. Exported as importable `rag_module.py` for Task 5.

### Task 4 — Fine-Tuning
QLoRA fine-tune of `TinyLlama-1.1B` on 200 NOVA brand voice training pairs using Unsloth on Colab T4 GPU. Training tracked with W&B. Model pushed to HuggingFace Hub.

### Task 5 — Multi-Agent Platform
LangGraph multi-agent system integrating Tasks 1–4. Six nodes: Intent Classifier → Tool Calling / RAG / Escalation → Brand Voice → Audit Logger. Human-in-the-loop escalation for frustrated customers and injection attempts. Full audit trails in `trinx_traces.json`.

---

## 📊 Evaluation

| Metric | Value |
|---|---|
| RAG Faithfulness | 0.55 |
| RAG Answer Relevancy | 0.615 |
| Tickets AI handled (Task 5 demo) | 3 / 5 (60%) |
| Escalations triggered | 2 / 5 (40%) |
| Injection attempts blocked | 1 / 1 (100%) |

---

## 📝 Notes

- All tasks run on **Google Colab Free Tier** — no paid compute required
- Task 4 requires switching to **T4 GPU** runtime in Colab
- All data is **synthetically generated** — no real customer data used
- `chroma_db/` folder is auto-generated when running Task 3 or Task 5






