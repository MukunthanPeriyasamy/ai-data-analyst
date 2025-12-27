# AI Data Analyst

**Project Overview:**
- **Description:**: A small project demonstrating data preprocessing, simple modeling, and vector-indexing of tabular data for semantic retrieval and QA over dataset content.
- **Purpose:**: Provide utilities to clean datasets, handle missing values and outliers, convert CSVs to document chunks, and build a vector index for semantic retrieval.

**Quick Start:**
- **Python:**: Use Python 3.8+.
- **Install dependencies:**:

```bash
pip install -r requirements.txt
```

- **Run the main script:**:

```bash
python src/data_chat.py --> This is to chat with the model

or

python src/agent.py --> This is to process the dataset
```

**Environment / Tokens:**
- **HF_TOKEN:**: If you use Hugging Face models, set `HF_TOKEN` in your environment to access private/large models.
- **MILVUS_URI:**: If you want to use Milvus as the vector store, set `MILVUS_URI` to a network URI such as `tcp://<host>:19530`. If `MILVUS_URI` is a filesystem path or missing, the code falls back to a local FAISS index.

**Project Structure:**
- **Files:**
- **Main script**: [main.py](main.py) — project entry point and demo runner.
- **Requirements**: [requirements.txt](requirements.txt) — Python dependencies.
- **Dataset**: [data/dataset.csv](data/dataset.csv) — sample dataset used in the examples.
- **Source code**: [src/](src/) — core modules (agent, dataset utilities, model wrappers, server, prompts).
- **Vector DB**: [vector_db/index.faiss](vector_db/index.faiss) — local FAISS index used by default.

**How It Works (high level):**
- **Data prep:** scripts in `src/` generate and clean the dataset (missing-value handling and outlier handling).
- **Conversion:** CSVs are converted to document chunks suitable for embedding.
- **Embedding & indexing:** Embeddings are computed (default uses a Hugging Face sentence-transformer) and stored in a vector store. By default the project uses FAISS for local experiments; optional Milvus support is available if you provide a valid network `MILVUS_URI`.

**Usage Notes & Troubleshooting:**
- **Invalid Milvus URI:**: Milvus requires a network address (for example `tcp://host:19530`). If the configured `MILVUS_URI` is a filesystem path (e.g., a temp folder) the code may log a warning and fall back to building a local FAISS index. To use Milvus, run a Milvus server and set `MILVUS_URI` accordingly.
- **Missing packages / import errors:**: Ensure you installed `requirements.txt` in the active environment. For optional vectorstores (Milvus, others) you may need extra packages.

**Development / Extending:**
- **Add models:** update `EMBED_MODEL_ID` or `GEN_MODEL_ID` in your environment or configuration if you want different embedding/generation models.
- **Vector store:** to test Milvus integration, deploy a Milvus server and set `MILVUS_URI` to its network address. For local experiments, FAISS requires no external server.

![WhatsApp Image 2025-12-27 at 12 39 36 PM](https://github.com/user-attachments/assets/e5b8c527-2da0-4170-ae7d-cb74a6a657f9)

![WhatsApp Image 2025-12-27 at 12 42 49 PM](https://github.com/user-attachments/assets/619b6faa-80c0-409b-bb5a-0e148cb1ee62)

![WhatsApp Image 2025-12-27 at 12 43 16 PM](https://github.com/user-attachments/assets/b8268171-61f0-4bf0-bcff-a08ff79d2bb3)

  
