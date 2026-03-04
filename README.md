## CLI QA Assistant over Local Documents

This is a small LLM project that implements a **command-line QA assistant** over your own documents (policies, FAQs, PDFs, etc.).
It uses a **SentenceTransformer embedding model** to find the most relevant sentence in your data for each question, instead of relying on a large generative chat model.

### Project structure

- `data/` – Place your domain documents here:
  - `.txt` files (e.g., policies, FAQs, notes)
  - `.pdf` files (e.g., official policies, holiday lists)
- `embeddings/` – Auto-generated folder where embeddings and metadata are saved.
- `requirements.txt` – Python dependencies.
- `build_index.py` – Offline script to preprocess documents and build the embeddings index.
- `qa_cli.py` – CLI QA assistant that answers questions using the built index.

### How it works

- **1. Document ingestion & chunking**
  - `build_index.py` reads all `.txt` and `.pdf` files in `data/`.
  - PDFs are parsed with `PyPDF2` and converted to text.
  - Each document is split into overlapping text chunks (500 characters with 100-character overlap) to preserve context.

- **2. Embeddings & index**
  - Uses `sentence-transformers/all-MiniLM-L6-v2` to convert each chunk into a vector embedding.
  - Saves:
    - `embeddings/vectors.npy` – NumPy array of all chunk embeddings.
    - `embeddings/metadata.json` – List of chunk metadata (`id`, `source`, `text`).

- **3. Question answering (retrieval-based)**
  - `qa_cli.py` loads the saved vectors and metadata once at startup.
  - For each user question:
    - Embeds the question with the same embedding model.
    - Uses cosine similarity to retrieve the top-k most relevant document chunks.
    - Splits those chunks into individual sentences/lines.
    - Embeds all candidate sentences and picks the **single most similar sentence** to the question.
  - That sentence is returned as the answer (or `not in the context` if nothing matches well enough).

This demonstrates understanding of embeddings, vector similarity search, and a non-generative retrieval-based QA pattern grounded only in your local data.

### Setup

From the project root:

```bash
python -m venv .myenv
.\\.myenv\\Scripts\\activate
pip install -r requirements.txt
```

### Prepare your data

- Put one or more `.txt` and/or `.pdf` files into the `data/` folder.
- Examples: `data/company_policies.txt`, `data/product_faq.txt`, `data/LeavePolicy.pdf`, etc.

### Build the index

```bash
python build_index.py
```

This will create:

- `embeddings/vectors.npy`
- `embeddings/metadata.json`

### Run the CLI assistant

```bash
python qa_cli.py
```

Then type your questions in the terminal. Type `exit` / `quit` / `q` (or press `Ctrl+C`) to stop.

### IMP Concepts

- **Embedding model as LLM**: `sentence-transformers/all-MiniLM-L6-v2` is a small LLM used to create semantic embeddings of text.
- **Embeddings**: Convert text into vectors so semantically similar texts are close in vector space.
- **Vector search**: Use cosine similarity between question and chunk/sentence embeddings to find the most relevant pieces of text.
- **Retrieval-based QA (non-generative)**: Instead of generating new free-form answers, the system selects the best matching sentence from your documents, keeping answers grounded and predictable.
# CLI QA Chatbot with Hugging Face RAG
