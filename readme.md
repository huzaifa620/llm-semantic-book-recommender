# LLM Semantic Book Recommender

A simple, local Gradio app that recommends books using semantic search over book descriptions. It builds an embeddings index with OpenAI Embeddings and Chroma, then filters and ranks results by category and emotional tone (happy, surprising, angry, suspenseful, sad).

Works out of the box on your machine with your OpenAI API key. No external server required.

## Features

- Natural-language search for books (“a story about forgiveness”)
- Fast semantic retrieval via OpenAI Embeddings + Chroma
- Filters by category and optionally sorts by emotional tone
- Clean, responsive Gradio UI with image gallery and captions
- Fully local data files; index is created on app start

## Quickstart

Prerequisites:

- Python 3.10+ recommended
- An OpenAI API key with access to text-embedding models

1) Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

Alternative (if you have the Python launcher):

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate
```

2) Install dependencies:

```powershell
pip install -r requirements.txt
```

3) Create a .env file in the project root and add your OpenAI key:

```
OPENAI_API_KEY=sk-...your_key...
```

4) Launch the app:

```powershell
python gradio-dashboard.py
```

Gradio will print a local URL like http://127.0.0.1:7860 — open it in your browser.

Notes:
- The first run creates an in-memory vector index from `tagged_description.txt`, which calls the OpenAI Embeddings API (usage may incur costs).
- Subsequent reruns will rebuild the index unless you add persistence (see “Performance tips”).

## How it works

High level flow:

1. Load dataset: `books_with_emotions.csv` (pandas) and construct larger thumbnails when available.
2. Load tagged book descriptions from `tagged_description.txt` using LangChain’s `TextLoader`.
3. Split text into documents and embed with `OpenAIEmbeddings`.
4. Build a Chroma vector store in-memory and perform similarity search on user queries.
5. Post-filter results:
	- Category filter: exact match on `simple_categories` (or “All”)
	- Emotional tone: sort by `joy`, `surprise`, `anger`, `fear`, or `sadness`
6. Render a gallery (8×2) with cover images and concise captions in Gradio.

Key tech:
- LangChain (loaders, text splitters)
- OpenAI Embeddings
- Chroma vector DB (in-memory for this app)
- Gradio UI

## Project structure

- `gradio-dashboard.py` — Main app; builds the index and serves the Gradio UI
- `books_with_emotions.csv` — Core dataset with metadata, categories, and emotion scores
- `books_cleaned.csv`, `books_with_categories.csv`, `books.csv` — Intermediate/alternate datasets
- `tagged_description.txt` — Text corpus used for embedding and retrieval
- `cover-not-found.jpg` — Fallback cover image when thumbnails are missing
- `data-exploration.ipynb` — Data profiling and EDA
- `sentiment-analysis.ipynb` — Deriving or inspecting emotion signals
- `vector-search.ipynb` — Vector search experiments
- `requirements.txt` — Pinned environment for app + notebooks

Expected columns used by the app in `books_with_emotions.csv` include (not exhaustive):

- `isbn13`, `title`, `authors`, `description`, `thumbnail`, `simple_categories`
- emotion columns: `joy`, `surprise`, `anger`, `fear`, `sadness`

## Configuration

Environment variables (via `.env`):

- `OPENAI_API_KEY` (required) — OpenAI API key for embeddings.

Optional Gradio settings (advanced): you can modify `dashboard.launch()` in `gradio-dashboard.py` to set host/port, e.g. `dashboard.launch(server_name="0.0.0.0", server_port=7860)`.

## Usage tips

- Query naturally (e.g., “cozy mystery set in a small town”, “non-fiction about habits and motivation”).
- Choose “All” categories for broader results, then narrow down.
- Use tone sorting to emphasize the emotional feel of the recommendations.

## Performance tips (optional)

By default, the vector store is in-memory and rebuilt every run. To speed up subsequent runs:

1) Persist the index by giving Chroma a directory, e.g.:
- When creating from documents: `Chroma.from_documents(..., persist_directory="chroma")`
- When reloading: `Chroma(persist_directory="chroma", embedding_function=OpenAIEmbeddings())`

2) Cache embeddings for the source text. Since `tagged_description.txt` changes infrequently, consider persisting the Chroma store once and reusing it.

## Notebooks

The included notebooks are optional and useful for exploration:

- `data-exploration.ipynb` — Inspect and clean raw book data
- `sentiment-analysis.ipynb` — Explore/compute emotion features
- `vector-search.ipynb` — Try out different retrieval strategies

Launch Jupyter Lab if desired:

```powershell
python -m jupyterlab
```

## Troubleshooting

- Authentication error (401/403): Ensure `OPENAI_API_KEY` is set and valid; verify your org/project settings in OpenAI.
- File not found: Confirm `books_with_emotions.csv` and `tagged_description.txt` exist in the project root.
- Slow startup: The app builds embeddings/index at launch; enable persistence to avoid recomputation.
- Port in use: Change the port in `dashboard.launch(server_port=...)`.
- Windows build issues: If you hit compiler errors for optional dependencies, upgrade pip (`python -m pip install -U pip`) and retry `pip install -r requirements.txt`.

## Privacy and costs

Your query text and the contents of `tagged_description.txt` are sent to OpenAI’s embeddings API. Review OpenAI’s data usage policies and be mindful of potential costs.

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [Chroma](https://www.trychroma.com/)
- [Gradio](https://www.gradio.app/)
- [OpenAI](https://platform.openai.com/)

## Roadmap (ideas)

- Persist the vector store by default for faster cold starts
- Add adjustable weights for tone vs. semantic similarity
- Switchable embedding providers (e.g., local or Azure OpenAI)
- Add evaluation notebook for recommendation quality

