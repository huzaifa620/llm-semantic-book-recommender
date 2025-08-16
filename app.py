import os
import logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# LangChain bits (match original logic)
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

# ----- boot -----
load_dotenv(override=True)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.WARNING)

BOOKS_CSV = "books_with_emotions.csv"
TAGGED_DESC = "tagged_description.txt"

if not os.path.exists(BOOKS_CSV):
    raise FileNotFoundError(f"Missing {BOOKS_CSV}")
if not os.path.exists(TAGGED_DESC):
    raise FileNotFoundError(f"Missing {TAGGED_DESC}")

books = pd.read_csv(BOOKS_CSV)

# thumbnails exactly like your code
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# load docs & split EXACTLY like your snippet
raw_documents = TextLoader(TAGGED_DESC, encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

# in-memory vector DB exactly like your snippet
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())


# ----- core logic (unchanged semantics) -----
def retrieve_semantic_recommendations(
    query: str,
    category: str | None = None,
    tone: str | None = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(
            final_top_k
        )
    else:
        book_recs = book_recs.head(final_top_k)

    # per-tone sorting exactly like your snippet
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def format_results(df: pd.DataFrame):
    results = []
    for _, row in df.iterrows():
        description = row.get("description") or ""
        words = description.split()
        truncated_description = " ".join(words[:30]) + (
            "..." if len(words) > 30 else ""
        )

        authors_raw = row.get("authors") or ""
        parts = [a.strip() for a in authors_raw.split(";") if a.strip()]
        if len(parts) == 2:
            authors_str = f"{parts[0]} and {parts[1]}"
        elif len(parts) > 2:
            authors_str = f"{', '.join(parts[:-1])}, and {parts[-1]}"
        else:
            authors_str = authors_raw

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append(
            {
                "thumbnail": row["large_thumbnail"],
                "title": row["title"],
                "authors": authors_str,
                "caption": caption,
            }
        )
    return results


# ----- FastAPI app & routes -----
app = FastAPI(title="Semantic Book Recommender", version="1.0.0")
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
FALLBACK_IMG_URL = "/cover-not-found.jpg"


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "categories": categories,
            "tones": tones,
            "fallback_img": FALLBACK_IMG_URL,
        },
    )


@app.post("/api/recommend", response_class=JSONResponse)
async def api_recommend(payload: dict):
    query = (payload.get("query") or "").strip()
    category = payload.get("category") or "All"
    tone = payload.get("tone") or "All"
    if not query:
        return {"results": []}
    recs = retrieve_semantic_recommendations(query, category, tone)
    return {"results": format_results(recs)}
