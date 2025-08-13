import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
import gradio as gr

load_dotenv(override=True)

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

raw_documents = TextLoader("tagged_description.txt", encoding='utf-8').load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())


def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
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


def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results


categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
    ),
    css="""
    .gradio-container {
        @apply max-w-6xl mx-auto p-4;
        font-family: 'Inter', sans-serif;
    }
    .header {
        @apply text-center mb-8;
    }
    .header h1 {
        @apply text-4xl font-bold text-indigo-700 dark:text-indigo-400 mb-2;
    }
    .header p {
        @apply text-lg text-slate-600 dark:text-slate-300;
    }
    .input-section {
        @apply bg-white dark:bg-slate-800 p-6 rounded-xl shadow-md mb-6;
    }
    .input-row {
        @apply flex flex-col md:flex-row gap-4 w-full;
    }
    .input-item {
        @apply flex-1;
    }
    .gallery-section {
        @apply bg-white dark:bg-slate-800 p-6 rounded-xl shadow-md;
    }
    .gallery-title {
        @apply text-2xl font-semibold text-slate-800 dark:text-slate-200 mb-4;
    }
    .gradio-button {
        @apply bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-3 px-6 rounded-lg transition-all duration-200;
    }
    .gradio-button:hover {
        @apply shadow-md transform -translate-y-0.5;
    }
    .gradio-textbox, .gradio-dropdown {
        @apply border border-slate-300 dark:border-slate-600 rounded-lg p-3 bg-white dark:bg-slate-700 text-slate-800 dark:text-slate-200;
    }
    .gradio-gallery {
        @apply grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4;
    }
    .gradio-gallery-item {
        @apply rounded-lg overflow-hidden shadow-sm hover:shadow-md transition-all duration-200;
    }
    .gradio-gallery-item:hover {
        @apply transform -translate-y-1;
    }
    .gradio-gallery-item img {
        @apply w-full h-48 object-cover rounded-t-lg;
    }
    .gradio-gallery-caption {
        @apply p-3 bg-slate-50 dark:bg-slate-700 text-slate-800 dark:text-slate-200 text-sm rounded-b-lg;
    }
    """,
) as dashboard:

    with gr.Column(elem_classes="gradio-container"):
        # Header
        with gr.Column(elem_classes="header"):
            gr.Markdown(
                """
            # Semantic Book Recommender
            Discover your next favorite read with AI-powered recommendations
            """
            )

        # Input section
        with gr.Column(elem_classes="input-section"):
            with gr.Row(elem_classes="input-row"):
                with gr.Column(elem_classes="input-item"):
                    user_query = gr.Textbox(
                        label="Describe what you're looking for:",
                        placeholder="e.g., A sci-fi adventure about space exploration",
                        elem_classes="gradio-textbox",
                    )
                with gr.Column(elem_classes="input-item"):
                    category_dropdown = gr.Dropdown(
                        choices=categories,
                        label="Category",
                        value="All",
                        elem_classes="gradio-dropdown",
                    )
                with gr.Column(elem_classes="input-item"):
                    tone_dropdown = gr.Dropdown(
                        choices=tones,
                        label="Emotional Tone",
                        value="All",
                        elem_classes="gradio-dropdown",
                    )
                with gr.Column(elem_classes="input-item", min_width=120):
                    submit_button = gr.Button(
                        "Find Books", elem_classes="gradio-button"
                    )

        with gr.Column(elem_classes="gallery-section"):
            gr.Markdown("## Recommendations", elem_classes="gallery-title")
            output = gr.Gallery(
                label="", columns=6, rows=2, elem_classes="gradio-gallery"
            )

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output,
    )

if __name__ == "__main__":
    dashboard.launch()
