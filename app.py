import streamlit as st
import os
import re
import html
from html.parser import HTMLParser
from groq import Groq
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.set_page_config(
    page_title="Financial Report Analyzer",
    page_icon="📈",
    layout="wide",
)

DATA_DIR = "/home/runner/workspace/attached_assets/data_fixed/text"
COMPANIES = {
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "GOOGL": "Google (Alphabet)",
    "META": "Meta Platforms",
}
CHROMA_PATH = ".chroma_db"


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []

    def handle_data(self, data):
        stripped = data.strip()
        if stripped:
            self.text.append(stripped)


def extract_text(filepath):
    with open(filepath, "r", errors="replace") as f:
        content = f.read()
    content = html.unescape(content)
    parser = TextExtractor()
    parser.feed(content)
    text = " ".join(parser.text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text, chunk_size=800, overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


@st.cache_resource(show_spinner="Loading documents and building index...")
def build_vector_store():
    client_groq = Groq(api_key=os.environ["GROQ_API_KEY"])
    embed_fn = DefaultEmbeddingFunction()
    chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
    collection = chroma_client.get_or_create_collection(
        "financial_reports",
        embedding_function=embed_fn,
    )

    company_texts = {}
    for ticker, name in COMPANIES.items():
        filepath = os.path.join(DATA_DIR, f"{ticker}_2024_item7.txt")
        if os.path.exists(filepath):
            text = extract_text(filepath)
            company_texts[ticker] = text

    all_docs = []
    all_ids = []
    all_metas = []

    for ticker, text in company_texts.items():
        if len(text) < 200:
            continue
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_docs.append(chunk)
            all_ids.append(f"{ticker}_chunk_{i}")
            all_metas.append({"ticker": ticker, "company": COMPANIES[ticker], "chunk": i})

    if all_docs:
        collection.add(
            documents=all_docs,
            ids=all_ids,
            metadatas=all_metas,
        )

    return collection, company_texts, client_groq


def query_rag(collection, question, selected_tickers, n_results=5):
    where = None
    if selected_tickers and len(selected_tickers) < len(COMPANIES):
        if len(selected_tickers) == 1:
            where = {"ticker": selected_tickers[0]}
        else:
            where = {"ticker": {"$in": selected_tickers}}

    results = collection.query(
        query_texts=[question],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return results


def build_answer(client_groq, question, results):
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context_parts = []
    for doc, meta in zip(docs, metas):
        context_parts.append(f"[{meta['company']} ({meta['ticker']})]\n{doc}")
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are a financial analyst assistant. Answer the user's question using only "
        "the provided excerpts from SEC 10-K annual report Item 7 (MD&A) filings. "
        "Be specific, cite companies by name, and use numbers when available. "
        "If the context doesn't contain enough information, say so clearly."
    )
    user_prompt = f"Question: {question}\n\nContext from annual reports:\n\n{context}"

    response = client_groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return response.choices[0].message.content


def extract_financials(company_texts):
    patterns = {
        "revenue": [
            r"(?:net sales|revenues?|total revenues?)[^\d]*\$?([\d,\.]+)\s*(?:billion|million)?",
        ],
    }
    data = {}
    for ticker, text in company_texts.items():
        data[ticker] = {"ticker": ticker, "company": COMPANIES[ticker], "text_length": len(text)}
    return data


def render_overview_tab(company_texts):
    st.subheader("Document Overview")

    rows = []
    for ticker, text in company_texts.items():
        word_count = len(text.split())
        status = "Full MD&A" if word_count > 500 else "Limited excerpt"
        rows.append({
            "Ticker": ticker,
            "Company": COMPANIES[ticker],
            "Characters": f"{len(text):,}",
            "Words": f"{word_count:,}",
            "Content": status,
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Document Size Comparison")
    companies = [r["Company"] for r in rows]
    word_counts = [int(r["Words"].replace(",", "")) for r in rows]
    fig = go.Figure(go.Bar(x=companies, y=word_counts, marker_color=px.colors.qualitative.Set2[:len(rows)]))
    fig.update_layout(
        xaxis_title="Company",
        yaxis_title="Word Count",
        showlegend=False,
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Raw Text Preview")
    col1, col2 = st.columns(2)
    tickers = list(company_texts.keys())
    for i, ticker in enumerate(tickers):
        col = col1 if i % 2 == 0 else col2
        with col:
            with st.expander(f"{COMPANIES[ticker]} ({ticker})"):
                text = company_texts[ticker]
                preview = text[:1500] + ("..." if len(text) > 1500 else "")
                st.text(preview)


def render_qa_tab(collection, company_texts, client_groq):
    st.subheader("Ask Questions About the Annual Reports")

    available = [t for t, txt in company_texts.items() if len(txt.split()) > 500]
    limited = [t for t in company_texts if t not in available]

    if limited:
        st.info(
            f"Note: {', '.join(COMPANIES[t] for t in limited)} only have table-of-contents excerpts "
            "in the uploaded data, so answers will primarily draw from "
            f"{', '.join(COMPANIES[t] for t in available)}."
        )

    selected = st.multiselect(
        "Filter by company (leave empty to search all):",
        options=list(company_texts.keys()),
        format_func=lambda t: f"{COMPANIES[t]} ({t})",
        default=[],
    )

    suggested = [
        "What were the main revenue drivers?",
        "What risks did management highlight?",
        "How did operating income change year over year?",
        "What is the company's strategy for AI and cloud?",
        "What are the key business segments?",
    ]

    st.write("**Suggested questions:**")
    cols = st.columns(len(suggested))
    for i, q in enumerate(suggested):
        if cols[i].button(q, key=f"suggest_{i}", use_container_width=True):
            st.session_state["qa_input"] = q

    question = st.text_input(
        "Your question:",
        value=st.session_state.get("qa_input", ""),
        placeholder="e.g. What were the key growth drivers?",
        key="qa_question",
    )

    if st.button("Ask", type="primary", disabled=not question):
        with st.spinner("Searching documents and generating answer..."):
            try:
                results = query_rag(collection, question, selected)
                answer = build_answer(client_groq, question, results)

                st.markdown("### Answer")
                st.markdown(answer)

                with st.expander("View source passages"):
                    for doc, meta, dist in zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0],
                    ):
                        relevance = max(0, 1 - dist)
                        st.markdown(
                            f"**{meta['company']} ({meta['ticker']})** — relevance: {relevance:.0%}"
                        )
                        st.text(doc[:400] + ("..." if len(doc) > 400 else ""))
                        st.divider()
            except Exception as e:
                st.error(f"Error generating answer: {e}")


def render_compare_tab(company_texts, client_groq):
    st.subheader("Side-by-Side Company Comparison")

    available = {t: txt for t, txt in company_texts.items() if len(txt.split()) > 500}
    if len(available) < 2:
        st.warning("Need at least 2 companies with full MD&A text for comparison.")
        return

    col1, col2 = st.columns(2)
    with col1:
        ticker_a = st.selectbox(
            "Company A",
            list(available.keys()),
            format_func=lambda t: f"{COMPANIES[t]} ({t})",
            index=0,
        )
    with col2:
        remaining = [t for t in available if t != ticker_a]
        ticker_b = st.selectbox(
            "Company B",
            remaining,
            format_func=lambda t: f"{COMPANIES[t]} ({t})",
            index=0,
        )

    compare_topic = st.text_input(
        "Topic to compare:",
        value="revenue growth and business performance",
        placeholder="e.g. risks, AI strategy, profitability",
    )

    if st.button("Compare", type="primary"):
        with st.spinner("Generating comparison..."):
            try:
                def get_summary(ticker, topic):
                    text = available[ticker]
                    words = text.split()
                    excerpt = " ".join(words[:3000])
                    resp = client_groq.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a financial analyst. Summarize the following MD&A excerpt "
                                    f"focusing specifically on: {topic}. Be concise and factual, "
                                    "use bullet points, and highlight key numbers."
                                ),
                            },
                            {"role": "user", "content": excerpt},
                        ],
                        temperature=0.2,
                        max_tokens=500,
                    )
                    return resp.choices[0].message.content

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"### {COMPANIES[ticker_a]} ({ticker_a})")
                    summary_a = get_summary(ticker_a, compare_topic)
                    st.markdown(summary_a)
                with col_b:
                    st.markdown(f"### {COMPANIES[ticker_b]} ({ticker_b})")
                    summary_b = get_summary(ticker_b, compare_topic)
                    st.markdown(summary_b)

                st.divider()
                st.markdown("### Head-to-Head Analysis")
                with st.spinner("Generating head-to-head analysis..."):
                    combined = client_groq.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a senior financial analyst. Given summaries of two companies' "
                                    f"MD&A sections on '{compare_topic}', provide a concise head-to-head "
                                    "analysis highlighting key differences, similarities, and which company "
                                    "appears stronger in this area based on the information provided."
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"{COMPANIES[ticker_a]}:\n{summary_a}\n\n"
                                    f"{COMPANIES[ticker_b]}:\n{summary_b}"
                                ),
                            },
                        ],
                        temperature=0.2,
                        max_tokens=400,
                    )
                    st.markdown(combined.choices[0].message.content)

            except Exception as e:
                st.error(f"Error generating comparison: {e}")


def main():
    st.title("Financial Report Analyzer")
    st.caption("AI-powered analysis of 2024 Annual Report MD&A sections — AAPL, AMZN, GOOGL, META")

    try:
        collection, company_texts, client_groq = build_vector_store()
    except Exception as e:
        st.error(f"Failed to initialize: {e}")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Overview", "Q&A", "Compare"])

    with tab1:
        render_overview_tab(company_texts)

    with tab2:
        render_qa_tab(collection, company_texts, client_groq)

    with tab3:
        render_compare_tab(company_texts, client_groq)


if __name__ == "__main__":
    main()

