import time

import streamlit as st

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import ESGReranker
from src.retrieval.graph_retriever import ESGGraphRetriever
from src.generation.gemini_generator import GeminiAnswerGenerator


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ESG Intelligence Assistant",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# INITIALIZE RAG COMPONENTS
# Cache models so they are not reloaded on every rerun.
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading ESG retrieval models...")
def initialize_components():
    retriever = HybridRetriever()
    reranker = ESGReranker()
    generator = GeminiAnswerGenerator()
    graph_retriever = ESGGraphRetriever(hybrid=retriever)

    return retriever, reranker, generator, graph_retriever


# --------------------------------------------------
# RAG PIPELINE
# --------------------------------------------------

def answer_esg_question(question, retriever, reranker, generator, graph_retriever=None, use_graph=False):
    total_start = time.perf_counter()

    # Stage 1: Hybrid retrieval
    retrieval_start = time.perf_counter()

    active_retriever = graph_retriever if use_graph and graph_retriever else retriever
    candidates = active_retriever.search(
        question,
        top_k=20,
        candidate_k=40,
    )

    if not candidates:
        return {
            "error": "No relevant passages were retrieved.",
        }

    # Stage 2: Cross-encoder reranking
    passages = reranker.rerank(
        query=question,
        candidates=candidates,
        top_k=8,
    )

    retrieval_time = time.perf_counter() - retrieval_start

    if not passages:
        return {
            "error": "No usable passages were found.",
        }

    # Stage 3: Gemini generation
    generation_start = time.perf_counter()

    result = generator.generate_answer(
        question=question,
        passages=passages,
    )

    generation_time = time.perf_counter() - generation_start
    total_time = time.perf_counter() - total_start

    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "model": result.get("model", "Unknown"),
        "passages": passages,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "total_time": total_time,
        "retrieval_mode": "graph-assisted" if use_graph else "hybrid",
    }


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.title("🌱 ESG Intelligence")

    st.markdown(
        """
        An evidence-grounded research assistant for
        Environmental, Social, and Governance reports.

        **Pipeline**
        1. Document retrieval
        2. Semantic + keyword search
        3. Cross-encoder reranking
        4. Gemini answer generation
        5. Source citations
        """
    )

    st.divider()

    st.subheader("Technology Stack")

    st.markdown(
        """
        - **Python** — application logic
        - **FAISS** — vector similarity search
        - **BM25** — keyword retrieval
        - **Sentence Transformers** — embeddings
        - **Cross-Encoder** — passage reranking
        - **Gemini** — answer generation
        - **Streamlit** — web interface
        """
    )

    st.divider()

    st.caption(
        "Answers are generated from indexed documents. "
        "Always verify important figures against the cited sources."
    )
    use_graph = st.toggle(
        "Use graph-assisted retrieval",
        value=False,
        help="Expand hybrid-search candidates through company, year, scope, and ESG-category links.",
    )


# --------------------------------------------------
# MAIN INTERFACE
# --------------------------------------------------

st.title("🌍 ESG Intelligence Assistant")

st.markdown(
    """
    ### Ask questions about corporate sustainability

    Search sustainability reports for emissions, climate targets,
    renewable energy, environmental performance, and other ESG metrics.
    """
)

st.info(
    "This proof of concept currently uses the BASF reports "
    "indexed in your local document collection."
)


# --------------------------------------------------
# QUESTION INPUT
# --------------------------------------------------

with st.form("esg_question_form"):
    question = st.text_area(
        "Enter your ESG question",
        placeholder=(
            "Example: What is BASF's greenhouse gas emissions "
            "reduction target for 2030?"
        ),
        height=100,
    )

    submitted = st.form_submit_button(
        "🔎 Ask ESG Assistant",
        use_container_width=True,
        type="primary",
    )


# --------------------------------------------------
# EXECUTE QUERY
# --------------------------------------------------

if submitted:
    if not question.strip():
        st.warning("Please enter an ESG question.")

    else:
        try:
            retriever, reranker, generator, graph_retriever = initialize_components()

            with st.spinner(
                "Searching reports, reranking evidence, and generating an answer..."
            ):
                response = answer_esg_question(
                    question=question.strip(),
                    retriever=retriever,
                    reranker=reranker,
                    generator=generator,
                    graph_retriever=graph_retriever,
                    use_graph=use_graph,
                )

            st.session_state["last_question"] = question.strip()
            st.session_state["last_response"] = response

        except Exception as exc:
            st.error(f"Unable to process your question: {exc}")
            st.info(
                "Check your Gemini model configuration, API quota, "
                "index files, and terminal logs."
            )


# --------------------------------------------------
# DISPLAY LAST RESULT
# --------------------------------------------------

if "last_response" in st.session_state:
    response = st.session_state["last_response"]

    st.divider()

    st.subheader("Your Question")
    st.write(st.session_state["last_question"])

    if response.get("error"):
        st.warning(response["error"])

    else:
        # ANSWER
        st.subheader("📝 Answer")
        st.markdown(response["answer"])

        # SOURCE CITATIONS
        st.subheader("📚 Sources")

        citations = response.get("citations", [])

        if citations:
            for citation in citations:
                citation_id = citation.get("id", "S?")
                source_file = citation.get(
                    "source_file",
                    "Unknown source",
                )
                page = citation.get("page", "Unknown")

                with st.container(border=True):
                    st.markdown(f"**[{citation_id}]** {source_file}")
                    st.caption(f"Report page: {page}")
        else:
            st.warning("No source citations were returned.")

        # PERFORMANCE METRICS
        st.subheader("⚡ Performance")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Retrieval + Reranking",
            f"{response['retrieval_time']:.2f} sec",
        )

        col2.metric(
            "Gemini Generation",
            f"{response['generation_time']:.2f} sec",
        )

        col3.metric(
            "Total Response Time",
            f"{response['total_time']:.2f} sec",
        )

        st.caption(f"Generation model: {response['model']} · Retrieval mode: {response.get('retrieval_mode', 'hybrid')}")

        # RETRIEVED EVIDENCE
        with st.expander("🔬 Inspect retrieved evidence"):
            passages = response.get("passages", [])

            if not passages:
                st.write("No retrieved passages are available.")
            else:
                for index, passage in enumerate(passages, start=1):
                    source = passage.get(
                        "source_file",
                        passage.get("source", "Unknown source"),
                    )

                    page = passage.get(
                        "page",
                        passage.get("page_number", "Unknown"),
                    )

                    text = passage.get("text", "")

                    st.markdown(
                        f"**Passage {index} — {source}, Page {page}**"
                    )

                    st.write(text or "No passage text available.")

                    st.divider()


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "ESG Intelligence Assistant | "
    "Hybrid RAG + Cross-Encoder Reranking + Gemini"
)
