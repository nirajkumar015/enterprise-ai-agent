
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Enterprise AI Agent",
    page_icon="🤖",
    layout="wide",
)

st.title("Enterprise AI Knowledge & Support Agent")
st.caption("RAG + SQL Analytics + ML Ticket Analysis")

page = st.sidebar.radio(
    "Choose a feature",
    ["AI Knowledge Chat", "Business Analytics", "Ticket Analyzer"],
)


def call_api(question):
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={"question": question},
            timeout=90,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as error:
        st.error(f"API request failed: {error}")
        return None


if page == "AI Knowledge Chat":
    st.header("AI Knowledge Chat")

    question = st.text_area(
        "Ask a company question",
        placeholder="What is the refund policy?",
    )

    if st.button("Ask AI", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Generating answer..."):
                result = call_api(question.strip())

            if result:
                st.write(result.get("answer", "No answer returned."))
                st.caption(f"Route: {result.get('route', 'Unknown')}")

                sources = result.get("sources", [])
                if sources:
                    st.subheader("Sources")
                    for source in sources:
                        if isinstance(source, dict):
                            st.markdown(
                                f"**{source.get('source', 'Document')}**"
                            )
                            if source.get("text"):
                                st.write(source["text"])
                            if source.get("distance") is not None:
                                st.caption(
                                    f"Retrieval distance: "
                                    f"{source['distance']:.4f}"
                                )
                        else:
                            st.write(source)


elif page == "Business Analytics":
    st.header("Business Analytics")

    reports = {
        "Total customers": "How many customers are there?",
        "Total orders": "How many orders are there?",
        "Total products": "How many products are there?",
        "Total tickets": "How many tickets are there?",
        "Open tickets": "How many open tickets are there?",
    }

    selected = st.selectbox("Choose a report", list(reports.keys()))

    if st.button("Run Report", type="primary"):
        with st.spinner("Fetching analytics..."):
            result = call_api(reports[selected])

        if result:
            st.metric(selected, result.get("answer", "No result"))

            if result.get("query"):
                with st.expander("View SQL"):
                    st.code(result["query"], language="sql")


elif page == "Ticket Analyzer":
    st.header("Support Ticket Analyzer")

    ticket = st.text_area(
        "Ticket description",
        placeholder="My laptop is broken and I need urgent support.",
    )

    if st.button("Analyze Ticket", type="primary"):
        if not ticket.strip():
            st.warning("Enter a ticket description first.")
        else:
            with st.spinner("Analyzing ticket..."):
                result = call_api(
                    f"Analyze this ticket: {ticket.strip()}"
                )

            if result:
                analysis = result.get("ticket_analysis")

                if analysis:
                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Sentiment",
                        analysis.get("sentiment", "Unknown").title(),
                    )
                    col2.metric(
                        "Category",
                        analysis.get("category", "Unknown").title(),
                    )
                    col3.metric(
                        "Priority",
                        analysis.get("predicted_priority", "Unknown"),
                    )

                    confidence = analysis.get("priority_confidence")
                    if confidence is not None:
                        st.write(f"Model confidence: {confidence:.2%}")

                    probabilities = analysis.get(
                        "priority_probabilities", {}
                    )

                    if probabilities:
                        st.subheader("Priority probabilities")

                        # Keep the app usable if the chart dependency fails.
                        try:
                            st.bar_chart(probabilities)
                        except (ImportError, ValueError, TypeError) as error:
                            st.info(
                                "The probability chart could not be rendered. "
                                "The raw probabilities are shown below."
                            )
                            st.json(probabilities)
                            st.caption(f"Chart error: {error}")

                    st.info(
                        "Model predictions are estimates, not guarantees."
                    )
                else:
                    st.warning(
                        "Ticket analysis was not returned by the API."
                    )


st.divider()
st.caption("FastAPI | PostgreSQL | RAG | FAISS | Machine Learning")
