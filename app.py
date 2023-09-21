"""
Streamlit app
"""
import json

import altair as alt
import pandas as pd
import streamlit as st
from streamlit import components

st.set_page_config(layout="wide")


@st.cache_data
def load_csv(filename):
    df = pd.read_csv(filename)
    return df


@st.cache_data
def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


@st.cache_data
def load_html(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


def main():
    st.sidebar.title("Topic Modelling")

    df = load_csv("./results/employee_feedback.csv")
    topics_keywords = load_json("./results/topics_keywords.json")
    df["topic"] = df["topic"].astype(str)
    topics = sorted(list(set(df["topic"])))

    html_string = load_html("./results/pyldavis_vis.html")

    st.write("#### Topics by LDA")
    components.v1.html(html_string, width=1300, height=900, scrolling=True)

    st.write("#### Topic keywords")
    for i, keywords in topics_keywords:
        st.success(f"**{i}**: {keywords}")

    st.write("---")

    c0, c1 = st.columns((1, 2))
    with c0:
        dept = st.selectbox("Dept", ["All", "Dept A", "Dept B", "Dept C", "Dept D"])
        topic = st.selectbox("Topic", topics)

    if dept == "All":
        subset_df = df
    else:
        subset_df = df.query(f"department == '{dept}'")
    counts = pd.DataFrame(subset_df["topic"].value_counts()).reset_index()

    with c1:
        st.write(f"#### Topic breakdown for `{dept}`")
        st.altair_chart(
            alt.Chart(counts)
            .mark_bar()
            .encode(
                x=alt.X("topic", title="Topic"),
                y="count",
            ),
        )

    st.success(f"**{topic}**: {topics_keywords[int(topic)][1]}")
    st.write(f"#### Feedback from `{dept}` which are classified as `Topic {topic}`")
    feedback = (
        subset_df.query(f"topic == '{topic}'")[["employee_feedback", "prob"]]
        .sort_values("prob", ascending=False)
        .values
    )
    for x, p in feedback:
        st.info(f"**{p:.4f}**: {x}")


if __name__ == "__main__":
    main()
