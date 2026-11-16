import pandas as pd
import streamlit as st

from Lab8 import evaluate_models


st.set_page_config(
    page_title="Water Potability Dataset",
    page_icon="💧",
    layout="wide",
)

st.title("Water Potability Dataset")
st.caption("Artificial Neural Network & Model Comparison")

with st.spinner("กำลังโหลดข้อมูลและประเมินโมเดล..."):
    dashboard = evaluate_models()

metric_columns = st.columns(4)
metric_columns[0].metric("Samples", f"{dashboard['rows']:,}")
metric_columns[1].metric("Features", dashboard["features"])
metric_columns[2].metric("Train Set", f"{dashboard['train_rows']:,}")
metric_columns[3].metric("Test Set", f"{dashboard['test_rows']:,}")

overview_column, distribution_column = st.columns(2)

with overview_column:
    st.subheader("Dataset Overview")
    st.write(f"**Features:** {dashboard['features']}")
    st.write(f"**Target:** {dashboard['target']}")
    st.write("**Train / Test:** 70% / 30%")
    st.write("**Columns:**", ", ".join(dashboard["columns_list"]))

with distribution_column:
    st.subheader("Class Distribution")
    distribution = dashboard["class_distribution"]
    distribution_table = {
        "Class": list(distribution.keys()),
        "Count": list(distribution.values()),
    }
    st.dataframe(distribution_table, width="stretch", hide_index=True)

st.subheader("Model Performance Comparison")
results = dashboard["results"]
st.dataframe(results, width="stretch", hide_index=True)

chart_data = pd.DataFrame(results).set_index("Method")
st.bar_chart(chart_data[["Accuracy", "Precision", "Recall", "F Measure"]])

with st.expander("Processing Steps"):
    for step in dashboard["steps"]:
        st.markdown(f"**{step['step']}** — {step['detail']}")
