import pandas as pd
import streamlit as st

from Lab8 import evaluate_models, load_dataset


st.set_page_config(
    page_title="Water Quality AI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .hero {
        padding: 1.5rem 1.75rem;
        border-radius: 1rem;
        background: linear-gradient(135deg, #0b5ed7, #11a6c8);
        color: white;
        margin-bottom: 1.25rem;
    }
    .hero h1 { margin: 0; font-size: 2.2rem; }
    .hero p { margin: 0.4rem 0 0; opacity: 0.92; }
    .result-good {
        padding: 1rem;
        border-radius: 0.75rem;
        background: #e8f7ee;
        border: 1px solid #a9dfbd;
        color: #176b3a;
        font-size: 1.15rem;
        font-weight: 600;
    }
    .result-bad {
        padding: 1rem;
        border-radius: 0.75rem;
        background: #fff0f0;
        border: 1px solid #f0b5b5;
        color: #9b2525;
        font-size: 1.15rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>💧 Water Quality AI</h1>
        <p>วิเคราะห์คุณภาพน้ำและทำนายว่าน้ำเหมาะสำหรับดื่มหรือไม่</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.spinner("กำลังเตรียมข้อมูลและประเมินโมเดล..."):
    dashboard = evaluate_models()
    data = load_dataset()

st.sidebar.header("เกี่ยวกับระบบ")
st.sidebar.write(
    "ระบบนี้ใช้คุณสมบัติทางกายภาพและเคมีของน้ำ "
    "เพื่อจำแนก Potable (ดื่มได้) หรือ Not Potable (ไม่ควรดื่ม)"
)
st.sidebar.info(
    f"โมเดลที่เลือก: {dashboard['best_model_name']}\n\n"
    f"Accuracy: {dashboard['results'][0]['Accuracy']:.2f}%"
)

overview_tab, models_tab, prediction_tab, process_tab = st.tabs(
    ["ภาพรวมข้อมูล", "เปรียบเทียบโมเดล", "ทำนายคุณภาพน้ำ", "ขั้นตอนการทำงาน"]
)

with overview_tab:
    metric_columns = st.columns(4)
    metric_columns[0].metric("จำนวนตัวอย่าง", f"{dashboard['rows']:,}")
    metric_columns[1].metric("จำนวน Features", dashboard["features"])
    metric_columns[2].metric("ข้อมูล Train", f"{dashboard['train_rows']:,}")
    metric_columns[3].metric("ข้อมูล Test", f"{dashboard['test_rows']:,}")

    overview_column, distribution_column = st.columns(2)
    with overview_column:
        st.subheader("รายละเอียด Dataset")
        st.write(f"**Target:** `{dashboard['target']}`")
        st.write("**สัดส่วน Train / Test:** 70% / 30%")
        st.write("**Features ที่ใช้:**")
        st.dataframe(
            pd.DataFrame({"Feature": dashboard["columns_list"]}),
            width="stretch",
            hide_index=True,
        )

    with distribution_column:
        st.subheader("การกระจายของกลุ่มเป้าหมาย")
        distribution = dashboard["class_distribution"]
        distribution_table = pd.DataFrame(
            {
                "Class": list(distribution.keys()),
                "ความหมาย": [
                    "ไม่เหมาะสำหรับดื่ม" if key == 0 else "เหมาะสำหรับดื่ม"
                    for key in distribution.keys()
                ],
                "จำนวน": list(distribution.values()),
            }
        )
        st.dataframe(distribution_table, width="stretch", hide_index=True)
        st.bar_chart(distribution_table.set_index("ความหมาย")["จำนวน"])

with models_tab:
    st.subheader("ผลการเปรียบเทียบโมเดล")
    st.caption("จัดเรียงจาก Accuracy สูงสุด และเลือกโมเดลอันดับแรกสำหรับการทำนาย")
    results = pd.DataFrame(dashboard["results"])
    st.dataframe(results, width="stretch", hide_index=True)
    st.bar_chart(results.set_index("Method")[["Accuracy", "Precision", "Recall", "F Measure"]])
    st.success(
        f"โมเดลที่ดีที่สุด: {dashboard['best_model_name']} "
        f"(Accuracy {results.iloc[0]['Accuracy']:.2f}%)"
    )

with prediction_tab:
    st.subheader("ทำนายคุณภาพน้ำจากค่าที่กรอก")
    st.write(
        f"ระบบจะใช้ **{dashboard['best_model_name']}** ซึ่งเป็นโมเดลที่มี "
        "Accuracy สูงสุดจากการทดสอบ"
    )

    input_values = {}
    input_columns = st.columns(3)
    for index, feature in enumerate(dashboard["columns_list"]):
        default_value = float(data[feature].median())
        with input_columns[index % 3]:
            input_values[feature] = st.number_input(
                feature,
                value=default_value,
                step=0.01,
                format="%.4f",
                help=f"ค่าเริ่มต้นคือ median ของ {feature} ใน dataset",
            )

    if st.button("🔍 วิเคราะห์คุณภาพน้ำ", type="primary", width="stretch"):
        input_frame = pd.DataFrame([input_values], columns=dashboard["columns_list"])
        scaled_input = dashboard["_scaler"].transform(input_frame)
        prediction = int(dashboard["_best_model"].predict(scaled_input)[0])

        if prediction == 1:
            st.markdown(
                "✅ ผลการทำนาย: น้ำมีแนวโน้ม **เหมาะสำหรับดื่ม (Potable)**",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "⚠️ ผลการทำนาย: น้ำมีแนวโน้ม **ไม่เหมาะสำหรับดื่ม (Not Potable)**",
                unsafe_allow_html=True,
            )

        if hasattr(dashboard["_best_model"], "predict_proba"):
            probability = dashboard["_best_model"].predict_proba(scaled_input)[0]
            st.write(
                f"ความมั่นใจของผลลัพธ์: "
                f"{float(probability[prediction]) * 100:.2f}%"
            )
        st.warning(
            "ผลนี้เป็นการคาดการณ์จาก Machine Learning "
            "ไม่ใช่ผลตรวจรับรองคุณภาพน้ำทางห้องปฏิบัติการ"
        )

with process_tab:
    st.subheader("ขั้นตอนการทำงาน")
    for step in dashboard["steps"]:
        with st.container(border=True):
            st.markdown(f"**{step['step']}**")
            st.caption(step["detail"])
