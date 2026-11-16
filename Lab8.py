# ============================================================
# 1. Import Library
# จัดการ import ไลบรารีที่ต้องใช้สำหรับการโหลดข้อมูล, การประมวลผล,
# การแบ่งข้อมูล, การสร้างโมเดล, และการประเมินผล
# ============================================================
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from flask import Flask, render_template
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# 2. สร้าง Flask App
# ตัวแปร app นี้ใช้สำหรับรันเว็บแอปและให้ route / แสดงหน้า HTML
# ============================================================
app = Flask(__name__)


# ============================================================
# 3. ฟังก์ชันโหลดข้อมูล
# อ่านไฟล์ non_nan.csv จากโฟลเดอร์เดียวกันกับไฟล์ Python
# และลบคอลัมน์แอดมินที่ไม่จำเป็น เช่น Unnamed: 0
# ============================================================
def load_dataset():
    file_path = Path(__file__).resolve().parent / "non_nan.csv"
    data = pd.read_csv(file_path)

    if "Unnamed: 0" in data.columns:
        data = data.drop("Unnamed: 0", axis=1)

    return data


# ============================================================
# 4. ฟังก์ชันประเมินผลโมเดล
# ทำหน้าที่ดังนี้:
# - โหลดข้อมูล
# - แยก Feature (X) กับ Target (y)
# - แบ่ง Train/Test เป็น 70/30
# - ทำ scaling
# - สร้างและประเมินแต่ละโมเดล
# - ส่งผลลัพธ์กลับเป็น dictionary สำหรับ template
# ============================================================
@lru_cache(maxsize=1)
def evaluate_models():
    # โหลดข้อมูลจาก CSV
    data = load_dataset()

    # แยกตัวแปรต้นทางและเป้าหมาย
    # X = ข้อมูลคุณลักษณะทั้งหมด, y = คอลัมน์ Potability
    X = data.drop("Potability", axis=1)
    y = data["Potability"]

    # ========================================================
    # 5. Train 70% / Test 30%
    # ใช้ stratify=y เพื่อให้จำนวนคลาสใน train/test คงสัดส่วนเดียวกัน
    # ========================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    # ========================================================
    # 6. Scaling
    # ใช้ StandardScaler เพื่อปรับ scale ของข้อมูลให้มีค่าใกล้ศูนย์และค่าเบี่ยงเบนมาตรฐาน 1
    # ทำเฉพาะบนชุด Train แล้วใช้ค่าจาก Train ในการแปลง Test
    # ========================================================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # เก็บผลลัพธ์แบบ dictionary เพื่อส่งไปหน้าเว็บ
    results = []
    trained_models = {}

    # รายการโมเดลที่ต้องเปรียบเทียบ
    model_defs = [
        ("Decision Tree", DecisionTreeClassifier(random_state=42)),
        ("Naive Bayes", GaussianNB()),
        ("KNN (K=3)", KNeighborsClassifier(n_neighbors=3)),
        (
            "ANN",
            MLPClassifier(hidden_layer_sizes=(1000,), max_iter=1000, random_state=42), 
        ),
    ]

    # ========================================================
    # 7. วนลูปประเมินแต่ละโมเดล
    # สำหรับแต่ละโมเดลจะทำดังนี้:
    # 1. Train กับข้อมูล Train
    # 2. Predict กับข้อมูล Test
    # 3. คำนวณ Accuracy, Precision, Recall, F1-score
    # 4. เก็บผลลัพธ์ไว้ใน results
    # ========================================================
    for model_name, model in model_defs:
        model.fit(X_train_scaled, y_train)
        trained_models[model_name] = model
        y_pred = model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f_measure = f1_score(y_test, y_pred, zero_division=0)

        results.append(
            {
                "Method": model_name,
                "Accuracy": round(float(accuracy) * 100, 2),
                "Precision": round(float(precision) * 100, 2),
                "Recall": round(float(recall) * 100, 2),
                "F Measure": round(float(f_measure) * 100, 2),
            }
        )

    # เรียงผลลัพธ์จาก Accuracy มากไปน้อย เพื่อให้เห็นโมเดลที่ดีที่สุดก่อน
    result_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
    best_model_name = str(result_df.iloc[0]["Method"])

    # ========================================================
    # 8. Flow steps สำหรับแสดงในหน้าเว็บ
    # เป็นลิสต์ของขั้นตอนการทำงานตามลำดับ เช่น Import → Load → Split → Scale → Model Compare
    # ========================================================
    flow_steps = [
        {"step": "1. Import Library", "detail": "Pandas, scikit-learn, preprocessing, and evaluation metrics"},
        {"step": "2. Load Water Potability", "detail": f"Loaded dataset from non_nan.csv with {data.shape[0]} rows and {data.shape[1]} columns"},
        {"step": "3. ตรวจสอบ Dataset", "detail": f"Checked columns, rows, and missing values: {data.isnull().sum().sum()} missing cells"},
        {"step": "4. เตรียมข้อมูล", "detail": "Removed index column (if present) and prepared clean data for training"},
        {"step": "5. X / y", "detail": f"Features: {X.shape[1]} columns, Target: Potability"},
        {"step": "6. Train 70% / Test 30%", "detail": f"Train set: {X_train.shape[0]} rows, Test set: {X_test.shape[0]} rows"},
        {"step": "7. Scaling", "detail": "StandardScaler applied using training data only"},
        {"step": "8. Decision Tree", "detail": "DecisionTreeClassifier(random_state=42)"},
        {"step": "9. Naive Bayes", "detail": "GaussianNB()"},
        {"step": "10. KNN", "detail": "KNeighborsClassifier(n_neighbors=3)"},
        {"step": "11. ANN", "detail": "MLPClassifier(hidden_layer_sizes=(1000,), max_iter=1000)"},
        {"step": "12. Accuracy / Precision / Recall / F Measure", "detail": "Compared all 4 models using evaluation metrics"},
    ]

    # ส่งค่าออกเป็น dictionary สำหรับหน้าเว็บ
    return {
        "rows": data.shape[0],
        "columns": data.shape[1],
        "features": X.shape[1],
        "columns_list": X.columns.tolist(),
        "target": "Potability",
        "train_rows": X_train.shape[0],
        "test_rows": X_test.shape[0],
        "missing_values": data.isnull().sum().to_dict(),
        "results": result_df.to_dict(orient="records"),
        "best_model_name": best_model_name,
        "_best_model": trained_models[best_model_name],
        "_scaler": scaler,
        "class_distribution": y.value_counts().sort_index().to_dict(),
        "steps": flow_steps,
    }


# ============================================================
# 9. Route สำหรับหน้าเว็บ
# Route / จะเรียก home() เพื่อสร้าง dashboard และส่งข้อมูลไปยัง template
# ============================================================
@app.route("/")
def home():
    # เรียก evaluate_models เพื่อคำนวณข้อมูลทั้งหมดที่ต้องใช้บนหน้าเว็บ
    dashboard = evaluate_models()

    # สร้าง summary cards สำหรับแสดงจำนวน Sample, Feature, Train, Test
    metrics = [
        {"label": "Samples", "value": f"{dashboard['rows']:,}"},
        {"label": "Features", "value": dashboard["features"]},
        {"label": "Train Set", "value": f"{dashboard['train_rows']:,}"},
        {"label": "Test Set", "value": f"{dashboard['test_rows']:,}"},
    ]

    # ส่งข้อมูลทั้ง dashboard และ metrics ไปยัง index.html
    return render_template("index.html", dashboard=dashboard, metrics=metrics)


# ============================================================
# 10. Run Web App
# เมื่อรันไฟล์นี้เป็นสคริปต์หลัก จะเริ่ม Flask server ที่ port 5000
# ============================================================
if __name__ == "__main__":
    app.run(
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
    )