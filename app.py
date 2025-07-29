# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer
from datetime import date
import optuna
import plotly.express as px
import matplotlib.pyplot as plt

nltk.download('stopwords', quiet=True)

# ---------------------------- CSS Styling ---------------------------- #
st.markdown("""
<style>
body {
    background-color: #1f1f2e;
}
header {
    font-size: 36px;
    font-weight: 700;
    padding-bottom: 1rem;
    color: #ffffff;
    text-align: center;
    background: linear-gradient(to right, #3a1c71, #d76d77, #ffaf7b);
    padding: 1rem;
    border-radius: 10px;
}
.topnav {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-bottom: 2rem;
}
.topnav button {
    background: linear-gradient(to right, #00c6ff, #0072ff);
    border: none;
    padding: 0.7rem 1.5rem;
    color: white;
    font-weight: 600;
    border-radius: 8px;
    cursor: pointer;
}
.card {
    background-color: #292c3f;
    color: #f5f5f5;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    text-align: center;
}
</style>

""", unsafe_allow_html=True)
# ---------------------------- Utility Functions ---------------------------- #

@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

def preprocess_text(text: str) -> str:
    text = re.sub(r"[^A-Za-z]", " ", text.lower())
    tokens = text.split()
    tokens = [PorterStemmer().stem(w) for w in tokens if w not in stopwords.words('english')]
    return " ".join(tokens)

@st.cache_resource
def get_sentence_embeddings(texts):
    model = load_embedder()
    return model.encode(texts, show_progress_bar=False)

def optimize_xgb(X, y):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        }
        model = XGBClassifier(**params, use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X, y)
        return model.score(X, y)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    best = XGBClassifier(**study.best_params, use_label_encoder=False, eval_metric='mlogloss')
    best.fit(X, y)
    return best

def recommend_user(df):
    return df['assigned_to'].value_counts().idxmin()

def calculate_completion_rate(tasks):
    if not tasks:
        return 0.0
    completed = sum(1 for t in tasks if t["status"] == "Completed")
    return round((completed / len(tasks)) * 100, 2)

def task_dataframe():
    return pd.DataFrame(st.session_state.tasks)

# ---------------------------- App Setup ---------------------------- #

st.set_page_config("AI Task Manager", layout="wide")

if "tasks" not in st.session_state:
    st.session_state.tasks = []

st.markdown('<header>Unified AI Task Management System</header>', unsafe_allow_html=True)

# ---------------------------- Top Navigation ---------------------------- #
pages = ["Home", "Add Task", "Tasks", "AI Insights"]
nav_selection = st.selectbox("Select Section", pages, index=0, key="nav", label_visibility="collapsed")

# ---------------------------- Home / Dashboard ---------------------------- #
if nav_selection == "Home":
    st.subheader("📊 Task Dashboard Overview")
    total = len(st.session_state.tasks)
    completed = sum(1 for t in st.session_state.tasks if t["status"] == "Completed")
    rate = calculate_completion_rate(st.session_state.tasks)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="card"><h3>Total Tasks</h3><h1>{total}</h1></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="card"><h3>Completed</h3><h1>{completed}</h1></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="card"><h3>Completion Rate</h3><h1>{rate}%</h1></div>', unsafe_allow_html=True)

    df = task_dataframe()
    if not df.empty:
        st.subheader("📂 Task Distribution")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(df, names='category', title="Task by Category", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.pie(df, names='priority', title="Task by Priority", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------- Add Task ---------------------------- #
elif nav_selection == "Add Task":
    st.subheader("📝 Add New Task")
    with st.form("add_task"):
        col1, col2 = st.columns(2)
        with col1:
            description = st.text_area("Task Description *")
            category = st.text_input("Category *")
            priority = st.selectbox("Priority", ["Low", "Medium", "High"])
            status = st.selectbox("Status", ["Pending", "In Progress", "Completed"])
        with col2:
            estimated_hours = st.number_input("Estimated Hours", min_value=0.0, step=0.5)
            due_date = st.date_input("Due Date", value=date.today())
            assigned_to = st.text_input("Assigned To *")

        submitted = st.form_submit_button("Add Task")

        if submitted:
            if not description or not category or not assigned_to:
                st.error("Please fill in all required fields.")
            else:
                st.session_state.tasks.append({
                    "description": description,
                    "category": category,
                    "priority": priority,
                    "status": status,
                    "estimated_hours": estimated_hours,
                    "due_date": str(due_date),
                    "assigned_to": assigned_to
                })
                st.success("✅ Task Added Successfully")

# ---------------------------- Tasks ---------------------------- #
elif nav_selection == "Tasks":
    st.subheader("📋 All Tasks")
    df = task_dataframe()
    if not df.empty:
        with st.expander("🔍 Filters"):
            f1 = st.selectbox("Status", ["All"] + sorted(df['status'].unique()))
            f2 = st.selectbox("Priority", ["All"] + sorted(df['priority'].unique()))
            f3 = st.selectbox("Assigned To", ["All"] + sorted(df['assigned_to'].unique()))
            if f1 != "All":
                df = df[df['status'] == f1]
            if f2 != "All":
                df = df[df['priority'] == f2]
            if f3 != "All":
                df = df[df['assigned_to'] == f3]

        styled_df = df.style.set_properties(**{
            "background-color": "#f4f6f9",
            "color": "#222",
            "border-color": "gray"
        }) 
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.info("No tasks available yet.")

# ---------------------------- AI Insights ---------------------------- #
elif nav_selection == "AI Insights":
    st.subheader("🤖 AI-Powered Priority Predictor")
    uploaded = st.file_uploader("📁 Upload a CSV (columns: task_description, priority, assigned_to)", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)
        required_cols = {"task_description", "priority", "assigned_to"}
        if not required_cols.issubset(df.columns):
            st.error(f"❌ CSV must contain columns: {', '.join(required_cols)}")
            st.stop()

        df.dropna(subset=required_cols, inplace=True)
        df['processed'] = df['task_description'].apply(preprocess_text)
        label_encoder = LabelEncoder()
        df['priority_encoded'] = label_encoder.fit_transform(df['priority'])

        if df['priority_encoded'].nunique() < 2:
            st.warning("⚠ Dataset must contain at least two priority classes.")
            st.stop()

        X = get_sentence_embeddings(df['task_description'].tolist())
        y = df['priority_encoded']
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

        model = optimize_xgb(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("📉 Classification Report")
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose())

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_).plot(ax=ax)
        st.pyplot(fig)

        st.subheader("🧠 Predict Priority")
        new_desc = st.text_area("Enter Task Description")
        if st.button("Predict"):
            vec = get_sentence_embeddings([new_desc])
            proba = model.predict_proba(vec)[0]
            pred_class = label_encoder.inverse_transform([np.argmax(proba)])[0]
            st.success(f"Predicted Priority: {pred_class}")
            st.info(f"Confidence: {round(np.max(proba)*100, 2)}%")
            st.info(f"Suggested User: {recommend_user(df)}")