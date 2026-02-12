from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
import matplotlib.pyplot as plt


def preprocess_data(
    data,
    target_col="Affects_Academic_Performance",
    id_col="Student_ID",
    positive_label="Yes",
    negative_label="No",
):
    df = data.copy()
    if id_col in df.columns:
        df = df.drop(id_col, axis=1)

    if "Usage_to_Sleep_Ratio" not in df.columns and {
        "Avg_Daily_Usage_Hours",
        "Sleep_Hours_Per_Night",
    }.issubset(df.columns):
        df["Usage_to_Sleep_Ratio"] = (
            df["Avg_Daily_Usage_Hours"] / df["Sleep_Hours_Per_Night"]
        )

    resolved_target_col = target_col
    if resolved_target_col not in df.columns and "Addicted" in df.columns:
        resolved_target_col = "Addicted"

    if resolved_target_col in df.columns:
        y = df[resolved_target_col]
        if y.dtype == "object":
            y = y.map({positive_label: 1, negative_label: 0})
        X = df.drop(resolved_target_col, axis=1)
    else:
        y = None
        X = df

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    encoder = None
    if categorical_cols:
        encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        x_cat = encoder.fit_transform(X[categorical_cols])
        cat_df = pd.DataFrame(
            x_cat,
            columns=encoder.get_feature_names_out(categorical_cols),
            index=X.index,
        )
    else:
        cat_df = pd.DataFrame(index=X.index)

    x_num = X[numerical_cols]
    x_processed = pd.concat([x_num, cat_df], axis=1)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_processed)

    return x_scaled, y, encoder, scaler, categorical_cols, numerical_cols, resolved_target_col

# Evaluation Function
def evaluate_model(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

# Page Configuration
st.set_page_config(
    page_title="ML Assignment 2 – Classification App",
    layout="wide"
)
st.title("📊 ML Assignment 2 – Classification Models Comparison")
st.write(
    "This Streamlit application demonstrates multiple machine learning "
    "classification models trained on a Kaggle dataset."
)


MODEL_PATH = "model/saved_models/"

models = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}



col1, col2 = st.columns(2, gap="large", vertical_alignment="center", width="stretch" )


# Upload dataset
with col1:
    uploaded_file = st.file_uploader("Upload Test Dataset")

# Model dropdown
with col2:
    with open("data/student-social-media-academic-performance-test-data.csv", "rb") as f:
        st.download_button(
            label="Download Sample Test Dataset",
            data=f,
            file_name="student-social-media-academic-performance-test-data.csv",
            mime="text/csv",
            icon=":material/download:",
            icon_position="left"
        )
    
selected_model_name = st.selectbox(
    "Select Classification Model",
    list(models.keys()), width=850
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Dataset Preview")
    st.write(df.head())

    x_scaled, y, encoder, scaler, categorical_cols, numerical_cols, target_col = preprocess_data(
        df
    )

    # Load selected model
    model = joblib.load(f"model/saved_models/{models[selected_model_name]}")

    if y is None:
        st.warning(
            "Target column not found. Upload a dataset with a target column to show metrics."
        )
    else:
        y_pred = model.predict(x_scaled)
        y_prob = model.predict_proba(x_scaled)[:, 1]
        
        metrics = evaluate_model(y, y_pred, y_prob)
        # Display Metrics
        st.subheader(f"📈 Evaluation Metrics – {selected_model_name}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", round(metrics["Accuracy"], 4))
        c2.metric("AUC Score", round(metrics["AUC"], 4))
        c3.metric("Precision", round(metrics["Precision"], 4))

        c4, c5, c6 = st.columns(3)
        c4.metric("Recall", round(metrics["Recall"], 4))
        c5.metric("F1 Score", round(metrics["F1 Score"], 4))
        c6.metric("MCC", round(metrics["MCC"], 4))
        

        # Confusion matrix
        st.subheader("📊 Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        display_labels = (
            ["Not Addicted", "Addicted"]
        )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=display_labels,
        )
        disp.plot(cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig, width=700)
        
        # Classification Report
        st.subheader("📄 Classification Report")
        report = classification_report(y, y_pred, output_dict=True)
        st.write(report)
