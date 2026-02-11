from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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

st.title("Social Media Addiction Prediction")

# Model dropdown
model_name = st.selectbox(
    "Select Model",
    ["logistic", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"]
)

# Upload dataset
uploaded_file = st.file_uploader("Upload Test Dataset CSV")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("Uploaded Data Preview")
    st.write(df.head())

    x_scaled, y, encoder, scaler, categorical_cols, numerical_cols, target_col = preprocess_data(
        df
    )

    # Load selected model
    model = joblib.load(f"model/saved_models/{model_name}.pkl")

    y_pred = model.predict(x_scaled)

    if y is None:
        st.warning(
            "Target column not found. Upload a dataset with a target column to show metrics."
        )
    else:
        st.subheader("Evaluation Metrics")
        st.write("Accuracy:", accuracy_score(y, y_pred))
        st.write("Precision:", precision_score(y, y_pred))
        st.write("Recall:", recall_score(y, y_pred))
        st.write("F1 Score:", f1_score(y, y_pred))
        st.write("AUC:", roc_auc_score(y, y_pred))
        st.write("MCC:", matthews_corrcoef(y, y_pred))

        # Confusion matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        display_labels = (
            ["Not Addicted", "Addicted"]
            if target_col == "Addicted"
            else ["No", "Yes"]
        )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=display_labels,
        )
        disp.plot(cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
