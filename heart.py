# # UCE2023402-Narmata Bhat
# # UCE2023403-Ichcha Bhat
# # UCE2023413-Kasak Boob

# #-------HEART RISK PREDICTION SYSTEM ---------------

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import streamlit as st
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import (
#     accuracy_score, confusion_matrix, roc_curve, auc, classification_report
# )
# import joblib


# st.title("Heart Disease Prediction System ❤️")

# st.markdown("""
# ### Overview
# Upload your heart disease dataset and let the app train predictive models to detect **High Risk** or **Low Risk** cases.
# """)

# uploaded_file = st.file_uploader("📂Upload your heart.csv dataset", type=["csv"])

# if uploaded_file is not None:
#     try:
#         df = pd.read_csv(uploaded_file)
#     except Exception as e:
#         st.error(f"Could not read the file: {e}")
#         st.stop()

#     st.subheader("📊 Dataset Preview")
#     st.dataframe(df.head())

#     # Clean dataset
#     df = df.dropna()

#     # Try to automatically detect the target/output column
#     target_col = None
#     for col in df.columns:
#         if col.lower() in ['target', 'output', 'result', 'disease', 'heartdisease', 'diagnosis']:
#             target_col = col
#             break

#     if target_col is None:
#         st.error("Could not detect target/output column automatically. Please ensure your file has a column named 'target' or 'output'.")
#         st.stop()

#     # Separate features and labels
#     X = df.drop(target_col, axis=1)
#     y = df[target_col]

#     # Convert text labels ('yes'/'no', etc.) to 1/0
#     if y.dtype == 'object':
#        y = y.astype(str).str.strip().str.lower().map({
#         'yes': 1, 'no': 0,
#         'presence': 1, 'absence': 0,
#         'y': 1, 'n': 0,
#         'positive': 1, 'negative': 0,
#         '1': 1, '0': 0
#       })
#     if y.isnull().any():
#         st.error("⚠️ The output column has unexpected values. Please use 'presence/absence', 'yes/no', or '1/0'.")
#         st.stop()

#     # Standardize numeric features
#     #As the data values may be large so we need to scale down it to 
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)


#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_scaled, y, test_size=0.2, random_state=42
#     )

    
#     log_model = LogisticRegression(max_iter=1000)
#     log_model.fit(X_train, y_train)

#     tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
#     tree_model.fit(X_train, y_train)

#     # Predictions
#     y_pred_log = log_model.predict(X_test)
#     y_prob_log = log_model.predict_proba(X_test)[:, 1]
#     y_pred_tree = tree_model.predict(X_test)
#     y_prob_tree = tree_model.predict_proba(X_test)[:, 1]

    
#     st.subheader("Model Evaluation")

#     acc_log = accuracy_score(y_test, y_pred_log)
#     acc_tree = accuracy_score(y_test, y_pred_tree)

#     st.write(f"**Logistic Regression Accuracy:** {acc_log:.2f}")
#     st.write(f"**Decision Tree Accuracy:** {acc_tree:.2f}")

#     st.text("Classification Report (Logistic Regression):")
#     st.text(classification_report(y_test, y_pred_log))

#     # Confusion Matrix
#     st.subheader(" Confusion Matrix (Logistic Regression)")
#     cm = confusion_matrix(y_test, y_pred_log)
#     fig_cm, ax = plt.subplots()
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
#     st.pyplot(fig_cm)

#     # ROC Curve
#     st.subheader(" ROC Curve Comparison")
#     fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
#     fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_tree)
#     fig_roc, ax = plt.subplots()
#     ax.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {auc(fpr_log, tpr_log):.2f})')
#     ax.plot(fpr_tree, tpr_tree, label=f'Decision Tree (AUC = {auc(fpr_tree, tpr_tree):.2f})')
#     ax.plot([0,1],[0,1],'--',color='gray')
#     ax.legend()
#     ax.set_xlabel("False Positive Rate")
#     ax.set_ylabel("True Positive Rate")
#     ax.set_title("ROC Curve")
#     st.pyplot(fig_roc)

#     # Save best model (Logistic Regression)
#     joblib.dump(log_model, 'heart_model.pkl')
#     joblib.dump(scaler, 'scaler.pkl')
#     st.success(" Model trained and saved successfully!")

    
#     #Entry data to predict based on the trained data
#     st.markdown("---")
#     st.header("🩺Predict Heart Disease Risk")

#     # Load model
#     model = joblib.load('heart_model.pkl')
#     scaler = joblib.load('scaler.pkl')

#     # Input fields
#     age = st.number_input("Age", 20, 100, 50)
#     sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
#     cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
#     trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
#     chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 400, 200)
#     fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1])
#     restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
#     thalach = st.number_input("Max Heart Rate Achieved", 70, 220, 150)
#     exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
#     oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
#     slope = st.selectbox("Slope of Peak Exercise ST Segment (0–2)", [0, 1, 2])
#     ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
#     thal = st.selectbox("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", [1, 2, 3])

#     input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
#                             thalach, exang, oldpeak, slope, ca, thal]])
#     scaled_input = scaler.transform(input_data)

#     if st.button("🔍 Predict Risk"):
#         prediction = model.predict(scaled_input)[0]
#         prob = model.predict_proba(scaled_input)[0][1]

#         st.markdown("---")
#         if prediction == 1:
#             st.error(f"**High Risk of Heart Disease!** (Probability: {prob*100:.2f}%)")
#         else:
#             st.success(f"**Low Risk of Heart Disease.** (Probability: {prob*100:.2f}%)")

#         st.progress(int(prob * 100))

#         # Visual comparison chart
#         st.subheader(" Your Risk Probability")
#         risk_df = pd.DataFrame({
#             'Condition': ['Low Risk', 'High Risk'],
#             'Probability': [1 - prob, prob]
#         })
#         st.bar_chart(risk_df.set_index('Condition'))

# else:
#     st.info("👆 Please upload your `heart.csv` dataset to begin training and prediction.")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc, classification_report
)
import joblib


st.title("Heart Disease Prediction System ❤️")

st.markdown("""
### Overview
Upload your heart disease dataset and let the app train predictive models to detect **High Risk** or **Low Risk** cases.
""")

uploaded_file = st.file_uploader("Upload your heart.csv dataset", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.stop()

    st.subheader(" Dataset Preview")
    st.dataframe(df.head())

    df = df.dropna()

    # Detect target column
    target_col = None
    for col in df.columns:
        if col.lower() in ['target', 'output', 'result', 'disease', 'heartdisease', 'diagnosis']:
            target_col = col
            break

    if target_col is None:
        st.error("Could not detect target/output column automatically.")
        st.stop()

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Convert labels
    if y.dtype == 'object':
        y = y.astype(str).str.strip().str.lower().map({
            'yes': 1, 'no': 0,
            'presence': 1, 'absence': 0,
            'y': 1, 'n': 0,
            'positive': 1, 'negative': 0,
            '1': 1, '0': 0
        })

    if y.isnull().any():
        st.error("Unexpected values in target column.")
        st.stop()

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    
    # Train Models
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)

    tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree_model.fit(X_train, y_train)

    # Predictions
    y_pred_log = log_model.predict(X_test)
    y_prob_log = log_model.predict_proba(X_test)[:, 1]

    y_pred_tree = tree_model.predict(X_test)
    y_prob_tree = tree_model.predict_proba(X_test)[:, 1]

    st.subheader("Model Evaluation")

    st.write(f"**Logistic Regression Accuracy:** {accuracy_score(y_test, y_pred_log):.2f}")
    st.write(f"**Decision Tree Accuracy:** {accuracy_score(y_test, y_pred_tree):.2f}")

    st.text("Classification Report (Logistic Regression):")
    st.text(classification_report(y_test, y_pred_log))

    # Confusion Matrix
    st.subheader(" Confusion Matrix (Logistic Regression)")
    cm = confusion_matrix(y_test, y_pred_log)
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig_cm)

    # ROC Curve
    st.subheader(" ROC Curve Comparison")
    fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
    fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_tree)

    fig_roc, ax = plt.subplots()
    ax.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {auc(fpr_log, tpr_log):.2f})')
    ax.plot(fpr_tree, tpr_tree, label=f'Decision Tree (AUC = {auc(fpr_tree, tpr_tree):.2f})')
    ax.plot([0,1],[0,1],'--',color='gray')
    ax.legend()
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    st.pyplot(fig_roc)

    # Save models
    joblib.dump(log_model, 'logistic_model.pkl')
    joblib.dump(tree_model, 'tree_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    st.success(" Models trained and saved successfully!")

    st.markdown("---")
    st.header("🩺 Predict Heart Disease Risk")

    # Load models
    logistic_model = joblib.load('logistic_model.pkl')
    decision_tree_model = joblib.load('tree_model.pkl')
    scaler = joblib.load('scaler.pkl')

    # Model Selection for Prediction
    model_choice = st.radio(
        "Choose Prediction Model:",
        ('Logistic Regression', 'Decision Tree Classifier')
    )

    # Input fields
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
    cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", 70, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [1, 0])
    oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia", [1, 2, 3])

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])

    scaled_input = scaler.transform(input_data)

    if st.button("🔍 Predict Risk"):
        # Use selected model
        if model_choice == 'Logistic Regression':
            model = logistic_model
        else:
            model = decision_tree_model

        prediction = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][1]

        st.markdown("---")
        if prediction == 1:
            st.error(f"**High Risk of Heart Disease!** (Probability: {prob*100:.2f}%)")
        else:
            st.success(f"**Low Risk of Heart Disease.** (Probability: {prob*100:.2f}%)")

        st.progress(int(prob * 100))

        st.subheader(" Your Risk Probability")
        risk_df = pd.DataFrame({
            'Condition': ['Low Risk', 'High Risk'],
            'Probability': [1 - prob, prob]
        })
        st.bar_chart(risk_df.set_index('Condition'))

else:
    st.info(" Please upload your `heart.csv` dataset to begin training and prediction.")
