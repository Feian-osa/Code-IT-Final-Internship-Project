import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("data/Student_performance_data.xlsx")
    df['Result'] = df['GPA'].apply(lambda x: 1 if x >= 1.6 else 0)
    return df

df = load_data()

# -----------------------------
# FEATURES (FULL MODEL)
# -----------------------------
features = [
    'Age', 'Gender', 'Ethnicity', 'ParentalEducation',
    'StudyTimeWeekly', 'Absences', 'Tutoring',
    'ParentalSupport', 'Extracurricular',
    'Sports', 'Music', 'Volunteering'
]

X = df[features]
y = df['Result']

# -----------------------------
# SCALING + SPLIT
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# TRAIN MODELS
# -----------------------------
# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, y_pred_lr)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_rf)

# -----------------------------
# SELECT BEST MODEL
# -----------------------------
best_model = rf_model if rf_acc > lr_acc else lr_model
best_model_name = "Random Forest" if rf_acc > lr_acc else "Logistic Regression"

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Students", "Prediction"])

# -----------------------------
# DASHBOARD PAGE
# -----------------------------
if page == "Dashboard":
    st.title("📊 Student Performance Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", len(df))
    col2.metric("Logistic Regression", f"{lr_acc:.3f}")
    col3.metric("Random Forest", f"{rf_acc:.3f}")

    st.subheader("🏆 Best Model")
    st.success(best_model_name)

    # Feature Importance (Random Forest)
    st.subheader("📈 Feature Importance")

    importance = rf_model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(feat_df.set_index("Feature"))

# -----------------------------
# STUDENTS PAGE
# -----------------------------
elif page == "Students":
    st.title("📋 Student Records")

    st.dataframe(df)

    st.subheader("⚠️ High Risk Students (Fail)")
    high_risk = df[df['Result'] == 0]

    st.write(f"Total High Risk Students: {len(high_risk)}")
    st.dataframe(high_risk)

# -----------------------------
# PREDICTION PAGE
# -----------------------------
elif page == "Prediction":
    st.title("🤖 Predict Student Performance")

    st.subheader("Enter Student Details")

    age = st.slider("Age", 15, 25, 18)
    gender = st.selectbox("Gender", [0, 1])
    ethnicity = st.selectbox("Ethnicity", [0, 1, 2, 3])
    parental_edu = st.selectbox("Parental Education", [0, 1, 2, 3, 4])
    study_time = st.slider("Study Time Weekly", 0.0, 20.0, 5.0)
    absences = st.slider("Absences", 0, 50, 5)
    tutoring = st.selectbox("Tutoring", [0, 1])
    parental_support = st.selectbox("Parental Support", [0, 1, 2, 3, 4])
    extracurricular = st.selectbox("Extracurricular", [0, 1])
    sports = st.selectbox("Sports", [0, 1])
    music = st.selectbox("Music", [0, 1])
    volunteering = st.selectbox("Volunteering", [0, 1])

if st.button("Predict"):
    input_data = [[
        age, gender, ethnicity, parental_edu,
        study_time, absences, tutoring,
        parental_support, extracurricular,
        sports, music, volunteering
    ]]

    input_scaled = scaler.transform(input_data)
    prediction = best_model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("✅ Student is likely to PASS")
    else:
        st.error("⚠️ Student is at RISK")

n == 1:
            st.success("✅ Student is likely to PASS")
        else:
            st.error("⚠️ Student is at RISK")