import streamlit as st
import pandas as pd
import joblib
import json

with open('mapping.json', 'r') as file:
    mappings = json.load(file)
model = joblib.load("model/best_model.joblib")
result_target = joblib.load("model/encoder_target.joblib")

def prediction(data):
    result = model.predict(data)
    final_result = result_target.inverse_transform(result)[0]
    return final_result

marital_status_mapping = mappings["marital_status"]
application_mode_mapping = mappings["application_mode"]
day_time_mapping = mappings["day_evening_attendance"]
previous_qualification_mapping = mappings["previous_qualification"]
parents_qualification_mapping = mappings["parents_qualification"]
parents_occupation_mapping = mappings["parents_occupation"]
displaced_mapping = mappings["displaced"]
tuition_mapping = mappings["tuition_fees_up_to_date"]
gender_mapping = mappings["gender"]
scholar_holder_mapping = mappings["scholar_holder"]

st.set_page_config(page_title="Student Dropout Prediction", layout="wide")

st.sidebar.image("https://toppng.com/uploads/preview/university-logo-design-1156335563731n8qyat0v.png", width=200)
st.sidebar.title("🎓 Student Dropout Prediction")
st.sidebar.markdown("Enter all the field to predict the final status of student (Graduate or Dropout).")

data = pd.DataFrame(columns=[
    "Marital_status", "Application_mode", "Daytime_evening_attendance",
    "Previous_qualification", "Mothers_qualification", "Fathers_qualification",
    "Mothers_occupation", "Fathers_occupation", "Displaced", "Tuition_fees_up_to_date",
    "Gender", "Scholarship_holder", 'Age_at_enrollment',
    'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_grade',
    'Curricular_units_1st_sem_evaluations', 'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_evaluations',
])

with st.expander("Personal Information"):
    age_selected = int(st.number_input(label='Age', value=15))
    marital_status_selected = st.selectbox('Marital Status', options=list(marital_status_mapping.keys()))
    gender_selected = st.selectbox('Gender', options=list(gender_mapping.keys()))
    mothers_qualification_selected = st.selectbox("Mother's Qualification", options=list(parents_qualification_mapping.keys()))
    fathers_qualification_selected = st.selectbox("Father's Qualification", options=list(parents_qualification_mapping.keys()))
    mothers_occupation_selected = st.selectbox("Mother's Occupation", options=list(parents_occupation_mapping.keys()))
    fathers_occupation_selected = st.selectbox("Father's Occupation", options=list(parents_occupation_mapping.keys()))

data.loc[0, 'Age_at_enrollment'] = age_selected
data.loc[0, 'Marital_status'] = marital_status_mapping[marital_status_selected]
data.loc[0, 'Gender'] = gender_mapping[gender_selected]

data.loc[0, 'Mothers_qualification'] = parents_qualification_mapping[mothers_qualification_selected]
data.loc[0, 'Fathers_qualification'] = parents_qualification_mapping[fathers_qualification_selected]
data.loc[0, 'Mothers_occupation'] = parents_occupation_mapping[mothers_occupation_selected]
data.loc[0, 'Fathers_occupation'] = parents_occupation_mapping[fathers_occupation_selected]

with st.expander("Academic Information"):
    previous_qualification_selected = st.selectbox('Previous Qualification', options=list(previous_qualification_mapping.keys()))
    day_evening_selected = st.selectbox('Daytime/Evening Attendance', options=list(day_time_mapping.keys()))
    displaced_selected = st.selectbox('Displaced', options=list(displaced_mapping.keys()))
    tuition_fees_up_to_date_selected = st.selectbox('Tuition Fees Up-to-date', options=list(tuition_mapping.keys()))
    scholarship_holder_selected = st.selectbox('Scholarship Holder', options=list(scholar_holder_mapping.keys()))

    curricular_units_1st_sem_enrolled = int(st.number_input(label='1st Sem Enrolled Units', value=0))
    curricular_units_1st_sem_grade = float(st.number_input(label='1st Sem Average Grade', value=0.0))
    curricular_units_1st_sem_evaluations = int(st.number_input(label='1st Sem Evaluations', value=0))

    curricular_units_2nd_sem_enrolled = int(st.number_input(label='2nd Sem Enrolled Units', value=0))
    curricular_units_2nd_sem_grade = float(st.number_input(label='2nd Sem Average Grade', value=0.0))
    curricular_units_2nd_sem_evaluations = int(st.number_input(label='2nd Sem Evaluations', value=0))

data.loc[0, 'Previous_qualification'] = previous_qualification_mapping[previous_qualification_selected]
data.loc[0, 'Daytime_evening_attendance'] = day_time_mapping[day_evening_selected]
data.loc[0, 'Displaced'] = displaced_mapping[displaced_selected]
data.loc[0, 'Tuition_fees_up_to_date'] = tuition_mapping[tuition_fees_up_to_date_selected]
data.loc[0, 'Scholarship_holder'] = scholar_holder_mapping[scholarship_holder_selected]
data.loc[0, 'Curricular_units_1st_sem_enrolled'] = curricular_units_1st_sem_enrolled
data.loc[0, 'Curricular_units_1st_sem_grade'] = curricular_units_1st_sem_grade
data.loc[0, 'Curricular_units_1st_sem_evaluations'] = curricular_units_1st_sem_evaluations
data.loc[0, 'Curricular_units_2nd_sem_enrolled'] = curricular_units_2nd_sem_enrolled
data.loc[0, 'Curricular_units_2nd_sem_grade'] = curricular_units_2nd_sem_grade
data.loc[0, 'Curricular_units_2nd_sem_evaluations'] = curricular_units_2nd_sem_evaluations


if st.button('Predict'):
    with st.expander("Input Data Summary"):
        st.dataframe(data=data, width=800, height=400)

    try:
        prediction_result = prediction(data)
        st.success(f"Prediction: {prediction_result}", icon="✅")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
