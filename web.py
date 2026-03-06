import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load trained model
model = pickle.load(open("salary_model.pkl","rb"))

st.title("AI Salary Prediction System")

st.write("Enter employee details to predict salary")

# Load dataset
data = pd.read_csv("employee_data.csv")

# Graph visualization
st.subheader("Experience vs Salary")

fig, ax = plt.subplots()
ax.scatter(data["experience"], data["salary"])
ax.set_xlabel("Experience")
ax.set_ylabel("Salary")

st.pyplot(fig)

# Input fields
age = st.slider("Age",18,60)
experience = st.slider("Years of Experience",0,20)

skills = st.selectbox(
    "Skill",
    ("Python","Java","AI","Web","Data Science")
)

education = st.selectbox(
    "Education",
    ("Bachelors","Masters","PhD")
)

# Convert text inputs to numbers
skill_map = {"Python":2,"Java":1,"AI":0,"Web":3,"Data Science":4}
edu_map = {"Bachelors":0,"Masters":1,"PhD":2}

skill_val = skill_map[skills]
edu_val = edu_map[education]

# Prediction
if st.button("Predict Salary"):

    input_data = np.array([[age,experience,skill_val,edu_val]])

    prediction = model.predict(input_data)

    st.success(f"Predicted Salary: ₹{int(prediction[0])}")

