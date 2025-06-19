import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('autism.pkl', 'rb'))

st.set_page_config(page_title="Autism Prediction In Toddlers", layout="centered")
st.title("Autism Prediction In Toddlers")

st.markdown("""
<style>
body {
    background-color: #f4f4f4;
    font-family: Arial, sans-serif;
}
</style>
""", unsafe_allow_html=True)

with st.form("autism_form"):
    d = st.text_input("Child Name", "")
    age = st.number_input("Child Age in Months", min_value=12, max_value=36, value=24)
    gender = st.radio("Gender", ["male", "female"])
    ethnicity = st.selectbox(
        "Select Ethnicity:",
        [
            "Hispanic", "Latino", "Native Indian", "Others", "Pacifica",
            "White European", "Asian", "Black", "Middle Eastern", "Mixed", "South Asian"
        ],
        index=0
    )
    ethnicity_map = {
        "Hispanic": 0, "Latino": 1, "Native Indian": 2, "Others": 3, "Pacifica": 4,
        "White European": 5, "Asian": 6, "Black": 7, "Middle Eastern": 8, "Mixed": 9, "South Asian": 10
    }
    jaundice = st.radio("Jaundice", ["yes", "no"])
    fma = st.radio("Family member with Autism", ["yes", "no"])

    # 25 questions
    q_keys = [f"q{i+1}" for i in range(25)]
    q_answers = {}
    for i, key in enumerate(q_keys):
        q_answers[key] = st.radio(f"{i+1}. Question {i+1} (yes/no)", ["yes", "no"], key=key)

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        data = {}
        for key in q_keys:
            data[key] = 0 if q_answers[key] == 'yes' else 1
        qch = sum(data.values())
        data['age'] = int(age)
        data['qch'] = qch
        data['ethnicity'] = ethnicity_map[ethnicity]
        data['jaundice'] = 1 if jaundice == 'yes' else 0
        data['fma'] = 1 if fma == 'yes' else 0

        result = np.array([list(data.values())])
        prediction = model.predict(result)
        if prediction[0] == 1:
            data['qch'] = data['qch'] * 0.3
            if data['ethnicity'] == 5:
                data['qch'] += 0.08
            if data['ethnicity'] == 0 or data['ethnicity'] == 1:
                data['qch'] += 0.07
            if data['ethnicity'] == 7:
                data['qch'] += 0.05
            if data['ethnicity'] == 6:
                data['qch'] += 0.03
            if data['ethnicity'] in [2, 4, 8, 9, 10]:
                data['qch'] += 0.02
            if data['jaundice'] == 1:
                data['qch'] += 0.085
            if data['jaundice'] == 1:
                data['qch'] += 0.08
        sol = prediction[0]
        data['qch'] *= 10
        
        if sol == 1:
            st.markdown(f"<h2 style='color:Red'>Yes, {d} has autism with {int(data['qch'])}%</h2>", unsafe_allow_html=True)
            st.markdown("""
            <h3>Suggestions to control Autism</h3>
            1. Learn about Autism: Understand what autism is and how it affects your child.<br>
            2. Keep a Routine: Stick to a regular schedule to help your child feel comfortable.<br>
            3. Use Clear Words: Speak simply and directly to avoid confusion.<br>
            4. Use Pictures: Show your child pictures to help them understand things better.<br>
            5. Practice Being with Others: Help your child learn how to play and talk with others.<br>
            6. Praise Them: Celebrate when your child does something well.<br>
            7. Get Help: Work with doctors and teachers who know about autism.<br>
            8. Take Care of Yourself: Make sure you also have time to relax and rest.<br>
            9. Encourage Their Hobbies: Support what your child enjoys doing.<br>
            10. Be Patient and Try Different Things: Keep trying new ways to help your child, and be patient with them.<br>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color:Green'>No, {d} does not have autism</h2>", unsafe_allow_html=True)
    except Exception as e:
        st.error("Please fill all inputs correctly. Error: " + str(e)) 