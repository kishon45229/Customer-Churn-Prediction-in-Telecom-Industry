import streamlit as st
import joblib
import pandas as pd

model = joblib.load('churn_prediction_model.pkl')

st.set_page_config(page_title="Telecom Churn Prediction", page_icon="📊")
st.header(":red[ChurnX]", divider='rainbow')
st.subheader("Your Telecom Churn Prediction App")
st.write("""
    Welcome to the **ChrunX App**! 
    Use this tool to predict whether a customer is likely to churn based on key attributes. 
    Just enter the customer's details below and click 'Predict'.
""")
st.write("----")

tenure = st.number_input('📅 **Tenure (in months)**', min_value=0, max_value=100, step=1)
monthly_charges = st.number_input('💸 **Monthly Charges (in $)**', min_value=0.0, max_value=200.0, step=0.1)
total_charges = st.number_input('💵 **Total Charges (in $)**', min_value=0.0, max_value=10000.0, step=0.1)

input_data = pd.DataFrame({
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'tenure': [tenure]
})

if st.button('🔮 Predict', type="primary"):
    st.markdown("### Prediction Results:")
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.error("The customer is likely to churn.")
        st.write(f"**Probability of Churn:** {prediction_proba[0][1]:.2f} 💔")
    else:
        st.success("The customer is not likely to churn.")
        st.write(f"**Probability of No Churn:** {prediction_proba[0][0]:.2f} 💚")

    st.write("#### Entered Data:")
    st.write(input_data)

st.write("----")
st.markdown(
    """
    📊 Developed by [Abisha Navaneethamani](https://github.com/Abishanavam)
    - For any inquiries, reach out at **itbin-2110-0074@horizoncampus.edu.lk**
    """
)
