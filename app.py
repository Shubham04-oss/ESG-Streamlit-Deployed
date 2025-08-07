
import streamlit as st
import joblib
import pandas as pd

# Load the saved model
try:
    model = joblib.load('linear_regression_model.joblib')
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop() # Stop execution if the model can't be loaded

# Load the saved preprocessing components
try:
    median_growth_rate = joblib.load('median_growth_rate.joblib')
    one_hot_encoded_columns = joblib.load('one_hot_encoded_columns.joblib')
    selected_features = joblib.load('selected_features.joblib')
    st.write("Preprocessing components loaded successfully!")
except Exception as e:
    st.error(f"Error loading preprocessing components: {e}")
    st.stop() # Stop execution if components can't be loaded


st.title('Company Revenue Prediction App')

st.write("""
This application predicts a company's revenue based on various financial and ESG (Environmental, Social, and Governance) metrics.
""")

st.header("How to Use:")
st.write("""
1. Enter the values for each of the features in the input fields below.
2. Click the "Predict Revenue" button to get the predicted revenue for the company with the entered characteristics.
""")

st.header("About the Model:")
st.write("""
The model used for this prediction is a Linear Regression model. It was trained on a dataset containing historical financial and ESG data for various companies.
""")

# Create input fields for each selected feature (excluding 'Revenue')
input_data = {}
# Ensure selected_features is a list before iterating
if isinstance(selected_features, list):
    for feature in selected_features:
        if feature != 'Revenue':
            # Assuming numerical features for now, can add type checking if needed
            input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

    # Create a button to trigger prediction
    if st.button('Predict Revenue'):
        # Convert input_data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Reindex the DataFrame to match the model's expected feature order
        # Exclude 'Revenue' as it's the target variable
        features_for_prediction = [f for f in selected_features if f != 'Revenue']

        # Ensure all expected features are in the input_df, add missing ones with default value 0
        for col in features_for_prediction:
            if col not in input_df.columns:
                input_df[col] = 0.0

        input_df = input_df[features_for_prediction]


        # Make a prediction
        try:
            prediction = model.predict(input_df)
            # Display the prediction
            st.write(f"Predicted Revenue: {prediction[0]:.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

else:
    st.error("Selected features not loaded correctly.")

