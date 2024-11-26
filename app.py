
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import joblib
st.set_page_config(page_title="Parkinson's Disease Predictor", layout="wide")

class ParkinsonsApp:
    def __init__(self):
        self.feature_names = [
            'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
            'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
            'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
            'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1',
            'spread2', 'D2', 'PPE'
        ]
        
    def load_model(self):
        try:
            with open('rf_model_parkinson.pkl', 'rb') as file:
                self.model = pickle.load(file)
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
            
    def preprocess_input(self, input_data):
        """Preprocess the input data."""
        # Convert to DataFrame
        df = pd.DataFrame([input_data], columns=self.feature_names)
        return df
        
    def predict(self, features):
        """Make prediction using the loaded model."""
        prediction = self.model.predict(features)
        probability = self.model.predict_proba(features)
        return prediction[0], probability[0]

    def run(self):
        """Run the Streamlit application."""
        st.title("Parkinson's Disease Prediction App")
        
        # Sidebar
        st.sidebar.header("About")
        st.sidebar.info(
            "This application uses machine learning to predict the likelihood "
            "of Parkinson's Disease based on voice measurements."
        )
        
        # Main content
        tab1, tab2, tab3 = st.tabs(["Prediction", "Batch Processing", "Model Info"])
        
        with tab1:
            self.show_prediction_tab()
            
        with tab2:
            self.show_batch_processing_tab()
            
        with tab3:
            self.show_model_info_tab()

    def show_prediction_tab(self):
        """Display the prediction interface."""
        st.header("Individual Prediction")
        
        # Create columns for input fields
        col1, col2 = st.columns(2)
        
        input_data = {}
        
        # First column
        with col1:
            for feature in self.feature_names[:11]:
                input_data[feature] = st.number_input(
                    f"Enter {feature}",
                    value=0.0,
                    format="%.6f"
                )
        
        # Second column
        with col2:
            for feature in self.feature_names[11:]:
                input_data[feature] = st.number_input(
                    f"Enter {feature}",
                    value=0.0,
                    format="%.6f"
                )
        
        if st.button("Predict"):
            features = self.preprocess_input(input_data)
            prediction, probability = self.predict(features)
            
            # Display prediction
            st.subheader("Prediction Results")
            if prediction == 1:
                st.warning("⚠️ Potential Parkinson's Disease detected")
            else:
                st.success("✅ No Parkinson's Disease detected")
            
            # Display probability
            st.write("Probability Distribution:")
            prob_df = pd.DataFrame({
                'Condition': ['No Parkinson\'s', 'Parkinson\'s'],
                'Probability': probability
            })
            fig = px.bar(prob_df, x='Condition', y='Probability',
                        color='Condition', range_y=[0,1])
            st.plotly_chart(fig)

    def show_batch_processing_tab(self):
        """Display the batch processing interface."""
        st.header("Batch Prediction")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with voice measurements",
            type=["csv"]
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if st.button("Process Batch"):
                    # Verify columns
                    missing_cols = set(self.feature_names) - set(df.columns)
                    if missing_cols:
                        st.error(f"Missing columns: {missing_cols}")
                        return
                    
                    # Make predictions
                    predictions = self.model.predict(df[self.feature_names])
                    probabilities = self.model.predict_proba(df[self.feature_names])
                    
                    # Add predictions to dataframe
                    results_df = df.copy()
                    results_df['Prediction'] = predictions
                    results_df['Probability_Parkinsons'] = probabilities[:, 1]
                    
                    # Display results
                    st.subheader("Batch Processing Results")
                    st.write(results_df)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "Download Results",
                        csv,
                        "parkinsons_predictions.csv",
                        "text/csv",
                        key='download-csv'
                    )
                    
                    # Show summary statistics
                    st.subheader("Summary Statistics")
                    positive_cases = (predictions == 1).sum()
                    total_cases = len(predictions)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Cases", total_cases)
                    with col2:
                        st.metric("Positive Cases", positive_cases)
                        
                    # Plot distribution
                    fig = px.histogram(
                        results_df,
                        x='Probability_Parkinsons',
                        title='Distribution of Parkinson\'s Probability'
                    )
                    st.plotly_chart(fig)
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    def show_model_info_tab(self):
        """Display model information."""
        st.header("Model Information")
        
        st.subheader("Feature Importance")
        try:
            importances = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            })
            importances = importances.sort_values('importance', ascending=False)
            
            fig = px.bar(
                importances,
                x='importance',
                y='feature',
                orientation='h',
                title='Feature Importance in Model Predictions'
            )
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error("Could not display feature importance.")
        
        st.subheader("Model Details")
        st.write("""
        This model uses a Random Forest Classifier trained on voice measurement data 
        to predict the likelihood of Parkinson's Disease. The model considers 22 
        different voice measurements to make its predictions.
        
        Key Features:
        - Uses voice frequency measurements
        - Considers jitter and shimmer variations
        - Analyzes noise ratios and nonlinear dynamics
        
        Note: This tool is for educational purposes only and should not be used as 
        a substitute for professional medical diagnosis.
        """)

if __name__ == "__main__":
    app = ParkinsonsApp()
    if app.load_model():
        app.run()
