# verify_model.py
import pickle
import numpy as np

def verify_model():
    """Verify that the saved model can be loaded and used."""
    
    try:
        # Load the model
        with open('rf_model_parkinson.pkl', 'rb') as file:
            model = pickle.load(file)
            
        print("Model loaded successfully!")
        print("\nModel type:", type(model))
        print("Feature names:", model.feature_names)
        
        # Create a sample input
        sample_input = np.zeros((1, len(model.feature_names)))
        
        # Test prediction
        prediction = model.predict(sample_input)
        probability = model.predict_proba(sample_input)
        
        print("\nTest prediction successful!")
        print("Sample prediction:", prediction)
        print("Sample probability:", probability)
        
    except FileNotFoundError:
        print("Error: Model file not found!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    verify_model()