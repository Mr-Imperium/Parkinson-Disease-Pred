# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
import warnings

warnings.filterwarnings('ignore')

def train_and_save_model():
    """Train the Random Forest model and save it as a pickle file."""
    
    print("Loading dataset...")
    try:
        # Load the dataset
        data = pd.read_csv('parkinsons.data')
        
        # Prepare features and target
        X = data.drop(['status', 'name'], axis=1)
        y = data['status']
        
        print("\nDataset Info:")
        print(f"Number of features: {X.shape[1]}")
        print(f"Number of samples: {X.shape[0]}")
        print(f"Distribution of classes:\n{y.value_counts(normalize=True)}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("\nTraining Random Forest model...")
        # Define parameter grid for GridSearchCV
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        
        # Create and train model using GridSearchCV
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        print("\nBest parameters found:")
        for param, value in grid_search.best_params_.items():
            print(f"{param}: {value}")
        
        # Evaluate model
        y_pred = best_model.predict(X_test)
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        
        # Save feature names
        feature_names = X.columns.tolist()
        best_model.feature_names = feature_names
        
        print("\nSaving model...")
        # Save the model
        with open('rf_model_parkinson.pkl', 'wb') as file:
            pickle.dump(best_model, file)
        
        print("Model saved successfully as 'rf_model_parkinson.pkl'")
        
        # Print feature importances
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': best_model.feature_importances_
        })
        feature_importance = feature_importance.sort_values(
            'importance', ascending=False
        )
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
    except FileNotFoundError:
        print("Error: 'parkinsons.data' file not found!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    train_and_save_model()