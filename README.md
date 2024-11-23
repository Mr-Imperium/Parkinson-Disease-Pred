# Parkinson's Disease Prediction Model

![Parkinson's Disease](https://img.shields.io/badge/Medical%20AI-Parkinson's%20Disease-blue)
![Python](https://img.shields.io/badge/Python-3.11%2B-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange)

A machine learning web application that predicts the likelihood of Parkinson's Disease based on biomedical voice measurements.

## 📋 Overview

This project implements a Random Forest Classifier to predict Parkinson's Disease using various voice measurement features. The model is deployed as a web application using Streamlit, making it accessible for medical professionals and researchers.

## 🔍 Features

- Trained on the UC Irvine ML Repository Parkinson's Dataset
- Uses Random Forest Classifier with optimized hyperparameters
- Interactive web interface for real-time predictions
- Visualization of feature importance and predictions
- Cross-validated model with detailed performance metrics

## 🛠️ Technical Architecture

- **Model**: Random Forest Classifier with GridSearchCV optimization
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly Express
- **Model Persistence**: Pickle, Joblib

## 📊 Model Performance

The model is trained using GridSearchCV with the following hyperparameters:
- Number of estimators: [100, 200]
- Max depth: [10, 15, 20]
- Min samples split: [2, 5]
- Min samples leaf: [1, 2]
- Max features: ['sqrt', 'log2']

Performance metrics on the test set are available in the training output.

## 🚀 Getting Started

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/parkinson-disease-pred.git
cd parkinson-disease-pred
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Train the model (optional, pre-trained model included):
```bash
python train_model.py
```

4. Run the web application:
```bash
streamlit run app.py
```

### Dataset

The dataset used for training includes various biomedical voice measurements from subjects with and without Parkinson's Disease. Key features include:
- MDVP measurements (Fo, Fhi, Flo, etc.)
- Jitter and Shimmer measurements
- NHR and HNR ratios
- Status (health status of the subject)

## 📖 Usage

1. Launch the web application
2. Input the required voice measurements
3. Click "Predict" to get the model's prediction
4. View the visualization of feature importance and prediction results

## 🔧 Model Training Process

The `train_model.py` script performs the following steps:
1. Loads and preprocesses the dataset
2. Splits data into training and test sets
3. Performs GridSearchCV for hyperparameter optimization
4. Trains the Random Forest model
5. Evaluates model performance
6. Saves the trained model

## 📈 Feature Importance

The model identifies the most significant voice measurements for prediction. The top features and their importance scores are displayed in the training output.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Parkinson's Disease dataset
- Streamlit team for the excellent web framework
- scikit-learn developers for the machine learning tools

## 📞 Contact

For questions and feedback, please open an issue in the GitHub repository.

## ⚠️ Disclaimer

This tool is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with healthcare professionals for medical advice and diagnosis.
