import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.manifold import TSNE
import warnings
import pickle
import itertools

# Suppress warnings
warnings.filterwarnings('ignore')
sns.set(style="whitegrid", color_codes=True)

class ParkinsonsDiseaseAnalysis:
    def __init__(self, data_path):
        """Initialize the analysis with data path."""
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Load and prepare the dataset."""
        try:
            self.data = pd.read_csv(self.data_path, index_col='name')
            print("Dataset shape:", self.data.shape)
            return True
        except FileNotFoundError:
            print(f"Error: Could not find {self.data_path}")
            return False
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False

    def plot_correlation_matrix(self):
        """Plot correlation matrix heatmap."""
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(self.data.corr(), dtype=bool))
        sns.heatmap(self.data.corr(), vmin=-1, vmax=1, cmap='BrBG', mask=mask)
        plt.title('Correlation Matrix')
        plt.show()

        # Plot correlation with status
        plt.figure(figsize=(10, 10))
        heatmap = sns.heatmap(
            self.data.corr()[['status']].sort_values(by='status', ascending=False),
            vmin=-1, vmax=1, annot=True, cmap='BrBG'
        )
        heatmap.set_title('Features Correlating with Parkinson\'s Disease', fontdict={'fontsize': 18}, pad=16)
        plt.show()

    def prepare_data(self):
        """Prepare features and target variables."""
        self.X = self.data.drop('status', axis=1)
        self.y = self.data['status']
        print("\nClass distribution:")
        print(self.y.value_counts(normalize=True))

    def visualize_tsne(self):
        """Visualize data using t-SNE."""
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(self.X)
        
        tsne_df = pd.DataFrame(data=X_tsne, columns=['TSNE1', 'TSNE2'])
        tsne_df['Class'] = self.y.values
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=tsne_df, x='TSNE1', y='TSNE2', hue='Class', palette='Set2')
        plt.title('t-SNE Visualization')
        plt.show()
        return tsne_df

    def split_data(self, test_size=0.3, random_state=11):
        """Split data into training and testing sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        print("\nData split shapes:")
        print(f"X_train: {self.X_train.shape}")
        print(f"X_test: {self.X_test.shape}")

    @staticmethod
    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
        """Plot confusion matrix with optional normalization."""
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def train_knn(self):
        """Train and evaluate KNN classifier."""
        print("\nTraining KNN Classifier...")
        param_grid = {'n_neighbors': [3, 5, 7, 9]}
        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(knn, param_grid, scoring='recall', cv=5)
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best Score: {grid_search.best_score_:.4f}")
        
        y_pred = grid_search.predict(self.X_test)
        self.evaluate_model(y_pred, "KNN")
        return grid_search.best_estimator_

    def train_logistic_regression(self):
        """Train and evaluate Logistic Regression."""
        print("\nTraining Logistic Regression...")
        lr = LogisticRegression(max_iter=10000)
        lr.fit(self.X_train, self.y_train)
        y_pred = lr.predict(self.X_test)
        self.evaluate_model(y_pred, "Logistic Regression")
        
        # Plot feature importance
        coef = abs(lr.coef_[0])
        plt.figure(figsize=(10, 6))
        plt.barh(self.X.columns, coef)
        plt.title('Feature Importance (Logistic Regression)')
        plt.show()
        
        return lr

    def train_random_forest(self):
        """Train and evaluate Random Forest."""
        print("\nTraining Random Forest...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5)
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"Best Parameters: {grid_search.best_params_}")
        y_pred = grid_search.predict(self.X_test)
        self.evaluate_model(y_pred, "Random Forest")
        
        # Plot feature importance
        feat_importances = pd.Series(
            grid_search.best_estimator_.feature_importances_, 
            index=self.X.columns
        )
        plt.figure(figsize=(10, 6))
        feat_importances.sort_values().plot(kind='barh')
        plt.title('Feature Importance (Random Forest)')
        plt.show()
        
        return grid_search.best_estimator_

    def evaluate_model(self, y_pred, model_name):
        """Evaluate model performance."""
        print(f"\n{model_name} Results:")
        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        self.plot_confusion_matrix(cm, classes=["Not Parkinson's", "Parkinson's"])
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        accuracy = accuracy_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Recall: {recall:.4f}")

    def save_model(self, model, filename):
        """Save the trained model."""
        try:
            with open(filename, 'wb') as file:
                pickle.dump(model, file)
            print(f"\nModel saved successfully to {filename}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")

def main():
    # Initialize analysis
    analysis = ParkinsonsDiseaseAnalysis('parkinsons.data')
    
    # Load and prepare data
    if not analysis.load_data():
        return
    
    # Perform analysis steps
    analysis.plot_correlation_matrix()
    analysis.prepare_data()
    tsne_df = analysis.visualize_tsne()
    analysis.split_data()
    
    # Train and evaluate models
    knn_model = analysis.train_knn()
    lr_model = analysis.train_logistic_regression()
    rf_model = analysis.train_random_forest()
    
    # Save the best performing model
    analysis.save_model(rf_model, 'rf_model_parkinson.pkl')

if __name__ == "__main__":
    main()