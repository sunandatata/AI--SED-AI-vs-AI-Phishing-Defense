import numpy as np
import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import onnxmltools
from onnx.defs import onnx_opset_version
from onnxmltools.convert.common.data_types import StringTensorType
from skl2onnx import convert_sklearn
import skl2onnx

# --- Configuration and Setup ---
# Define paths and file names for modularity and easy upgrading
MODEL_DIR = 'models'
LR_MODEL_NAME = 'logistic_regression_detector.joblib'
TFIDF_VECTORIZER_NAME = 'tfidf_vectorizer.joblib'
ONNX_MODEL_NAME = 'lr_detector.onnx'

os.makedirs(MODEL_DIR, exist_ok=True)
print(f"Model Directory: {MODEL_DIR}")

def load_mock_data():
    """Loads a mock dataset for training/testing the Blue Agent."""
    print("Loading mock dataset...")
    data = {
        'message': [
            "Your account is temporarily suspended. Click here to verify now!",
            "Hello, your delivery is stuck. Track it via this link.",
            "Meeting at 2 PM today to discuss Q3 results.",
            "I noticed a security issue with your login. Please change your password immediately.",
            "Just wanted to check in on the project status. Let me know when you have time.",
            "Congratulations! You won a free iPhone. Claim your prize now!",
            "The quick brown fox jumps over the lazy dog.",
            "Urgent action required: Wire transfer pending.",
        ],
        'label': [1, 1, 0, 1, 0, 1, 0, 1] # 1: Phishing/Attack, 0: Benign
    }
    df = pd.DataFrame(data)
    # Split into train/test (simple split for demonstration)
    X_train, X_test = df['message'][:5], df['message'][5:]
    y_train, y_test = df['label'][:5], df['label'][5:]
    return X_train, X_test, y_train, y_test

def train_and_save_blue_agent(X_train, y_train):
    """Trains the Logistic Regression model and the TF-IDF vectorizer."""
    print("\n--- Phase 1: Training TF-IDF Vectorizer ---")
    
    # Use the max_features and sublinear_tf arguments for optimized feature engineering
    vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True, max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    
    # Save the vectorizer, as it must be used for all future predictions
    vectorizer_path = os.path.join(MODEL_DIR, TFIDF_VECTORIZER_NAME)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Vectorizer saved to: {vectorizer_path}")

    print("\n--- Phase 2: Training Logistic Regression Model ---")
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train_vec, y_train)

    # Save the trained model
    model_path = os.path.join(MODEL_DIR, LR_MODEL_NAME)
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    return vectorizer, model

def evaluate_model(vectorizer, model, X_test, y_test):
    """Evaluates the trained model on the test data."""
    print("\n--- Phase 3: Model Evaluation ---")
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

def convert_to_onnx(vectorizer, model):
    """Converts the trained TF-IDF + LR pipeline to the ONNX format for deployment."""
    print("\n--- Phase 4: ONNX Conversion (Extensible Link Upgrade) ---")
    
    # 1. Define the input type. ONNX models require explicit input shapes and types.
    # Since the input is a text message, we use StringTensorType.
    initial_type = [('string_input', StringTensorType())]

    # 2. Convert the Scikit-learn pipeline (Vectorization + Classification)
    # skl2onnx handles the conversion of TfidfVectorizer and LogisticRegression
    # The target_opset defines the ONNX version to use.
    try:
        onnx_model = convert_sklearn(
            (vectorizer, model),  # Pass the entire pipeline
            initial_types=initial_type,
            target_opset=onnx_opset_version
        )
        
        onnx_path = os.path.join(MODEL_DIR, ONNX_MODEL_NAME)
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
            
        print(f"Successfully converted model pipeline to ONNX at: {onnx_path}")
        print(f"This ONNX file is the scalable 'link' for deployment (Phase 5).")
        
    except Exception as e:
        print(f"ONNX Conversion Failed! Error: {e}")
        print("Ensure all required libraries (onnxmltools, skl2onnx) are installed.")

# --- Main Execution Flow ---
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_mock_data()
    
    # Train and save the models (joblib format)
    vectorizer, model = train_and_save_blue_agent(X_train, y_train)
    
    # Evaluate the initial performance
    evaluate_model(vectorizer, model, X_test, y_test)
    
    # Convert the pipeline to the optimized ONNX format
    convert_to_onnx(vectorizer, model)
    
    print("\nBlue Agent setup complete. The detection model is saved in joblib and ONNX formats.")

    # How to load and use the ONNX model (requires onnxruntime, not demonstrated here)
    # print("\nTo use the ONNX model for deployment, you would install 'onnxruntime' and load the model.")
    # Example:
    # import onnxruntime as rt
    # sess = rt.InferenceSession(onnx_path)
    # input_name = sess.get_inputs()[0].name
    # output_name = sess.get_outputs()[0].name
    # print(f"ONNX Model Ready: Input={input_name}, Output={output_name}")
    
