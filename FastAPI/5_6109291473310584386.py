import pickle
import numpy as np
import sklearn

def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    """
    Predict diabetes based on input features using a pre-trained Random Forest classifier.

    Parameters:
    Pregnancies (int): Number of pregnancies
    Glucose (float): Glucose level
    BloodPressure (float): Blood pressure level
    SkinThickness (float): Skin thickness
    Insulin (float): Insulin level
    BMI (float): Body Mass Index
    DiabetesPedigreeFunction (float): Diabetes pedigree function
    Age (int): Age in years
    model_path (str): Path to the pickle file containing the trained model

    Returns:
    int: Predicted class (0 or 1)
    """
    # Load the Random Forest classifier from the pickle file
    with open('/content/diabetes_model.pkl', 'rb') as file:
        rf_classifier = pickle.load(file)
    
    # Create a numpy array from the input features
    input_features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

    # Make a prediction using the Random Forest classifier
    prediction = rf_classifier.predict(input_features)

    return prediction[0]

# Example usage:
result = predict_diabetes(2, 120, 70, 30, 80, 25.0, 0.5, 30)
print("Predicted class:", result)

