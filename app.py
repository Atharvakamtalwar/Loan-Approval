from fastapi import FastAPI
import tensorflow as tf
from pydantic import BaseModel
import numpy as np
from sklearn.preprocessing import StandardScaler




# Initialize FastAPI app
app = FastAPI()

# Load the saved model
classify_model = tf.keras.models.load_model("neural_network_model_classification.h5")

# Define input data structure
class ModelInput(BaseModel):
    input_data: list  # Expecting a list of 34 values from the user

# Endpoint for model prediction
@app.post("/predict/")
async def predict(input: ModelInput):
    # Convert input data to numpy array and reshape it to match model's input shape
    input_data = np.array(input.input_data).reshape(1, -1)  # Reshape for a single prediction
    
    # Scale the input data (use the same scaler used during training)
    # input_data_scaled = scaler.transform(input_data)  # Uncomment if you're using a saved scaler
    
    # Get model predictions
    prediction_prob = classify_model.predict(input_data)
    # Convert probabilities to binary predictions (assuming binary classification)
    prediction = (prediction_prob > 0.5).astype("int32").tolist()


    return {"prediction": prediction}
