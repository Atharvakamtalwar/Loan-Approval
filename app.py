from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods = [*],
    allow_origins = [*],
    allow_methods = ["*"],
    allow_origins = ["*"]
)

# Load the saved model
classify_model = tf.keras.models.load_model("neural_network_model_classification.h5")

# Example mappings for categorical features
employment_status_mapping = {
    "Employed": 1,
    "Unemployed": 2,
    "Self-Employed": 3,
    "Retired": 4,
    "Student": 5
}

education_level_mapping = {
    "High School": 1,
    "Associate's Degree": 2,
    "Bachelor's Degree": 3,
    "Master's Degree": 4,
    "Doctorate": 5
}

marital_status_mapping = {
    "Single": 1,
    "Married": 2,
    "Divorced": 3,
    "Widowed": 4
}

home_ownership_status_mapping = {
    "Renting": 1,
    "Owned": 2,
    "Mortgaged": 3
}

loan_purpose_mapping = {
    "Home Improvement": 1,
    "Debt Consolidation": 2,
    "Business Loan": 3,
    "Personal Loan": 4,
    "Other": 5
}

# Define input data structure based on your dataset
class ModelInput(BaseModel):
    Age: int
    AnnualIncome: int
    CreditScore: int
    EmploymentStatus: str
    EducationLevel: str
    Experience: int
    LoanAmount: int
    LoanDuration: int
    MaritalStatus: str
    NumberOfDependents: int
    HomeOwnershipStatus: str
    MonthlyDebtPayments: int
    CreditCardUtilizationRate: float
    NumberOfOpenCreditLines: int
    NumberOfCreditInquiries: int
    DebtToIncomeRatio: float
    BankruptcyHistory: int
    LoanPurpose: str
    PreviousLoanDefaults: int
    PaymentHistory: int
    LengthOfCreditHistory: int
    SavingsAccountBalance: int
    CheckingAccountBalance: int
    TotalAssets: int
    TotalLiabilities: int
    MonthlyIncome: float
    UtilityBillsPaymentHistory: float
    JobTenure: int
    NetWorth: int
    BaseInterestRate: float
    InterestRate: float
    MonthlyLoanPayment: float
    TotalDebtToIncomeRatio: float

# Function to encode the categorical variables
def encode_categorical(input: ModelInput):
    employment_status = employment_status_mapping.get(input.EmploymentStatus, 0)
    education_level = education_level_mapping.get(input.EducationLevel, 0)
    marital_status = marital_status_mapping.get(input.MaritalStatus, 0)
    home_ownership_status = home_ownership_status_mapping.get(input.HomeOwnershipStatus, 0)
    loan_purpose = loan_purpose_mapping.get(input.LoanPurpose, 0)
    
    return employment_status, education_level, marital_status, home_ownership_status, loan_purpose

# Endpoint for model prediction
@app.post("/predict/")
async def predict(input: ModelInput):
    # Encode categorical features
    employment_status, education_level, marital_status, home_ownership_status, loan_purpose = encode_categorical(input)
    
    # Convert input data to a numpy array
    input_data = np.array([
        input.Age,
        input.AnnualIncome,
        input.CreditScore,
        employment_status,  # Encoded EmploymentStatus
        education_level,    # Encoded EducationLevel
        input.Experience,
        input.LoanAmount,
        input.LoanDuration,
        marital_status,     # Encoded MaritalStatus
        input.NumberOfDependents,
        home_ownership_status,  # Encoded HomeOwnershipStatus
        input.MonthlyDebtPayments,
        input.CreditCardUtilizationRate,
        input.NumberOfOpenCreditLines,
        input.NumberOfCreditInquiries,
        input.DebtToIncomeRatio,
        input.BankruptcyHistory,
        loan_purpose,  # Encoded LoanPurpose
        input.PreviousLoanDefaults,
        input.PaymentHistory,
        input.LengthOfCreditHistory,
        input.SavingsAccountBalance,
        input.CheckingAccountBalance,
        input.TotalAssets,
        input.TotalLiabilities,
        input.MonthlyIncome,
        input.UtilityBillsPaymentHistory,
        input.JobTenure,
        input.NetWorth,
        input.BaseInterestRate,
        input.InterestRate,
        input.MonthlyLoanPayment,
        input.TotalDebtToIncomeRatio
    ]).reshape(1, -1)  # Reshape for a single prediction
    
    # Scale the input data if a scaler was used during training
    # input_data_scaled = scaler.transform(input_data)  # Uncomment if using a saved scaler
    
    # Get model predictions
    prediction_prob = classify_model.predict(input_data)
    
    # Convert probabilities to binary predictions (assuming binary classification)
    prediction = (prediction_prob > 0.5).astype("int32").tolist()

    return {"prediction": prediction}
