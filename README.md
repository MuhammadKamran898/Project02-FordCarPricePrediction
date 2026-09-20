# 🚗 Ford Car Price Prediction

An end-to-end machine learning application that predicts the price of a Ford car based on its specifications.

The project takes a trained machine learning model and exposes it through a **FastAPI backend**, with a **Streamlit frontend** that allows users to enter vehicle information and receive a predicted price.

---

## 📌 Project Overview

The goal of this project is to build a complete machine learning workflow — from data preprocessing and model training to deployment through an API and interactive web interface.

The application accepts information such as:

* Ford model
* Manufacturing year
* Transmission type
* Mileage
* Fuel type
* Tax
* MPG
* Engine size

and returns an estimated Ford car price.

### Application Flow

```text
User
 │
 ▼
Streamlit Frontend
 │
 │ HTTP POST Request
 ▼
FastAPI Backend
 │
 ▼
Trained ML Model
 │
 ▼
Predicted Ford Price
```

---

## ✨ Features

* 📊 Exploratory Data Analysis
* 🧹 Data preprocessing
* 🔧 Numerical and categorical feature preprocessing
* 🤖 Machine learning regression
* ⚙️ Hyperparameter tuning
* 💾 Trained model serialization with Joblib
* 🚀 FastAPI prediction API
* 🖥️ Streamlit interactive frontend
* 📖 Automatic FastAPI Swagger documentation
* 🔌 Frontend-to-backend API communication

---

## 🧰 Tech Stack

### Programming Language

* Python

### Data Science & Machine Learning

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

### Machine Learning Model

* HistGradientBoostingRegressor

### Backend

* FastAPI
* Uvicorn

### Frontend

* Streamlit

### Model Serialization

* Joblib

### Package / Environment Management

* uv

---

## 📂 Project Structure

```text
Project02_FordCarPricePrediction/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   └── ...
│   │
│   └── models/
│       └── ford_price_model.pkl
│
├── frontend/
│   └── app.py
│
├── data/
│   └── ford(1).csv
│
├── notebooks/
│   └── Project(2).ipynb
│
├── pyproject.toml
├── uv.lock
├── README.md
└── .gitignore
```

---

## 📊 Dataset

The project uses a Ford car dataset containing **17,966 records** and **9 columns**.

### Features

| Feature        | Description                 |
| -------------- | --------------------------- |
| `model`        | Ford car model              |
| `year`         | Manufacturing year          |
| `price`        | Target variable — car price |
| `transmission` | Transmission type           |
| `mileage`      | Vehicle mileage             |
| `fuelType`     | Fuel type                   |
| `tax`          | Vehicle tax                 |
| `mpg`          | Miles per gallon            |
| `engineSize`   | Engine size                 |

The target variable is:

```text
price
```

The model is therefore solving a **regression problem**.

---

## 🔬 Machine Learning Pipeline

The machine learning workflow follows these major stages:

```text
Raw Dataset
     │
     ▼
Data Exploration
     │
     ▼
Data Cleaning
     │
     ▼
Train / Test Split
     │
     ▼
Feature Preprocessing
     │
     ├── Numerical Features
     │       ├── Missing-value imputation
     │       └── Standardization
     │
     └── Categorical Features
             ├── Missing-value imputation
             └── One-Hot Encoding
     │
     ▼
Model Training
     │
     ▼
Hyperparameter Tuning
     │
     ▼
Model Evaluation
     │
     ▼
Final Model
     │
     ▼
Joblib Model File
     │
     ▼
FastAPI
     │
     ▼
Streamlit
```

---

## 🧹 Data Preprocessing

The preprocessing pipeline handles numerical and categorical features separately.

### Numerical Features

Numerical data is processed using:

* Median imputation for missing values
* StandardScaler for feature scaling

```python
numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)
```

### Categorical Features

Categorical data is processed using:

* Most-frequent imputation
* One-hot encoding

```python
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)
```

The preprocessing and model are combined into a single Scikit-learn pipeline.

This ensures that the same preprocessing logic used during training is also applied when making predictions.

---

## 🤖 Model

The final model used by the application is:

```text
HistGradientBoostingRegressor
```

The model is placed inside a Scikit-learn `Pipeline` together with the preprocessing stage.

The final model configuration includes:

```python
HistGradientBoostingRegressor(
    random_state=42,
    l2_regularization=1.0,
    learning_rate=0.1,
    max_depth=None,
    max_leaf_nodes=63,
    min_samples_leaf=20
)
```

Hyperparameters were tuned using `GridSearchCV`.

The final trained pipeline is saved using Joblib:

```python
joblib.dump(hgb_best, "ford_price_model.pkl")
```

---

## 📈 Model Evaluation

The model can be evaluated using standard regression metrics:

### MAE — Mean Absolute Error

Measures the average absolute difference between the actual and predicted prices.

```text
MAE = average prediction error
```

### RMSE — Root Mean Squared Error

Penalizes larger prediction errors more strongly than MAE.

### R² — R-Squared

Measures how much of the variation in the target variable is explained by the model.

Example evaluation code:

```python
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

y_pred = hgb_best.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE:  £{mae:,.2f}")
print(f"RMSE: £{rmse:,.2f}")
print(f"R²:   {r2:.4f}")
```

> Model metric values should be added here after the final evaluation is recorded.

---

# 🚀 Running the Project

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate into the project:

```bash
cd Project02_FordCarPricePrediction
```

---

## 2. Create / Activate the Environment

This project uses `uv` for Python environment and dependency management.

Create the environment if required:

```bash
uv venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```bash
uv sync
```

---

## 3. Start the FastAPI Backend

From the project root:

```powershell
uvicorn backend.app.main:app --reload --port 8001
```

The API will be available at:

```text
http://127.0.0.1:8001
```

---

## 4. Open API Documentation

FastAPI automatically provides interactive Swagger documentation.

Open:

```text
http://127.0.0.1:8001/docs
```

You can use the documentation to test the prediction endpoint directly.

---

## 5. Start the Streamlit Frontend

Open a second terminal and activate the same environment.

Then run:

```powershell
streamlit run frontend/app.py
```

Streamlit will provide a local URL in the terminal.

Open that URL in your browser to use the application.

---

# 🔮 Example Prediction

Example input:

```json
{
  "model": "Ka+",
  "year": 2019,
  "transmission": "Automatic",
  "mileage": 16500,
  "fuelType": "Diesel",
  "tax": 165,
  "mpg": 50.3,
  "engineSize": 1.8
}
```

The application returned an estimated price of:

```text
£21,826
```

This value is a model prediction and should be treated as an estimate rather than a guaranteed market price.

---

# 🔌 API

The FastAPI backend exposes a prediction endpoint that accepts vehicle information and returns the predicted price.

### Endpoint

```text
POST /predict
```

### Request

```json
{
  "model": "Ka+",
  "year": 2019,
  "transmission": "Automatic",
  "mileage": 16500,
  "fuelType": "Diesel",
  "tax": 165,
  "mpg": 50.3,
  "engineSize": 1.8
}
```

### Response

```json
{
  "predicted_price": 21826
}
```

The exact response structure depends on the implementation of the current FastAPI endpoint.

---

# 🖥️ Frontend

The Streamlit frontend provides an interactive interface where users can enter vehicle specifications.

The interface includes inputs for:

* Ford model
* Year
* Transmission
* Mileage
* Fuel type
* Tax
* MPG
* Engine size

When the user clicks **Predict Price**, Streamlit sends the data to the FastAPI backend.

```text
Streamlit
    │
    │ POST /predict
    ▼
FastAPI
    │
    ▼
ML Pipeline
    │
    ▼
Prediction
    │
    ▼
Streamlit Result
```

---

# 🧠 Why FastAPI + Streamlit?

### FastAPI

FastAPI provides a lightweight API layer between the machine learning model and the user interface.

It also provides:

* Automatic API documentation
* Request validation
* JSON-based communication
* Easy integration with other applications

### Streamlit

Streamlit provides a simple interactive interface for demonstrating the machine learning model without requiring a separate frontend framework.

Together, they create a clean separation:

```text
Frontend → API → ML Model
```

---

# 📁 Main Components

### `notebooks/`

Contains the machine learning development and experimentation workflow.

### `backend/app/`

Contains the FastAPI application and prediction logic.

### `backend/models/`

Contains the serialized trained machine learning model.

### `frontend/`

Contains the Streamlit application.

### `data/`

Contains the dataset used for the project.

---

# ⚠️ Notes

The predicted price is generated from patterns learned from the training dataset.

It should not be considered an official valuation or guaranteed selling price.

The quality of predictions depends on:

* Training data
* Feature quality
* Data distribution
* Model performance
* Similarity between new input data and training data

---

# 🔮 Future Improvements

Potential improvements include:

* [ ] Deploy the application publicly
* [ ] Add automated API tests
* [ ] Add stronger input validation
* [ ] Add model performance visualizations
* [ ] Add prediction confidence / uncertainty information
* [ ] Add more detailed vehicle information
* [ ] Add model monitoring
* [ ] Add automated model retraining
* [ ] Add CI/CD
* [ ] Containerize the application with Docker

---

# 📚 Learning Outcomes

This project provided practical experience with:

* Python for data science
* Pandas and NumPy
* Exploratory data analysis
* Data preprocessing
* Feature transformation
* Regression algorithms
* Scikit-learn pipelines
* Hyperparameter tuning
* Model evaluation
* Model serialization
* REST APIs
* FastAPI
* Streamlit
* Connecting an ML model to a web application
* Structuring an ML project for deployment

---

## 👨‍💻 Author

**Muhammad Kamran**

BS Computer Science

---

## ⭐ Project Summary

This project demonstrates the complete journey from a machine learning dataset to a usable application:

```text
Data
 ↓
Exploration
 ↓
Preprocessing
 ↓
Model Training
 ↓
Hyperparameter Tuning
 ↓
Evaluation
 ↓
Model Serialization
 ↓
FastAPI
 ↓
Streamlit
 ↓
Ford Price Prediction
```

The project is designed as a practical demonstration of deploying a machine learning regression model as an interactive application.
