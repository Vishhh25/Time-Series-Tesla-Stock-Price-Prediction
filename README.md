# 🚀 **Tesla Stock Price Prediction Using Machine Learning**  

![Stock Market](https://img.shields.io/badge/Stock_Price_Prediction-Time_Series_Analysis-red?style=for-the-badge&logo=tesla&logoColor=white)  
![ML Model](https://img.shields.io/badge/ML_Models-LSTM_%7C_XGBoost_%7C_Lasso-blue?style=for-the-badge&logo=python&logoColor=white)  
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)  

---

## 📌 **Project Overview**  

This project focuses on **predicting Tesla’s stock prices** using **machine learning models** for time-series forecasting. Various approaches, including **Random Forest Classifier, Long Short-Term Memory (LSTM), XGBoost, and Lasso Regression**, were applied to forecast stock price movements.  

### 🎯 **Goals:**  
- **Predict** whether the **stock price will go up or down** using a **classification model**.  
- **Forecast** the **exact stock price for the next day** using **regression models**.  
- **Compare different models** and optimize for **accuracy and precision**.  

---

## 📊 **Dataset & Preprocessing**  

- **Source:** Data was fetched using `yfinance`, containing **3,632 rows × 7 columns**.  
- **Data Cleaning:** Removed irrelevant columns (`Dividends` and `Stock Splits`).  
- **Feature Engineering:**  
  - Created `Tomorrow` column storing **next-day closing price**.  
  - Added `Target` column: `1` if next day’s price is **higher**, `0` if **lower**.  
  - Calculated **rolling averages and moving ratios** (2, 5, 60, 250, 1000-day windows).  

---

## 🧠 **Machine Learning Models Used**  

### **1️⃣ Random Forest Classifier (Binary Classification)**  
- **Used to predict** if the **next day's stock price** will increase (`1`) or decrease (`0`).  
- **Initial Accuracy:** `52%` (low precision).  
- **After Backtesting:** **Precision improved to `67%`**.  

### **2️⃣ Long Short-Term Memory (LSTM) Network**  
- **Deep learning model** used for time-series prediction.  
- **Sequential Model:** 50 LSTM units per layer + Dropout layers for regularization.  
- **Performance Metrics:**  
  - **Accuracy:** `61.05%`  
  - **Precision:** `61.05%`  
  - **Mean Absolute Error (MAE):** `11.97`  
  - **Root Mean Squared Error (RMSE):** `17.51`  
  - **R² Score:** `0.85`  

### **3️⃣ XGBoost (Gradient Boosting for Time-Series)**  
- **Powerful tree-based model** for structured time-series data.  
- **Parameters Used:** `n_estimators=100`, `learning_rate=0.05`, `max_depth=6`.  
- **Performance:**  
  - **Accuracy:** `81.48%`  
  - **Precision:** `81.48%`  
  - **R² Score:** `0.94`  

### **4️⃣ Lasso Regression (Regularized Linear Regression)**  
- **Trained on scaled features** to predict the exact stock price.  
- **Performance:**  
  - **Accuracy:** `82.54%`  
  - **Precision:** `82.54%`  
  - **Mean Absolute Error (MAE):** `6.73`  
  - **R² Score:** `0.95`  

### **5️⃣ Lasso Regression + PCA (Dimensionality Reduction)**  
- **Principal Component Analysis (PCA)** applied to retain `95%` variance.  
- **Lower accuracy (`39.68%`) due to loss of important features** in dimensionality reduction.  

---

## 🔥 **Results & Model Comparison**  

| Model          | Accuracy | Precision | R² Score | MAE  | RMSE  |
|---------------|----------|----------|--------|------|------|
| Random Forest | 52%      | 67%      | N/A    | N/A  | N/A  |
| LSTM          | 61.05%   | 61.05%   | 0.85   | 11.97 | 17.51 |
| XGBoost       | 81.48%   | 81.48%   | 0.94   | 7.28  | 11.97 |
| **Lasso**     | **82.54%** | **82.54%** | **0.95** | **6.73**  | **11.97** |
| Lasso + PCA   | 39.68%   | 39.68%   | N/A    | N/A  | N/A  |

✅ **Best Performing Model:** **Lasso Regression (without PCA)**  
- **Achieved `82.54%` accuracy** in predicting stock prices.  
- **Lowest Mean Absolute Error (MAE)** among all models.  
- **XGBoost also performed well (`81.48%` accuracy)** but had slightly **higher error rates** than Lasso.  

---

## 🚀 **How to Run the Project**  

### 🔧 **Requirements**
Before running the code, install the required dependencies:  

```bash
pip install numpy pandas matplotlib scikit-learn xgboost tensorflow keras yfinance
📌 Steps to Run
Clone the repository

bash

Copy

Edit

git clone https://github.com/Vishhh25/Tesla-Stock-Prediction.git

cd Tesla-Stock-Prediction

Run the Jupyter Notebook

bash

Copy

Edit

jupyter notebook

Open Time_Series_Tesla_Stock_Price_Prediction.ipynb and execute all cells.
📬 Contact
📧 Email: vishwapraval@gmail.com

🚀 "Predicting Tesla's stock prices using AI-driven insights!" 🚀
