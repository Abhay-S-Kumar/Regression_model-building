# Regression_model-building
📈 A repository for building and evaluating various machine learning regression models. This project covers data preprocessing, model training, and performance analysis using Python.
# Regression Model Comparison for California Housing Price Prediction

This repository contains a machine learning project focused on predicting median house values in California using various regression models. The entire workflow, from data exploration and preprocessing to model training and evaluation, is documented in the `Regression_Assignment.ipynb` Jupyter Notebook.

## 📊 Project Overview

The primary goal of this project is to build and compare the performance of several regression algorithms on the California Housing dataset. The model that provides the most accurate predictions for the median house value (`MedHouseVal`) will be identified.

The project follows these key steps:
1.  **Data Loading and Exploration:** The dataset is loaded and analyzed to understand its structure, identify data types, and check for missing or duplicate values.
2.  **Data Preprocessing:**
    * **Outlier Handling:** Outliers are detected using boxplots and are capped using the Interquartile Range (IQR) method to ensure they don't skew the model's performance.
    * **Feature Scaling:** All features are standardized using `StandardScaler` to bring them to a similar scale, which is crucial for many regression algorithms.
3.  **Feature Selection:** A correlation heatmap is used to analyze the relationships between different features and the target variable.
4.  **Model Training:** The preprocessed data is split into training and testing sets. Five different regression models are trained on this data.
5.  **Model Evaluation & Comparison:** Each model's performance is evaluated using standard regression metrics, and the results are compared to determine the best-performing model.

## 💾 Dataset

The project uses the **California Housing** dataset, available through `sklearn.datasets`. The dataset contains 20,640 instances and 8 predictive features. The target variable is the median house value for California districts.

**Features (Independent Variables):**
- `MedInc`: Median income in block group
- `HouseAge`: Median house age in block group
- `AveRooms`: Average number of rooms per household
- `AveBedrms`: Average number of bedrooms per household
- `Population`: Block group population
- `AveOccup`: Average number of household members
- `Latitude`: Block group latitude
- `Longitude`: Block group longitude

**Target (Dependent Variable):**
- `MedHouseVal`: Median house value in units of $100,000

## ⚙️ Methodology

The notebook implements a full machine learning pipeline:

1.  **Exploratory Data Analysis (EDA):**
    - Initial data inspection (`.head()`, `.info()`, `.describe()`).
    - Checked for and confirmed no null values or duplicate rows.
    - Visualized outliers for each feature using boxplots.

2.  **Data Cleaning & Preprocessing:**
    - A custom function was created to handle outliers by capping values at 1.5 times the IQR above the third quartile and below the first quartile.
    - Features were scaled using `StandardScaler` to normalize their distribution.

3.  **Model Building:**
    The following regression models were trained and evaluated:
    - **Linear Regression**
    - **Decision Tree Regressor**
    - **Random Forest Regressor**
    - **Gradient Boosting Regressor**
    - **Support Vector Regressor (SVR)**

4.  **Evaluation Metrics:**
    The models were compared based on three key metrics:
    - **Mean Absolute Error (MAE):** The average absolute difference between predicted and actual values.
    - **Mean Squared Error (MSE):** The average of the squares of the errors.
    - **R-Squared (R²) Score:** The proportion of the variance in the dependent variable that is predictable from the independent variables.

## 📈 Results

The performance of each model on the test set is summarized in the table below:

| Model                       | MAE     | MSE     | R² Score |
| --------------------------- | ------- | ------- | -------- |
| Linear Regression           | 0.491   | 0.431   | 0.663    |
| Decision Tree Regressor     | 0.470   | 0.527   | 0.588    |
| **Random Forest Regressor** | **0.330** | **0.247** | **0.807** |
| Gradient Boosting Regressor | 0.365   | 0.277   | 0.784    |
| Support Vector Regressor    | 0.369   | 0.298   | 0.767    |

## 🚀 Conclusion

Based on the evaluation metrics, the **Random Forest Regressor** is the best-performing model for this dataset. It achieved the highest **R² Score (0.807)** and the lowest **MAE (0.330)** and **MSE (0.247)**, indicating the most accurate and reliable predictions among the tested models.

The Decision Tree Regressor was the worst-performing model, showing a tendency to overfit and generalize poorly on the unseen test data.

## 💻 How to Run

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Install the required libraries:**
    It is recommended to create a virtual environment first.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If a `requirements.txt` file is not available, you can install the libraries manually)*
    ```bash
    pip install numpy pandas scikit-learn matplotlib seaborn jupyter
    ```

3.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook Regression_Assignment.ipynb
    ```

## 🛠️ Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
