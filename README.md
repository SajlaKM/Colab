# Diabetes Prediction Using Pima Indians Diabetes Dataset

This project predicts whether a person is diabetic or not based on the **Pima Indians Diabetes Dataset** from Kaggle. The model is built using **Random Forest Classifier** and includes data preprocessing, feature scaling, model training, evaluation, and prediction.

---

## 📂 Dataset
The dataset contains the following features:
- `Pregnancies` - Number of times pregnant
- `Glucose` - Plasma glucose concentration
- `BloodPressure` - Diastolic blood pressure (mm Hg)
- `SkinThickness` - Triceps skinfold thickness (mm)
- `Insulin` - 2-Hour serum insulin (mu U/ml)
- `BMI` - Body mass index (weight in kg/(height in m)^2)
- `DiabetesPedigreeFunction` - Diabetes pedigree function
- `Age` - Age in years
- `Outcome` - 1 for diabetic, 0 for non-diabetic (Target variable)

---

## 🚀 Installation & Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/diabetes-prediction.git
   cd diabetes-prediction
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Download the dataset from Kaggle and place it in the project directory.

---

## 📜 Usage
### 1️⃣ Run the Program
```sh
python diabetes_prediction.py
```

### 2️⃣ Code Workflow
1. **Import Required Libraries**
2. **Load & Explore the Dataset**
3. **Handle Missing Values**
4. **Split Data into Train and Test Sets**
5. **Scale Features Using StandardScaler**
6. **Train the Model Using Random Forest**
7. **Evaluate the Model**
8. **Make Predictions on New Data**

### 3️⃣ Example Prediction
```python
import numpy as np
from joblib import load

# Load the trained model
model = load('diabetes_prediction_model.pkl')
scaler = load('scaler.pkl')

# Example new data
new_data = np.array([[5, 116, 74, 0, 0, 25.6, 0.201, 30]])
new_data = scaler.transform(new_data)
prediction = model.predict(new_data)

print("Diabetic" if prediction[0] == 1 else "Non-Diabetic")
```

---

## 📊 Model Performance
- **Accuracy:** ~85% (Varies based on data split)
- **Confusion Matrix & Classification Report included**

---

## 🛠 Future Improvements
- Implement **hyperparameter tuning** using GridSearchCV.
- Try **other machine learning models** like SVM, XGBoost, and Neural Networks.
- Deploy the model using **Flask or FastAPI**.

---

## 🤝 Contributing
Feel free to contribute by forking this repo, making improvements, and submitting a pull request. 🚀

---

## 📜 License
This project is licensed under the MIT License.

Happy Coding! 😊

