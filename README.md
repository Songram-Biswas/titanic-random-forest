# titanic-random-forest
# 🚢 Titanic Survival Prediction using Random Forest

This project applies a complete machine learning pipeline to predict passenger survival on the Titanic dataset. It includes data preprocessing, custom transformers, feature encoding, standardization, model training with Random Forest, and hyperparameter tuning using GridSearchCV.

---

## 📊 Dataset

The dataset is from the classic [Kaggle Titanic competition](https://www.kaggle.com/c/titanic). It includes features like passenger class, age, gender, and port of embarkation.

- `train.csv` — Used to train and evaluate the model
- `test.csv` — Used for making final survival predictions

---

## ⚙️ Project Features

- **Data Preprocessing**:
  - Stratified sampling to maintain survival ratio
  - Imputation of missing `Age` values using mean
  - One-hot encoding of `Sex` and `Embarked`
  - Dropping irrelevant features like `Name`, `Cabin`, and `Ticket`
- **Custom Transformers**:
  - `AgeImputer` for imputing missing ages
  - `FeatureEncoder` for encoding categorical variables
  - `FeatureDropper` to remove unused columns
- **Pipeline Integration**:
  - Scikit-learn `Pipeline` used to organize preprocessing steps
- **Model Training**:
  - `RandomForestClassifier` with `GridSearchCV` for hyperparameter tuning
  - Evaluation using accuracy on test data
- **Prediction Output**:
  - Generates a `prediction.csv` file for submission or review

---

## 🧪 Libraries Used

```python
numpy
pandas
matplotlib
seaborn
scikit-learn

## 👨‍💻 Author
Songram Biswas
Machine Learning Enthusiast
GitHub: @Songram-Biswas
