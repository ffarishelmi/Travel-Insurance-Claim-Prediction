# Travel Insurance Claim Prediction

**Description**

A machine learning project to predict the likelihood of travel insurance claims using various models. It involves data preprocessing, exploratory data analysis, and model evaluation to assist insurance companies in risk assessment and decision-making. Best model results are saved into a sav format file that can be used to deploy to the application.

## Project Overview

This project focuses on predicting travel insurance claims using machine learning techniques. The goal is to build a predictive model that helps insurance companies identify the likelihood of a customer filing a claim based on historical data. This can assist companies in risk assessment and creating more tailored insurance products.

## Objectives

- Analyze historical travel insurance data to understand customer behavior.
- Build a machine learning model to predict the likelihood of insurance claims.
- Provide actionable insights for optimizing insurance policies and reducing risk.

## Dataset

The dataset used in this project was sourced from Kaggle and contains information about customer demographics, travel details, and whether an insurance claim was filed. The key features include customer age, duration of travel, travel type, and insurance claim status.

## Methodology

1. **Data Preprocessing**: Cleaned and prepared the data by handling missing values, encoding categorical variables, and normalizing numerical features.
2. **Exploratory Data Analysis (EDA)**: Analyzed key features to identify trends and relationships that impact claim likelihood.
3. **Model Building**: Experimented with several machine learning algorithms for classification problems to predict insurance claims.
4. **Model Evaluation**: Evaluated models using recall to choose the best performing model.

## Key Insights

- Older customers tend to have a higher likelihood of filing travel insurance claims.
- Customers with longer travel durations are more likely to make claims.
- The Random Forest model showed the best performance in terms of accuracy and generalizability.

## Tools Used

- **Basic Libraries**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Statistical Hypothesis Testing**: SciPy
- **Applying Algorithm Chains**: Scikit-learn (ColumnTransformer, Pipeline)
- **Data Encoding**: Scikit-learn (OneHotEncoder), Category Encoders
- **Data Scaling**: Scikit-learn (RobustScaler)
- **Data Splitting**: Scikit-learn (train\_test\_split)
- **Modeling**: Scikit-learn (Logistic Regression, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier), XGBoost, LightGBM
- **Model Benchmarking**: Scikit-learn (cross\_val\_score, StratifiedKFold)
- **Metrics Evaluation**: Scikit-learn (confusion\_matrix, classification\_report, recall\_score, precision\_recall\_curve, auc, learning\_curve)
- **Handling Imbalanced Data**: Scikit-learn (compute\_sample\_weight), Imbalanced-learn (Pipeline, SMOTE, ADASYN, SMOTEENN)
- **Hyperparameter Tuning**: Scikit-learn (RandomizedSearchCV)
- **Saving Model**: Pickle
- **Calculate Training Time**: Time

## How to Run

1. Clone the repository:
   ```sh
   git clone https://github.com/ffarishelmi/Travel-Insurance-Claim-Prediction.git
   ```
2. Install the required library:
   ```sh
   pip install scipy 
   pip install scikit-learn
   pip install category_encoders
   pip install xgboost
   pip install lightgbm
   pip install imbalanced-learn 
   pip install pickle5
   ```
3. Run travel\_insurance\_claim\_prediction.ipynb file the notebook environment you are using.

## Conclusion

This project provides a machine learning-based solution to predict travel insurance claims, which can help insurance companies in risk assessment and decision-making processes. By understanding key factors that influence claims, insurers can better tailor their products to customer needs.

## Next Steps

- Experiment with other advanced machine learning algorithms to further improve prediction accuracy.
- Create a dashboard to visualize key statistics data to monitor insurance sales performance.

Feel free to contribute by suggesting improvements or adding new features!
