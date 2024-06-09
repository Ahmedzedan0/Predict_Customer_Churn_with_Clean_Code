"""
This module contains functions for predicting customer churn.

The functions include:
- import_data: to import data from a CSV file.
- perform_eda: to perform exploratory data analysis and save figures.
- encoder_helper: to encode categorical features.
- perform_feature_engineering: to perform feature engineering.
- classification_report_image: to produce and save classification reports.
- feature_importance_plot: to create and save feature importance plots.
- train_models: to train machine learning models and store results.

"""

import os
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay, classification_report
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    filename=os.path.expanduser('~/logs/churn_library.log'),
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger()

class ChurnPredictor:
    def __init__(self) -> None:
        pass

    @staticmethod
    def import_data(pth: str) -> pd.DataFrame:
        logger.info("Importing data from path: %s", pth)
        return pd.read_csv(pth)

    @staticmethod
    def perform_eda(df: pd.DataFrame) -> None:
        logger.info("Performing EDA")
        df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
        os.makedirs('images/eda', exist_ok=True)

        plt.figure(figsize=(20, 10))
        df['Churn'].hist()
        plt.title('Churn Distribution')
        plt.xlabel('Churn')
        plt.ylabel('Frequency')
        plt.savefig('images/eda/churn_distribution.png')
        plt.close()

        plt.figure(figsize=(20, 10))
        df['Customer_Age'].hist()
        plt.title('Customer Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.savefig('images/eda/customer_age_distribution.png')
        plt.close()

        plt.figure(figsize=(20, 10))
        df['Marital_Status'].value_counts('normalize').plot(kind='bar')
        plt.title('Marital Status Distribution')
        plt.xlabel('Marital Status')
        plt.ylabel('Proportion')
        plt.savefig('images/eda/marital_status_distribution.png')
        plt.close()

        plt.figure(figsize=(20, 10))
        sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
        plt.title('Total Transactions Distribution')
        plt.xlabel('Total Transactions')
        plt.ylabel('Density')
        plt.savefig('images/eda/total_transactions_distribution.png')
        plt.close()

        plt.figure(figsize=(20, 10))
        sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=False, cmap='Dark2_r', linewidths=2)
        plt.title('Correlation Heatmap')
        plt.savefig('images/eda/correlation_heatmap.png')
        plt.close()

    @staticmethod
    def encoder_helper(df: pd.DataFrame, category_lst: list, response: str = 'Churn') -> pd.DataFrame:
        logger.info("Encoding categorical features")
        if not pd.api.types.is_numeric_dtype(df[response]):
            logger.error("The response column '%s' must be numeric.", response)
            raise ValueError(f"The response column '{response}' must be numeric.")

        for category in category_lst:
            new_column_name = f"{category}_{response}"
            category_groups = df.groupby(category)[response].mean()
            df[new_column_name] = df[category].map(category_groups)

        return df

    @staticmethod
    def perform_feature_engineering(df: pd.DataFrame, response: str):
        logger.info("Performing feature engineering")
        X = df.drop(columns=[response])
        y = df[response]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def classification_report_image(y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf):
        logger.info("Generating classification report images")
        os.makedirs('images/results', exist_ok=True)

        plt.figure(figsize=(10, 5))
        plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10}, fontproperties='monospace')

        plt.text(0.5, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.5, 0.05, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.5, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.5, 0.7, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties='monospace')
        
        plt.axis('off')
        plt.savefig('images/results/classification_report.png')
        plt.close()

    @staticmethod
    def feature_importance_plot(model, X_data, output_pth):
        logger.info("Generating feature importance plot")
        os.makedirs(output_pth, exist_ok=True)

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        names = [X_data.columns[i] for i in indices]

        plt.figure(figsize=(20, 10))
        plt.title("Feature Importance")
        plt.bar(range(X_data.shape[1]), importances[indices])
        plt.xticks(range(X_data.shape[1]), names, rotation=90)
        plt.savefig(f"{output_pth}/feature_importance.png")
        plt.close()

    def train_models(self, X_train, X_test, y_train, y_test, param_grid=None, cv=5, max_iter=3000):
        logger.info("Training models")
        os.makedirs('models', exist_ok=True)

        if param_grid is None:
            param_grid = {
                'n_estimators': [200, 500],
                'max_features': ['sqrt', 'log2'],
                'max_depth': [4, 5, 100],
                'criterion': ['gini', 'entropy']
            }

        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression(solver='lbfgs', max_iter=max_iter)

        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=cv, verbose=3)

        print("Training Random Forest...")
        n_iter = len(param_grid['n_estimators']) * len(param_grid['max_features']) * len(param_grid['max_depth']) * len(param_grid['criterion']) * cv

        with tqdm(total=n_iter, desc="Training Random Forest") as pbar:
            cv_rfc.fit(X_train, y_train)
            pbar.update(n_iter)
        
        print("Training Logistic Regression...")
        lrc.fit(X_train, y_train)

        y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
        y_train_preds_lr = lrc.predict(X_train)
        y_test_preds_lr = lrc.predict(X_test)

        joblib.dump(cv_rfc.best_estimator_, 'models/rfc_model.pkl')
        joblib.dump(lrc, 'models/logistic_model.pkl')

        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        RocCurveDisplay.from_estimator(cv_rfc, X_test, y_test, ax=ax, alpha=0.8)
        RocCurveDisplay.from_estimator(lrc, X_test, y_test, ax=ax, alpha=0.8)
        plt.savefig('images/results/roc_curve.png')
        plt.close()

        self.classification_report_image(
            y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf
        )

        self.feature_importance_plot(cv_rfc.best_estimator_, X_train, 'images/results')

        explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.savefig('images/results/shap_summary.png')
        plt.close()


cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'                
]

quant_columns = [
    'Customer_Age',
    'Dependent_count', 
    'Months_on_book',
    'Total_Relationship_Count', 
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 
    'Credit_Limit', 
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 
    'Total_Amt_Chng_Q4_Q1', 
    'Total_Trans_Amt',
    'Total_Trans_Ct', 
    'Total_Ct_Chng_Q4_Q1', 
    'Avg_Utilization_Ratio'
]

keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']


def main():
    churn_predictor = ChurnPredictor()
    df = churn_predictor.import_data('data/bank_data.csv')
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    df = churn_predictor.encoder_helper(df, cat_columns, 'Churn')
    churn_predictor.perform_eda(df)

    df.drop(columns=['Attrition_Flag'], inplace=True)
    df = pd.get_dummies(df, columns=cat_columns, drop_first=True)

    X = df[keep_cols]
    y = df['Churn']

    X_train, X_test, y_train, y_test = churn_predictor.perform_feature_engineering(df, 'Churn')

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv = 5
    max_iter = 3000

    churn_predictor.train_models(X_train, X_test, y_train, y_test, param_grid=param_grid, cv=cv, max_iter=max_iter)


if __name__ == '__main__':
    main()
