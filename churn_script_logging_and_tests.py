#!/Users/ahmedzidane/anaconda3/envs/udacity/bin/python

import os
import logging
import pytest
import pandas as pd

try:
    from churn_library import ChurnPredictor
except ImportError as e:
    logging.error("Error importing churn_library or dependencies: %s", e)
    ChurnPredictor = None
    raise

# Configure logging
log_directory = os.path.expanduser('~/logs')
os.makedirs(log_directory, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_directory, 'churn_script.log'),
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger()

# Unit tests using pytest
def test_import_data():
    churn_predictor = ChurnPredictor()
    try:
        df = churn_predictor.import_data('data/bank_data.csv')
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        logger.info("test_import_data: SUCCESS")
    except Exception as e:
        logger.error("test_import_data: FAILED")
        logger.error(e)
        raise e

def test_perform_eda():
    churn_predictor = ChurnPredictor()
    df = churn_predictor.import_data('data/bank_data.csv')
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    try:
        churn_predictor.perform_eda(df)
        assert os.path.exists('images/eda/churn_distribution.png')
        assert os.path.exists('images/eda/customer_age_distribution.png')
        assert os.path.exists('images/eda/marital_status_distribution.png')
        assert os.path.exists('images/eda/total_transactions_distribution.png')
        assert os.path.exists('images/eda/correlation_heatmap.png')
        logger.info("test_perform_eda: SUCCESS")
    except Exception as e:
        logger.error("test_perform_eda: FAILED")
        logger.error(e)
        raise e

def test_encoder_helper():
    churn_predictor = ChurnPredictor()
    df = churn_predictor.import_data('data/bank_data.csv')
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    category_lst = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    try:
        df_encoded = churn_predictor.encoder_helper(df, category_lst)
        for category in category_lst:
            assert f"{category}_Churn" in df_encoded.columns
        logger.info("test_encoder_helper: SUCCESS")
    except Exception as e:
        logger.error("test_encoder_helper: FAILED")
        logger.error(e)
        raise e

def test_perform_feature_engineering():
    churn_predictor = ChurnPredictor()
    df = churn_predictor.import_data('data/bank_data.csv')
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    try:
        X_train, X_test, y_train, y_test = churn_predictor.perform_feature_engineering(df, 'Churn')
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        logger.info("test_perform_feature_engineering: SUCCESS")
    except Exception as e:
        logger.error("test_perform_feature_engineering: FAILED")
        logger.error(e)
        raise e

def test_train_models():
    churn_predictor = ChurnPredictor()
    df = churn_predictor.import_data('data/bank_data.csv')
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    df = churn_predictor.encoder_helper(df, ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category'])
    X_train, X_test, y_train, y_test = churn_predictor.perform_feature_engineering(df, 'Churn')
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    try:
        churn_predictor.train_models(X_train, X_test, y_train, y_test, param_grid=param_grid, cv=5, max_iter=3000)
        assert os.path.exists('models/rfc_model.pkl')
        assert os.path.exists('models/logistic_model.pkl')
        logger.info("test_train_models: SUCCESS")
    except Exception as e:
        logger.error("test_train_models: FAILED")
        logger.error(e)
        raise e

if __name__ == "__main__":
    pytest.main(["-v", __file__])
