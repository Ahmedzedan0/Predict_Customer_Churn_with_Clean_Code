# Predict Customer Churn with Clean Code

## Project Structure

```bash
Predict_Customer_Churn_with_Clean_Code
├── .DS_Store
├── .pytest_cache
│   ├── .gitignore
│   ├── CACHEDIR.TAG
│   ├── README.md
│   └── v
│       ├── cache
│       │   ├── lastfailed
│       │   ├── nodeids
│       │   └── stepwise
├── Guide.ipynb
├── README.md
├── __pycache__
│   ├── churn_library.cpython-310.pyc
│   ├── churn_library.cpython-311.pyc
│   ├── churn_script_logging_and_tests.cpython-310-pytest-8.2.2.pyc
│   ├── churn_script_logging_and_tests.cpython-311-pytest-8.2.2.pyc
│   ├── conftest.cpython-310-pytest-8.2.2.pyc
│   ├── conftest.cpython-311-pytest-8.2.2.pyc
│   └── test_churn_library.cpython-311-pytest-8.2.2.pyc
├── churn_library.py
├── churn_notebook.ipynb
├── churn_script_logging_and_tests.py
├── conftest.py
├── data
│   └── bank_data.csv
├── images
│   ├── eda
│   │   ├── churn_distribution.png
│   │   ├── correlation_heatmap.png
│   │   ├── customer_age_distribution.png
│   │   ├── marital_status_distribution.png
│   │   └── total_transactions_distribution.png
│   └── results
│       ├── classification_report.png
│       ├── feature_importance.png
│       ├── roc_curve.png
│       └── shap_summary.png
├── logs
│   ├── churn_library.log
│   └── churn_script.log
├── models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
├── requirements_py3.10.txt
└── sequencediagram.jpeg
```

## Description

This project aims to predict customer churn using machine learning models with a focus on clean code principles. The following functionalities are implemented:

- **Data Import**: Load data from a CSV file.
- **Exploratory Data Analysis (EDA)**: Perform EDA and save visualizations.
- **Feature Encoding**: Encode categorical features.
- **Feature Engineering**: Prepare features for model training.
- **Model Training**: Train machine learning models and store results.
- **Model Evaluation**: Generate and save evaluation reports and plots.

## Files

- `churn_library.py`: Contains the main functions and classes for data processing, model training, and evaluation.
- `churn_script_logging_and_tests.py`: Includes unit tests for the functions in `churn_library.py` and configures logging.
- `conftest.py`: Configuration file for pytest.
- `churn_notebook.ipynb`: Jupyter notebook for interactive data analysis and model development.
- `Guide.ipynb`: Jupyter notebook guide for the project.
- `requirements_py3.10.txt`: Lists the Python dependencies required for the project.
- `sequencediagram.jpeg`: Sequence diagram of the project workflow.
- `data/bank_data.csv`: Dataset used for training and evaluation.
- `logs/`: Directory containing log files.
- `models/`: Directory containing trained model files.
- `images/`: Directory containing generated visualizations.

## Setup

1. **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd Predict_Customer_Churn_with_Clean_Code
    ```

2. **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements_py3.10.txt
    ```

4. **Run the main script**:
    ```bash
    python churn_library.py
    ```

5. **Run the tests**:
    ```bash
    python -m pytest churn_script_logging_and_tests.py --disable-warnings
    ```

## Usage

The main functions in `churn_library.py` are designed to be run in sequence to process the data, perform EDA, engineer features, train models, and evaluate the results.

1. **Data Import**: Load the data using `import_data`.
2. **EDA**: Perform exploratory data analysis using `perform_eda`.
3. **Feature Encoding**: Encode categorical features using `encoder_helper`.
4. **Feature Engineering**: Prepare the features for model training using `perform_feature_engineering`.
5. **Model Training**: Train the machine learning models using `train_models`.

## Logging

Logs are configured to be stored in the `logs/` directory. If the log files do not exist, they will be created automatically.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
```

Feel free to adjust any sections or details as necessary for your specific project.