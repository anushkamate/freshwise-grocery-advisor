# 🛒 FreshWise — Smart Grocery Catalog Advisor

FreshWise is a student machine-learning project that explores grocery product catalog data through an interactive Streamlit dashboard. It combines catalog summaries, product search and filtering, an expiry-tracking prototype, and machine-learning demonstrations for listed-price estimation and discount-status classification.

> **Data note:** The expiry dates, manufacturing dates, and stock quantities added to the provided dataset are synthetic demonstration values. They are not verified supplier or product-batch records and must not be used for real food-safety or inventory decisions.

## ✨ Features

- **Overview dashboard** — summary metrics and charts for products, categories, prices, and listed discounts.
- **Product Explorer** — search product names, filter by category or discount status, and download filtered results.
- **Expiry Tracker** — view manufacturing dates, expiry dates, stock quantities, and days remaining; filter records by expiry status and download a report.
- **Prediction Demo** — demonstrate price estimation and discount-status classification using saved machine-learning models.
- **About FreshWise** — project purpose, current limitations, and future improvement ideas.

## 🧰 Technology Stack

- **Python** — application and data processing
- **Pandas / NumPy** — dataset loading and data preparation
- **Scikit-learn** — machine-learning algorithms
- **Joblib** — saving and loading trained models
- **Streamlit** — interactive web interface
- **Matplotlib / Seaborn** — data visualization and exploratory analysis

## 📁 Project Structure

```text
FreshWise/
├── app.py
├── GroceryDataset.csv
├── requirements.txt
├── README.md
├── grocery_price_model.joblib              # required for Prediction Demo
└── grocery_discount_status_model.joblib    # required for Prediction Demo
```

The two `.joblib` model files are created by running the training notebook. Their filenames must match the paths defined in `app.py`. If your notebook uses different names, either rename the files or update `app.py` accordingly.

## 🚀 Run Locally

### 1. Download or clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

Replace the placeholders with your repository URL and folder name.

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the website

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) that you can open in your browser.

## ☁️ Deploy on Streamlit Community Cloud

1. Push `app.py`, `GroceryDataset.csv`, and `requirements.txt` to your GitHub repository.
2. Include the trained `.joblib` files if you want the Prediction Demo to work.
3. Open [Streamlit Community Cloud](https://share.streamlit.io/) and create or select the app connected to your repository.
4. Select the correct branch and set `app.py` as the main file.
5. Deploy the app. New commits to the connected branch normally trigger a redeployment.

### Model compatibility

Scikit-learn models saved with Joblib can be sensitive to differences in Python and library versions. Train and save the models using compatible versions of Python, NumPy, scikit-learn, and Joblib, and use those same compatible versions in the deployed environment. If the app fails while loading a model, check the final traceback line and the installed package versions, then retrain and replace the saved models if needed.

For reproducible deployment, pin the tested package versions in `requirements.txt` rather than relying indefinitely on unpinned latest versions.

## 📊 Dataset

`GroceryDataset.csv` contains grocery catalog attributes, including product name/title, sub-category, price, discount, rating, currency, features, and product description. The extended dataset also includes:

- `Manufacturing Date`
- `Expiry Date`
- `Stock Quantity`
- `Days Remaining Until Expiry`

The dataset may use either `Title` or `Product Name` for the product-name field, and either `Feature` or `Features` for feature text, depending on which dataset version is being used. Ensure the column names in the CSV match the expectations of the notebook and app.

### About the expiry fields

The original catalog did not contain verified expiry dates, manufacturing dates, or inventory counts. These additional fields were generated synthetically for demonstrating the interface. Days remaining is calculated relative to the dataset's documented reference date; it will not automatically update each day unless the code recalculates it against the current date.

For a real deployment, replace these illustrative values with actual batch-level manufacturing/expiry dates and inventory quantities. Validate the date information before acting on it.

## 🤖 Machine-Learning Models

The project notebook demonstrates five algorithms across two tasks:

| Algorithm | Task |
|---|---|
| Linear Regression | Listed-price estimation |
| Artificial Neural Network (`MLPRegressor`) | Listed-price estimation |
| Decision Tree | Discount-status classification |
| K-Nearest Neighbors (KNN) | Discount-status classification |
| Naive Bayes | Discount-status classification |

Model quality should be assessed using appropriate metrics. For regression, the notebook uses measures such as MAE, RMSE, and R². For classification, use accuracy alongside precision, recall, F1-score, and the confusion matrix—especially because the discount labels may be imbalanced.

These are educational demonstrations using catalog data. Their predictions are not guaranteed market prices and do not establish that a product is safe to consume.

## ⚠️ Current Limitations

- The dataset is a product catalog, not a verified live inventory feed.
- Expiry dates and stock quantities are synthetic demonstration values.
- The current model does **not** learn from actual expiry outcomes, sales history, or measured food waste.
- The Prediction Demo requires compatible trained model files.
- Model results depend on data quality, feature engineering, and training/deployment package compatibility.

## 🔮 Future Improvements

- Replace simulated inventory and expiry fields with verified product-batch records.
- Recalculate days remaining dynamically from the current date.
- Add validated expiry alerts and inventory management workflows.
- Integrate actual sales and stock history to investigate demand forecasting.
- Collect reliable labels before developing and evaluating expiry-risk or food-waste prediction models.
- Improve model evaluation, cross-validation, and handling of class imbalance.
- Add automated tests and pin compatible dependency versions for deployment.

## 👩‍🎓 Project Purpose

FreshWise is intended as an educational prototype demonstrating data preprocessing, exploratory data analysis, supervised machine learning, model persistence, and deployment through an interactive web application.

## 📄 License

No license is specified by default. Add a `LICENSE` file if you intend to publish the project for reuse by others.
