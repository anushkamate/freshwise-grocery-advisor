# Smart Grocery Expiry and Discount Advisor — supplied dataset version

## Included files
- `GroceryDataset.csv`: the dataset supplied by the user (1,757 rows, 8 columns)
- `Smart_Grocery_Expiry_Discount_Advisor.ipynb`: EDA, cleaning, regression, classification, confusion matrices, ANN, saved models and demo
- `app.py`: Streamlit catalog analytics and prediction website
- `requirements.txt`: Python dependencies

## Run
1. Extract the project ZIP.
2. In a terminal in the extracted folder, run `pip install -r requirements.txt`.
3. Run `jupyter notebook` and execute the notebook from top to bottom.
4. Run `streamlit run app.py`.

## Important scope note
The supplied dataset has product category, price, discount text, rating text, title, currency, features and description. It does not include daily sales, current stock, expiry dates or actual waste outcomes. This project therefore uses the real dataset to estimate listed price and classify whether a discount is listed. It cannot honestly train a demand forecast or expiry/waste-risk classifier until those missing fields are collected.
