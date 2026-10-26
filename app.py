import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

st.set_page_config(page_title="FreshWise | Grocery Assistant", page_icon="🛒", layout="wide")

BASE = Path(__file__).parent
DATA_PATH = BASE / "GroceryDataset.csv"
PRICE_MODEL_PATH = BASE / "grocery_price_model.joblib"
DISCOUNT_MODEL_PATH = BASE / "grocery_discount_status_model.joblib"

st.markdown("""
<style>
.block-container {padding-top: 1.7rem; padding-bottom: 2.5rem; max-width: 1280px;}
.fw-hero {padding: 1.6rem 1.8rem; border-radius: 18px; background: linear-gradient(115deg,#123f32,#26795b); color: white; margin-bottom: 1rem;}
.fw-hero h1 {font-size: 2.35rem; margin: 0 0 .35rem 0; color: white;}
.fw-hero p {font-size: 1.02rem; margin: 0; color: #e4f4ec;}
.fw-note {padding: .85rem 1rem; border-radius: 12px; background: #fff7dc; color: #59450b; margin: .6rem 0 1rem 0; border: 1px solid #f0df9d;}
.fw-section {font-size: 1.35rem; font-weight: 700; margin: .6rem 0 .2rem 0;}
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 1rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    # support both original and arranged column labels
    aliases = {"Title": "Product Name", "Feature": "Features"}
    data = data.rename(columns={k:v for k,v in aliases.items() if k in data.columns and v not in data.columns})
    for col in ["Product Name", "Sub Category", "Price", "Currency", "Discount", "Rating",
                "Features", "Product Description", "Manufacturing Date", "Expiry Date",
                "Stock Quantity", "Days Remaining Until Expiry"]:
        if col not in data.columns:
            data[col] = np.nan
    data["Price_numeric"] = pd.to_numeric(data["Price"].astype(str).str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
    data["Rating_numeric"] = pd.to_numeric(data["Rating"].astype(str).str.extract(r"([0-5](?:\.\d+)?)", expand=False), errors="coerce")
    data["Features"] = data["Features"].fillna("").astype(str)
    data["Product Description"] = data["Product Description"].fillna("").astype(str)
    data["Description_length"] = data["Product Description"].str.len()
    data["Feature_count"] = data["Features"].apply(lambda x: sum(1 for line in x.splitlines() if line.strip()))
    data["Discount"] = data["Discount"].fillna("Unknown").astype(str)
    data["Has_discount"] = (~data["Discount"].str.lower().isin(["no discount", "nan", "", "unknown"])).astype(int)
    data["Stock Quantity"] = pd.to_numeric(data["Stock Quantity"], errors="coerce")
    data["Days Remaining Until Expiry"] = pd.to_numeric(data["Days Remaining Until Expiry"], errors="coerce")
    for col in ["Manufacturing Date", "Expiry Date"]:
        data[col] = pd.to_datetime(data[col], errors="coerce")
    return data

if not DATA_PATH.exists():
    st.error("GroceryDataset.csv was not found. Upload it beside app.py in your GitHub repository.")
    st.stop()

df = load_data(DATA_PATH)

st.markdown("""
<div class="fw-hero">
  <h1>🛒 FreshWise</h1>
  <p>Your simple grocery catalog and expiry-tracking dashboard</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="fw-note"><b>Demo data note:</b> Manufacturing dates, expiry dates and stock quantities in this dataset are simulated examples. Verify them against real package labels and inventory records before making real decisions.</div>', unsafe_allow_html=True)

page = st.sidebar.radio("Navigate", ["Overview", "Product Explorer", "Expiry Tracker", "Prediction Demo", "About FreshWise"])
st.sidebar.markdown("---")
st.sidebar.caption("FreshWise helps you explore product catalog information and demonstrate ML predictions.")

if page == "Overview":
    st.subheader("At a glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Products", f"{len(df):,}")
    c2.metric("Categories", f"{df['Sub Category'].nunique():,}")
    c3.metric("Discounts listed", f"{int(df['Has_discount'].sum()):,}")
    c4.metric("Average listed price", f"{df['Price_numeric'].mean():.2f}" if df["Price_numeric"].notna().any() else "N/A")
    st.markdown("### Explore your catalog")
    left, right = st.columns(2)
    with left:
        st.caption("Product count by category")
        counts = df["Sub Category"].fillna("Unspecified").value_counts().head(12)
        st.bar_chart(counts, height=310)
    with right:
        st.caption("Listed price distribution")
        prices = df["Price_numeric"].dropna()
        if not prices.empty:
            st.bar_chart(prices.clip(upper=prices.quantile(.98)).value_counts(bins=12).sort_index(), height=310)
        else:
            st.info("No numeric price values were found.")
    st.info("Use **Product Explorer** to search the catalog, **Expiry Tracker** to review days remaining, or **Prediction Demo** to try the saved ML models.")

elif page == "Product Explorer":
    st.subheader("Find a product")
    st.caption("Search by product name and filter by category or discount status.")
    f1, f2, f3 = st.columns([2, 1, 1])
    with f1:
        query = st.text_input("Search product", placeholder="e.g., cereal, milk, snacks")
    with f2:
        categories = ["All categories"] + sorted(df["Sub Category"].dropna().astype(str).unique().tolist())
        selected_cat = st.selectbox("Category", categories)
    with f3:
        discount_filter = st.selectbox("Discount", ["All", "Discount listed", "No discount"])
    view = df.copy()
    if query.strip():
        view = view[view["Product Name"].astype(str).str.contains(query.strip(), case=False, na=False)]
    if selected_cat != "All categories":
        view = view[view["Sub Category"].astype(str) == selected_cat]
    if discount_filter == "Discount listed":
        view = view[view["Has_discount"] == 1]
    elif discount_filter == "No discount":
        view = view[view["Has_discount"] == 0]
    st.caption(f"Showing {len(view):,} matching products")
    show_cols = ["Product Name", "Sub Category", "Price", "Currency", "Discount", "Rating"]
    st.dataframe(view[show_cols].head(500), use_container_width=True, hide_index=True)
    st.download_button("Download filtered results (CSV)", view.to_csv(index=False).encode("utf-8"), "freshwise_filtered_products.csv", "text/csv")

elif page == "Expiry Tracker":
    st.subheader("Expiry Tracker")
    st.caption("Review product dates, stock quantity and time remaining until expiry.")
    if df["Days Remaining Until Expiry"].notna().sum() == 0:
        st.warning("Expiry fields are not present in this dataset. Add Manufacturing Date, Expiry Date, Stock Quantity and Days Remaining Until Expiry columns.")
    else:
        days = df["Days Remaining Until Expiry"]
        expired = int((days < 0).sum())
        soon = int(days.between(0, 7).sum())
        later = int((days > 7).sum())
        a,b,c = st.columns(3)
        a.metric("Past illustrative expiry", f"{expired:,}")
        b.metric("Expires in 0–7 days", f"{soon:,}")
        c.metric("More than 7 days left", f"{later:,}")
        cutoff = st.selectbox("Show products", ["All", "Expired", "Expires in 7 days", "More than 7 days left"])
        exp_view = df.copy()
        if cutoff == "Expired":
            exp_view = exp_view[exp_view["Days Remaining Until Expiry"] < 0]
        elif cutoff == "Expires in 7 days":
            exp_view = exp_view[exp_view["Days Remaining Until Expiry"].between(0, 7)]
        elif cutoff == "More than 7 days left":
            exp_view = exp_view[exp_view["Days Remaining Until Expiry"] > 7]
        exp_cols = ["Product Name", "Manufacturing Date", "Expiry Date", "Stock Quantity", "Days Remaining Until Expiry", "Sub Category"]
        exp_view = exp_view.sort_values("Days Remaining Until Expiry", na_position="last")
        st.dataframe(exp_view[exp_cols].head(500), use_container_width=True, hide_index=True)
        st.download_button("Download expiry report (CSV)", exp_view.to_csv(index=False).encode("utf-8"), "freshwise_expiry_report.csv", "text/csv")
        st.caption("Dates and stock are simulated demonstration values, not verified batch records. Do not use them for food-safety decisions.")

elif page == "Prediction Demo":
    st.subheader("Try the machine-learning demo")
    st.caption("This demo estimates catalog price and classifies discount status from product attributes.")
    if not PRICE_MODEL_PATH.exists() or not DISCOUNT_MODEL_PATH.exists():
        st.warning("Trained model files are not available yet. Run the notebook to create both .joblib files and upload them to GitHub.")
    else:
        try:
            price_model = joblib.load(PRICE_MODEL_PATH)
            discount_model = joblib.load(DISCOUNT_MODEL_PATH)
            categories = sorted(df["Sub Category"].dropna().astype(str).unique().tolist())
            with st.form("predict_form"):
                st.markdown("#### Product details")
                subcat = st.selectbox("Sub-category", categories)
                discount = st.selectbox("Discount label", ["No Discount", "After $5 OFF", "After $10 OFF", "Other listed discount"])
                r1, r2 = st.columns(2)
                with r1:
                    rating = st.slider("Rating (if known)", 0.0, 5.0, 4.0, 0.1)
                    reviews = st.number_input("Review count (if known)", min_value=0, value=100, step=10)
                with r2:
                    desc_len = st.number_input("Description length", min_value=0, value=500, step=50)
                    feature_count = st.number_input("Number of feature lines", min_value=0, value=4, step=1)
                submitted = st.form_submit_button("Generate prediction", type="primary", use_container_width=True)
            if submitted:
                price_x = pd.DataFrame([{"Sub Category":subcat, "Discount":discount, "Rating_numeric":rating,
                                         "Review_count":reviews, "Description_length":desc_len, "Feature_count":feature_count}])
                estimated_price = max(0.0, float(price_model.predict(price_x)[0]))
                discount_x = pd.DataFrame([{"Sub Category":subcat, "Price_numeric":estimated_price, "Rating_numeric":rating,
                                            "Review_count":reviews, "Description_length":desc_len, "Feature_count":feature_count}])
                predicted_discount = int(discount_model.predict(discount_x)[0])
                a,b = st.columns(2)
                a.metric("Estimated catalog price", f"{estimated_price:.2f}")
                b.metric("Predicted discount status", "Discount listed" if predicted_discount else "No discount")
                st.success("Prediction generated. Treat this as a model demonstration, not a guaranteed market price.")
        except Exception as e:
            st.error("The saved model could not be loaded. This is usually caused by missing or incompatible package versions.")
            st.code(f"{type(e).__name__}: {e}")
            st.info("Retrain and save the models using package versions compatible with requirements.txt, then replace the .joblib files in GitHub.")
    st.markdown("---")
    st.caption("This prediction demo does not predict expiry dates or food waste.")

else:
    st.subheader("About FreshWise")
    st.markdown("""
**Purpose**  
FreshWise is a student project that combines grocery catalog exploration, basic expiry tracking and machine-learning demonstrations in one interface.

**What it can do**
- Summarize catalog products, categories, prices and listed discounts.
- Search and filter products.
- Display expiry dates, stock quantities and days remaining when those fields are available.
- Demonstrate price estimation and discount-status classification using saved models.

**Current limitations**
- The expiry dates and stock quantities in the extended dataset are synthetic demonstration values.
- The supplied catalog is not verified live store inventory.
- Predictions are educational demonstrations and should not be treated as guaranteed prices or food-safety advice.

**Future improvements**
- Replace simulated inventory fields with verified product-batch data.
- Connect real store stock and sales data.
- Improve model evaluation and add validated demand or expiry-risk models.
""")
