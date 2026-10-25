from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="FreshWise | Grocery Advisor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent
DATA_PATH = BASE / "GroceryDataset.csv"
PRICE_MODEL_PATH = BASE / "grocery_price_model.joblib"
DISCOUNT_MODEL_PATH = BASE / "grocery_discount_status_model.joblib"

st.markdown("""
<style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1450px;}
    .hero {
        padding: 1.5rem 1.7rem; border-radius: 18px;
        background: linear-gradient(120deg, #103c35 0%, #176b55 58%, #80b99a 100%);
        color: white; margin-bottom: 1.2rem;
    }
    .hero h1 {font-size: 2.35rem; margin: 0 0 .4rem 0; color: white;}
    .hero p {font-size: 1.02rem; margin: 0; color: #e9fff2;}
    .small-note {font-size: .88rem; color: #64748b;}
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color); padding: 14px 16px;
        border-radius: 14px; border: 1px solid rgba(128,128,128,.15);
    }
    .section-title {font-size: 1.25rem; font-weight: 650; margin: .4rem 0 .7rem 0;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_catalog(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates().copy()
    df["Price_numeric"] = pd.to_numeric(
        df["Price"].astype(str).str.replace(r"[^0-9.]", "", regex=True), errors="coerce"
    )
    rating_text = df["Rating"].fillna("").astype(str)
    df["Rating_numeric"] = pd.to_numeric(
        rating_text.str.extract(r"Rated\s*([0-5](?:\.\d+)?)", expand=False), errors="coerce"
    )
    df["Review_count"] = pd.to_numeric(
        rating_text.str.extract(r"based on\s*(\d+)\s*reviews?", expand=False), errors="coerce"
    )
    df["Feature"] = df["Feature"].fillna("").astype(str)
    df["Product Description"] = df["Product Description"].fillna("").astype(str)
    df["Title"] = df["Title"].fillna("Untitled product").astype(str)
    df["Sub Category"] = df["Sub Category"].fillna("Unknown").astype(str)
    df["Discount"] = df["Discount"].fillna("Unknown").astype(str)
    df["Currency"] = df["Currency"].fillna("").astype(str)
    df["Description_length"] = df["Product Description"].str.len()
    df["Feature_count"] = df["Feature"].apply(
        lambda x: sum(1 for line in x.splitlines() if line.strip())
    )
    df["Has_discount"] = (df["Discount"].str.lower() != "no discount").astype(int)
    return df

if not DATA_PATH.exists():
    st.error("Dataset not found. Place GroceryDataset.csv in the same folder as app.py.")
    st.stop()

df = load_catalog(DATA_PATH)
categories = sorted(df["Sub Category"].dropna().unique().tolist())
discount_labels = sorted(df["Discount"].dropna().unique().tolist())

with st.sidebar:
    st.markdown("## 🛒 FreshWise")
    st.caption("Smart Grocery Catalog Advisor")
    page = st.radio(
        "Navigate",
        ["Overview", "Product Explorer", "Prediction Demo", "Project Notes"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Prototype built with Python, Pandas, Scikit-learn and Streamlit.")

if page == "Overview":
    st.markdown("""
    <div class="hero">
      <h1>FreshWise Grocery Advisor</h1>
      <p>Explore product catalog patterns, review listed discounts, and try the machine-learning demo.</p>
    </div>
    """, unsafe_allow_html=True)
    st.warning(
        "Dataset scope: the supplied file is a product catalog. It does not contain daily sales, "
        "stock quantity, expiry dates or actual waste outcomes. This website therefore demonstrates "
        "catalog analysis and discount-status prediction—not real expiry-risk prediction."
    )
    total = len(df)
    cat_count = df["Sub Category"].nunique()
    discount_count = int(df["Has_discount"].sum())
    price_count = int(df["Price_numeric"].notna().sum())
    avg_price = df["Price_numeric"].mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Catalog products", f"{total:,}")
    c2.metric("Sub-categories", f"{cat_count:,}")
    c3.metric("Products with discount label", f"{discount_count:,}")
    c4.metric("Average listed price", f"{avg_price:,.2f}" if pd.notna(avg_price) else "N/A")
    st.write("")
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown('<div class="section-title">Largest product categories</div>', unsafe_allow_html=True)
        counts = df["Sub Category"].value_counts().head(12).sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        counts.plot(kind="barh", ax=ax)
        ax.set_xlabel("Number of products")
        ax.set_ylabel("")
        ax.spines[["top", "right", "left"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.markdown('<div class="section-title">Discount label split</div>', unsafe_allow_html=True)
        split = df["Has_discount"].map({0: "No discount", 1: "Discount listed"}).value_counts()
        fig, ax = plt.subplots(figsize=(7, 5))
        split.plot(kind="bar", ax=ax)
        ax.set_ylabel("Number of products")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    st.markdown('<div class="section-title">Recently listed products in the dataset</div>', unsafe_allow_html=True)
    st.dataframe(
        df[["Title", "Sub Category", "Price", "Discount", "Rating", "Currency"]].head(8),
        use_container_width=True, hide_index=True
    )

elif page == "Product Explorer":
    st.markdown('<div class="hero"><h1>Product Explorer</h1><p>Search and filter the supplied grocery catalog.</p></div>', unsafe_allow_html=True)
    search = st.text_input("Search product title", placeholder="Try: cereal, juice, snacks...")
    f1, f2, f3 = st.columns(3)
    with f1:
        cat_filter = st.selectbox("Sub-category", ["All categories"] + categories)
    with f2:
        disc_filter = st.selectbox("Discount", ["All discount labels", "No discount", "Discount listed"])
    with f3:
        price_max_default = float(df["Price_numeric"].dropna().quantile(.95)) if df["Price_numeric"].notna().any() else 100.0
        max_price = st.number_input("Maximum listed price", min_value=0.0, value=max(1.0, round(price_max_default, 2)))
    result = df.copy()
    if search.strip():
        result = result[result["Title"].str.contains(search.strip(), case=False, na=False)]
    if cat_filter != "All categories":
        result = result[result["Sub Category"] == cat_filter]
    if disc_filter == "No discount":
        result = result[result["Discount"].str.lower() == "no discount"]
    elif disc_filter == "Discount listed":
        result = result[result["Discount"].str.lower() != "no discount"]
    result = result[result["Price_numeric"].isna() | (result["Price_numeric"] <= max_price)]
    st.caption(f"{len(result):,} matching products")
    display_cols = ["Title", "Sub Category", "Price", "Discount", "Rating", "Currency"]
    st.dataframe(result[display_cols], use_container_width=True, hide_index=True, height=440)
    st.download_button(
        "⬇️ Download filtered products as CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="freshwise_filtered_products.csv",
        mime="text/csv",
        use_container_width=False
    )

elif page == "Prediction Demo":
    st.markdown('<div class="hero"><h1>Prediction Demo</h1><p>Test the catalog-based machine-learning models.</p></div>', unsafe_allow_html=True)
    st.info("The models predict catalog patterns only. They do not predict product expiry or actual food waste.")
    if not (PRICE_MODEL_PATH.exists() and DISCOUNT_MODEL_PATH.exists()):
        st.warning("Saved model files are not present yet. Run all cells in the Jupyter notebook to create them.")
        st.code("jupyter notebook\n# Run all notebook cells, then:\nstreamlit run app.py", language="bash")
    else:
        price_model = joblib.load(PRICE_MODEL_PATH)
        discount_model = joblib.load(DISCOUNT_MODEL_PATH)
        with st.form("predict_form"):
            st.markdown("#### Enter product characteristics")
            a, b = st.columns(2)
            with a:
                subcat = st.selectbox("Product sub-category", categories)
                discount_label = st.selectbox("Discount label", ["No Discount"] + [x for x in discount_labels if x.lower() != "no discount"])
                rating = st.slider("Rating (if known)", min_value=0.0, max_value=5.0, value=4.0, step=0.1)
            with b:
                reviews = st.number_input("Review count", min_value=0, max_value=1000000, value=100, step=10)
                desc_len = st.number_input("Description length", min_value=0, max_value=50000, value=500, step=50)
                feature_count = st.number_input("Number of feature lines", min_value=0, max_value=100, value=4, step=1)
            run = st.form_submit_button("Analyze product", type="primary", use_container_width=True)
        if run:
            price_x = pd.DataFrame([{
                "Sub Category": subcat, "Discount": discount_label,
                "Rating_numeric": rating, "Review_count": reviews,
                "Description_length": desc_len, "Feature_count": feature_count
            }])
            estimated_price = max(0.0, float(price_model.predict(price_x)[0]))
            discount_x = pd.DataFrame([{
                "Sub Category": subcat, "Price_numeric": estimated_price,
                "Rating_numeric": rating, "Review_count": reviews,
                "Description_length": desc_len, "Feature_count": feature_count
            }])
            discount_prediction = int(discount_model.predict(discount_x)[0])
            p1, p2 = st.columns(2)
            p1.metric("Estimated listed price", f"{estimated_price:,.2f}")
            p2.metric("Predicted discount status", "Discount listed" if discount_prediction else "No discount")
            st.caption("Price is an estimate from catalog attributes, not a price recommendation or guarantee.")
    st.divider()
    st.markdown("#### What the intended expiry advisor will need")
    st.write("To add true expiry alerts and unsold-stock risk, extend the dataset with product ID, observation date, opening/current stock, daily units sold, expiry date, promotions, and actual unsold/wasted units.")

else:
    st.markdown('<div class="hero"><h1>Project Notes</h1><p>Scope, methods and next steps.</p></div>', unsafe_allow_html=True)
    st.markdown("### Current dataset")
    st.write("The supplied dataset includes product title, sub-category, listed price, discount label, rating text, currency, features and product description.")
    st.markdown("### Machine-learning demonstrations")
    st.markdown("""
    - **Linear Regression:** estimates listed price.
    - **Decision Tree:** predicts discount status.
    - **KNN:** predicts discount status from similar catalog attributes.
    - **Naive Bayes:** predicts discount status using feature distributions.
    - **ANN / MLPRegressor:** performs an additional listed-price estimation experiment.
    """)
    st.markdown("### Future work for the original expiry advisor")
    st.markdown("""
    1. Record daily sales and stock for each product.
    2. Record expiry dates and actual waste/unsold outcomes.
    3. Create time-based training and test splits.
    4. Train demand forecasting and waste-risk models on real labels.
    5. Calibrate discount suggestions against margins, store policy and food-safety requirements.
    """)
    st.error("Never use a discount recommendation to sell expired or unsafe food.")
