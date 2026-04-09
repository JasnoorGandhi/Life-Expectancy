import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Life Expectancy Predictor",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
    .result-box {
        background: #f0f7f0;
        border: 2px solid #2d6a4f;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-number {
        font-size: 4rem;
        font-weight: 700;
        color: #2d6a4f;
        line-height: 1;
    }
    .result-label {
        font-size: 1rem;
        color: #555;
        margin-top: 0.4rem;
    }
    .section-note {
        font-size: 0.78rem;
        color: #888;
        font-style: italic;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("life_expectancy_model.pkl")

model = load_model()
ohe = model.named_steps['preprocessor'].named_transformers_['country']
COUNTRIES = ohe.categories_[0].tolist()

st.title("🏥 Life Expectancy Predictor")
st.caption("WHO dataset · Ridge Regression · Enter health & economic indicators to predict life expectancy")
st.markdown("---")

col_left, col_right = st.columns([2, 1], gap="large")

with col_left:

    st.subheader("🌍 Demographics")
    c1, c2, c3 = st.columns(3)
    with c1:
        country = st.selectbox("Country", COUNTRIES, index=COUNTRIES.index("India"))
    with c2:
        year = st.number_input("Year", min_value=2000, max_value=2015, value=2015, step=1)
    with c3:
        status_label = st.radio("Development Status", ["Developing", "Developed"], horizontal=True)
    status = 1 if status_label == "Developed" else 0

    st.markdown("---")

    st.subheader("💀 Mortality & Disease")
    st.markdown('<p class="section-note">Enter raw values — log1p transform applied automatically</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        adult_mortality = st.number_input("Adult Mortality", min_value=1, max_value=800, value=150,
                                          help="Deaths per 1000 population aged 15-60")
    with c2:
        infant_deaths = st.number_input("Infant Deaths", min_value=0, max_value=1800, value=20,
                                        help="Per 1000 population")
    with c3:
        hiv_aids = st.number_input("HIV/AIDS Deaths", min_value=0.0, max_value=50.0, value=0.1, step=0.1,
                                   help="Per 1000 live births (ages 0-4)")
    with c4:
        measles = st.number_input("Measles Cases", min_value=0, max_value=212183, value=50,
                                  help="Reported cases per 1000 population")

    st.markdown("---")

    st.subheader("💉 Immunization Coverage")
    st.markdown('<p class="section-note">Enter as percentage 0-99 — scaling applied automatically</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        hepatitis_b = st.slider("Hepatitis B (%)", min_value=1, max_value=99, value=80,
                                help="% of 1-year-olds immunized")
    with c2:
        polio = st.slider("Polio (%)", min_value=3, max_value=99, value=85,
                          help="% of 1-year-olds immunized")
    with c3:
        diphtheria = st.slider("Diphtheria (%)", min_value=2, max_value=99, value=85,
                               help="% of 1-year-olds immunized")

    st.markdown("---")

    st.subheader("💰 Economy & Lifestyle")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        gdp = st.number_input("GDP per Capita (USD)", min_value=1, max_value=120000, value=3000,
                              help="Raw value — log1p applied automatically")
    with c2:
        population = st.number_input("Population", min_value=34, max_value=1400000000, value=10000000,
                                     help="Raw value — log1p applied automatically")
    with c3:
        alcohol = st.number_input("Alcohol (litres/capita)", min_value=0.01, max_value=17.87,
                                  value=4.5, step=0.1, help="Recorded per capita (age 15+)")
    with c4:
        bmi = st.number_input("Avg BMI", min_value=1.0, max_value=87.3, value=38.0, step=0.1)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        schooling = st.number_input("Schooling (years)", min_value=0.0, max_value=20.7, value=12.0, step=0.1)
    with c2:
        income_comp = st.number_input("Income Composition (HDI)", min_value=0.0, max_value=0.948,
                                      value=0.6, step=0.001, help="HDI income index, 0-1")
    with c3:
        pct_expenditure = st.number_input("% Expenditure (GDP%)", min_value=0.0, max_value=19479.0,
                                          value=50.0, step=0.1,
                                          help="Health expenditure as % of GDP — log1p applied automatically")
    with c4:
        total_expenditure = st.number_input("Total Expenditure (%)", min_value=0.1, max_value=17.6,
                                            value=6.0, step=0.1,
                                            help="Govt health spend as % of total govt expenditure")

    st.markdown("---")

    st.subheader("📊 Nutrition")
    st.markdown('<p class="section-note">Enter as percentage 0-28 — scaling applied automatically</p>', unsafe_allow_html=True)
    c1, _ = st.columns([1, 3])
    with c1:
        thinness_5_9 = st.number_input("Thinness 5-9 yrs (%)", min_value=0.1, max_value=28.6,
                                        value=5.0, step=0.1,
                                        help="Prevalence of thinness among children aged 5-9")

with col_right:
    st.subheader("📈 Prediction")

    predict_btn = st.button("Predict Life Expectancy", type="primary", use_container_width=True)

    if predict_btn:
        # Preprocessing to match the notebook's training transformations:
        #
        # Cell 15 → log1p on skewed cols
        # Cell 14 → immunization cols divided by 100 TWICE (notebook bug),
        #           confirmed by scaler means: HepatitisB mean ≈ 0.008 = 80/100/100

        input_data = pd.DataFrame([{
            'Country':                      country,
            'Year':                         year,
            'Status':                       status,
            'AdultMortality':               adult_mortality,

            # log1p (Cell 15)
            'infantdeaths':                 np.log1p(infant_deaths),
            'percentageexpenditure':        np.log1p(pct_expenditure),
            'Measles':                      np.log1p(measles),
            'HIV/AIDS':                     np.log1p(hiv_aids),
            'GDP':                          np.log1p(gdp),
            'Population':                   np.log1p(population),

            # no transform
            'Alcohol':                      alcohol,
            'BMI':                          bmi,
            'Totalexpenditure':             total_expenditure,
            'Schooling':                    schooling,
            'Incomecompositionofresources': income_comp,

            # divided by 100 twice due to Cell 14 bug
            'HepatitisB':                   hepatitis_b / 100 / 100,
            'Polio':                        polio / 100 / 100,
            'Diphtheria':                   diphtheria / 100 / 100,
            'thinness5-9years':             thinness_5_9 / 100 / 100,
        }])

        try:
            prediction = round(float(model.predict(input_data)[0]), 1)

            if 30 <= prediction <= 100:
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-number">{prediction}</div>
                    <div class="result-label">years — predicted life expectancy</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("")
                if prediction >= 75:
                    st.success("Above global average (~72 years)")
                elif prediction >= 65:
                    st.info("Near global average (~72 years)")
                else:
                    st.warning("Below global average (~72 years)")

            else:
                st.error(f"Model returned an unusual value ({prediction} years). "
                         "Please check your inputs are within realistic ranges.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    else:
        st.info("Fill in the inputs on the left, then click **Predict Life Expectancy**.")

    st.markdown("---")
    st.subheader("ℹ️ Model Info")
    st.markdown("""
| Detail | Value |
|---|---|
| Algorithm | Ridge Regression |
| Dataset | WHO Life Expectancy |
| Countries | 191 |
| Year range | 2000–2015 |
| Dropped | `under-fivedeaths`, `thinness1-19years` |
| Scaler | StandardScaler on numeric features |
    """)

    st.markdown("---")
    st.subheader("⚠️ Known Training Issues")
    st.markdown("""
- **Double division bug**: immunization cols and `thinness5-9years` were divided by 100 twice in Cell 14 — this GUI replicates that to match the trained model.
- **NaN target rows**: `Lifeexpectancy` NaNs were not dropped before training, which may slightly affect accuracy.
- **Preprocessing outside pipeline**: `log1p` and scaling must be applied here manually before calling `model.predict()`.
    """)
