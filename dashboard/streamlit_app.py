"""
streamlit_app.py
================
PriceOracle — AI-Powered Price Prediction Dashboard (Streamlit)

Run:
    streamlit run dashboard/streamlit_app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'dashboard'))

from src.predictor import PricePredictionPipeline
from src.scraper import scrape_product
from src.timeseries import TimeSeriesForecaster
from src.festival_effect import FestivalAnalyzer
from src.statistics import DatasetAnalyzer
from src.visualizations import (
    price_trend_with_ma, bollinger_bands_chart,
    price_volatility_chart, seasonal_trends_chart,
    forecast_chart, festival_impact_chart,
)
from dashboard.ml.category_detector import CategoryDetector
from dashboard.ml.price_predictor import PricePredictor
from dashboard.ml.similarity_engine import SimilarityEngine

MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title='PriceOracle — AI Price Prediction',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── Dark theme custom CSS ─────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #06060f; }
    .metric-box {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-box h3 { color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; margin: 0; }
    .metric-box p { color: #e2e8f0; font-size: 1.5rem; font-weight: 700; margin: 0.3rem 0 0; }
    .festival-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }
    .festival-card h4 { margin: 0 0 0.3rem; color: #e2e8f0; }
    .festival-card .price { color: #34d399; font-size: 1.3rem; font-weight: 700; }
    .festival-card .discount { color: #fbbf24; font-size: 0.9rem; }
    .product-detail { padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.05); }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_predictor():
    return PricePredictionPipeline()

@st.cache_resource(show_spinner=False)
def load_forecaster():
    return TimeSeriesForecaster()

@st.cache_resource(show_spinner=False)
def load_festival_analyzer():
    return FestivalAnalyzer()

@st.cache_resource(show_spinner=False)
def load_dataset_analyzer():
    return DatasetAnalyzer()

@st.cache_resource(show_spinner=False)
def load_ts_predictor():
    return PricePredictor(MODELS_DIR)

@st.cache_resource(show_spinner=False)
def load_similarity_engine():
    return SimilarityEngine(MODELS_DIR)


# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title('📊 PriceOracle')
st.sidebar.markdown('AI-Powered Price Prediction')
st.sidebar.markdown('---')

page = st.sidebar.radio(
    'Navigate',
    ['Product Analyzer', 'Price Trend & Forecast', 'Festival Impact', 'Dataset Analytics'],
    index=0,
)

st.sidebar.markdown('---')
st.sidebar.caption('Built with Streamlit, Plotly, scikit-learn & real Flipkart data')


# ══════════════════════════════════════════════════════════════
#   PAGE 1: Product Analyzer
# ══════════════════════════════════════════════════════════════
if page == 'Product Analyzer':
    st.title('Product Analyzer')
    st.markdown('Paste a product link or enter details manually to get price predictions.')

    tab_url, tab_manual = st.tabs(['Paste Product URL', 'Enter Manually'])

    product_data = None

    with tab_url:
        url = st.text_input(
            'Product URL',
            placeholder='https://www.flipkart.com/product-link or https://www.amazon.in/product-link',
        )
        if st.button('Analyze', key='url_btn', type='primary'):
            if url.strip():
                with st.spinner('Scraping product data...'):
                    product_data = scrape_product(url.strip())
                if product_data.get('scrape_failed'):
                    st.warning(
                        f"Scraping was blocked: {product_data.get('fail_reason', '')}\n\n"
                        "Please use the manual entry tab instead."
                    )
                    product_data = None

    with tab_manual:
        col1, col2 = st.columns(2)
        with col1:
            m_name = st.text_input('Product Name', placeholder='e.g. Samsung Galaxy S24')
            m_brand = st.text_input('Brand', placeholder='e.g. Samsung')
            m_category = st.text_input('Category', placeholder='e.g. Electronics')
        with col2:
            m_price = st.number_input('Current Price (₹)', min_value=1.0, value=10000.0, step=100.0)
            m_original = st.number_input('Original/Retail Price (₹)', min_value=1.0, value=12000.0, step=100.0)
            m_rating = st.slider('Rating', 1.0, 5.0, 4.0, 0.1)

        if st.button('Predict Price', key='manual_btn', type='primary'):
            discount_pct = max(0, (1 - m_price / m_original) * 100) if m_original > 0 else 0
            product_data = {
                'scrape_failed': False,
                'name': m_name,
                'brand': m_brand,
                'category': m_category,
                'current_price': m_price,
                'original_price': m_original,
                'discount_pct': round(discount_pct, 1),
                'rating': m_rating,
                'num_reviews': 0,
                'description': '',
                'source': 'manual',
            }

    # ── Render results ────────────────────────────────
    if product_data and not product_data.get('scrape_failed'):
        st.markdown('---')

        # Product details card
        st.subheader(product_data.get('name', 'Unknown Product'))
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric('Current Price', f"₹{product_data.get('current_price', 0):,.0f}")
        col_b.metric('Original Price', f"₹{product_data.get('original_price', 0):,.0f}")
        col_c.metric('Discount', f"{product_data.get('discount_pct', 0):.1f}%")
        col_d.metric('Rating', f"{product_data.get('rating', 0):.1f} / 5")

        if product_data.get('brand'):
            st.markdown(f"**Brand:** {product_data['brand']}  |  **Category:** {product_data.get('category', 'Unknown')}  |  **Source:** {product_data.get('source', '')}")

        # ── Fair Price Prediction ─────────────
        st.markdown('---')
        st.subheader('Predicted Fair Price')

        predictor = load_predictor()
        pred = predictor.predict(
            brand=product_data.get('brand', 'Unknown'),
            category=product_data.get('category', 'Unknown'),
            rating=product_data.get('rating', 3.0),
            retail_price=product_data.get('original_price', product_data.get('current_price', 0)),
            discount_percentage=product_data.get('discount_pct', 0),
        )

        pc1, pc2, pc3 = st.columns(3)
        pc1.markdown(f"""<div class="metric-box">
            <h3>Predicted Fair Price</h3>
            <p style="color:#34d399;">₹{pred['predicted_fair_price']:,.0f}</p>
        </div>""", unsafe_allow_html=True)
        pc2.markdown(f"""<div class="metric-box">
            <h3>Expected Discount</h3>
            <p style="color:#fbbf24;">{pred['expected_discount']:.1f}%</p>
        </div>""", unsafe_allow_html=True)
        pc3.markdown(f"""<div class="metric-box">
            <h3>Confidence Score</h3>
            <p>{pred['confidence_score']:.0%}</p>
        </div>""", unsafe_allow_html=True)

        if pred.get('error'):
            st.info(pred['error'])

        # ── Time-Series Category Forecast ─────
        st.markdown('---')
        st.subheader('Future Price Prediction')

        ts_predictor = load_ts_predictor()
        category = product_data.get('category', 'Unknown')
        current_price = product_data.get('current_price', 0)

        if current_price > 0:
            ts_result = ts_predictor.predict(current_price, category, 30)

            if ts_result and not ts_result.get('error'):
                fc1, fc2, fc3, fc4 = st.columns(4)
                fc1.metric('7-Day Forecast', f"₹{ts_result.get('predicted_7d', 0):,.0f}",
                           delta=f"{ts_result.get('week_change_pct', 0):+.1f}%")
                fc2.metric('30-Day Forecast', f"₹{ts_result.get('predicted_30d', 0):,.0f}",
                           delta=f"{ts_result.get('month_change_pct', 0):+.1f}%")
                trend_icons = {'up': 'Rising', 'down': 'Falling', 'stable': 'Stable'}
                fc3.metric('Trend', trend_icons.get(ts_result.get('trend', ''), 'Unknown'))
                fc4.metric('Model Used', ts_result.get('model_used', 'N/A'))

                # Price Trend Chart
                if ts_result.get('historical', {}).get('dates'):
                    import plotly.graph_objects as go
                    hist_dates = ts_result['historical']['dates']
                    hist_prices = [p for p in ts_result['historical']['prices'] if p is not None]
                    hist_dates_valid = hist_dates[-len(hist_prices):]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hist_dates_valid, y=hist_prices,
                        mode='lines', name='Historical',
                        line=dict(color='#818cf8', width=2),
                    ))
                    fig.add_trace(go.Scatter(
                        x=ts_result['dates'], y=ts_result['predictions'],
                        mode='lines', name='Predicted',
                        line=dict(color='#34d399', width=2, dash='dash'),
                    ))
                    if ts_result.get('upper_bound'):
                        fig.add_trace(go.Scatter(
                            x=ts_result['dates'], y=ts_result['upper_bound'],
                            mode='lines', name='Upper Bound',
                            line=dict(color='rgba(251,191,36,0.3)', width=1),
                            showlegend=False,
                        ))
                        fig.add_trace(go.Scatter(
                            x=ts_result['dates'], y=ts_result['lower_bound'],
                            mode='lines', name='Lower Bound',
                            line=dict(color='rgba(56,189,248,0.3)', width=1),
                            fill='tonexty',
                            fillcolor='rgba(129,140,248,0.08)',
                            showlegend=False,
                        ))

                    fig.update_layout(
                        title='Historical Price vs Predicted Price',
                        template='plotly_dark',
                        height=450,
                        hovermode='x unified',
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Model performance
                if ts_result.get('model_scores'):
                    with st.expander('ML Model Performance'):
                        model_data = []
                        for name, scores in ts_result['model_scores'].items():
                            model_data.append({
                                'Model': name,
                                'R² Score': scores['r2'],
                                'MAE': scores['mae'],
                                'RMSE': scores['rmse'],
                            })
                        if model_data:
                            st.dataframe(pd.DataFrame(model_data), hide_index=True, use_container_width=True)

        # ── Festival Impact ───────────────────
        st.markdown('---')
        st.subheader('Festival Price Predictions')

        fest_analyzer = load_festival_analyzer()
        fest_results = fest_analyzer.estimate_festival_price(
            current_price, category,
        )

        if fest_results:
            cols = st.columns(3)
            for i, f in enumerate(fest_results[:6]):
                with cols[i % 3]:
                    disc_text = f"Expected Discount: {f['expected_discount_pct']:.1f}%" if f.get('data_available') else "Estimated ~10%"
                    conf = f.get('confidence', 'low')
                    conf_emoji = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}.get(conf, '⚪')
                    st.markdown(f"""<div class="festival-card">
                        <h4>{f['festival']}</h4>
                        <div class="price">₹{f['expected_price']:,.0f}</div>
                        <div class="discount">{disc_text}</div>
                        <div style="font-size:0.8rem; color:#94a3b8; margin-top:0.3rem;">
                            Confidence: {conf_emoji} {conf.title()}
                        </div>
                    </div>""", unsafe_allow_html=True)

            # Festival comparison chart
            fig = festival_impact_chart(fest_results, title=f'Festival Price Impact — {category}')
            st.plotly_chart(fig, use_container_width=True)

        # ── Similar Products ──────────────────
        sim_engine = load_similarity_engine()
        similar = sim_engine.find_similar(
            product_data.get('name', ''), category, current_price, top_n=6,
        )
        if similar:
            st.markdown('---')
            st.subheader('Similar Products')
            cols = st.columns(3)
            for i, sp in enumerate(similar):
                with cols[i % 3]:
                    disc = 0
                    if sp.get('original_price') and sp.get('price'):
                        disc = round((1 - sp['price'] / sp['original_price']) * 100)
                    st.markdown(f"""<div class="festival-card">
                        <h4>{sp['name'][:60]}</h4>
                        <div style="color:#94a3b8;font-size:0.8rem;">{sp.get('brand','')}</div>
                        <div class="price">₹{sp['price']:,.0f}</div>
                        {f'<div class="discount">{disc}% off</div>' if disc > 0 else ''}
                    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#   PAGE 2: Price Trend & Forecast
# ══════════════════════════════════════════════════════════════
elif page == 'Price Trend & Forecast':
    st.title('Price Trend & Forecast')
    st.markdown('Stock-market style price analysis with ARIMA and Prophet models.')

    forecaster = load_forecaster()
    categories = forecaster.get_categories()

    if not categories:
        st.warning('No category data available. Run prepare_dashboard.py first.')
    else:
        col_sel, col_days = st.columns([3, 1])
        with col_sel:
            selected_cat = st.selectbox('Select Category', categories)
        with col_days:
            days_ahead = st.selectbox('Forecast Horizon', [7, 30, 90], index=1)

        dates, prices = forecaster.get_category_series(selected_cat)

        if dates is not None and len(dates) > 0:
            # Stock-market style charts
            st.subheader('Price Trend with Moving Averages')
            fig_ma = price_trend_with_ma(dates, prices, title=f'{selected_cat} — Price Trend')
            st.plotly_chart(fig_ma, use_container_width=True)

            st.subheader('Bollinger Bands')
            fig_bb = bollinger_bands_chart(dates, prices, title=f'{selected_cat} — Bollinger Bands')
            st.plotly_chart(fig_bb, use_container_width=True)

            col_v, col_s = st.columns(2)
            with col_v:
                st.subheader('Price Volatility')
                fig_vol = price_volatility_chart(dates, prices, title=f'{selected_cat} — Volatility')
                st.plotly_chart(fig_vol, use_container_width=True)
            with col_s:
                st.subheader('Seasonal Patterns')
                fig_season = seasonal_trends_chart(dates, prices, title=f'{selected_cat} — Seasonal')
                st.plotly_chart(fig_season, use_container_width=True)

            # ARIMA / Prophet Forecasts
            st.markdown('---')
            st.subheader(f'{days_ahead}-Day Price Forecast')

            with st.spinner('Running ARIMA forecast...'):
                try:
                    arima_result = forecaster.forecast_arima(dates, prices, days_ahead)
                except Exception as e:
                    arima_result = None
                    st.warning(f'ARIMA forecast failed: {e}')

            with st.spinner('Running Prophet forecast...'):
                try:
                    prophet_result = forecaster.forecast_prophet(dates, prices, days_ahead)
                except Exception as e:
                    prophet_result = None
                    st.warning(f'Prophet forecast failed: {e}')

            f_col1, f_col2 = st.columns(2)

            if arima_result and not arima_result.get('error'):
                with f_col1:
                    st.markdown(f"**ARIMA** (order={arima_result.get('order', '')})")
                    fig_arima = forecast_chart(
                        dates[-90:], prices[-90:],
                        arima_result['dates'], arima_result['predicted'],
                        arima_result.get('lower'), arima_result.get('upper'),
                        model_name='ARIMA',
                        title=f'ARIMA {days_ahead}-Day Forecast',
                    )
                    st.plotly_chart(fig_arima, use_container_width=True)

                    # Forecast summary
                    pred_arr = arima_result['predicted']
                    st.metric('Predicted Price (End)', f"₹{pred_arr[-1]:,.0f}")

            if prophet_result and not prophet_result.get('error'):
                with f_col2:
                    st.markdown('**Facebook Prophet**')
                    fig_prophet = forecast_chart(
                        dates[-90:], prices[-90:],
                        prophet_result['dates'], prophet_result['predicted'],
                        prophet_result.get('lower'), prophet_result.get('upper'),
                        model_name='Prophet',
                        title=f'Prophet {days_ahead}-Day Forecast',
                    )
                    st.plotly_chart(fig_prophet, use_container_width=True)

                    pred_arr = prophet_result['predicted']
                    st.metric('Predicted Price (End)', f"₹{pred_arr[-1]:,.0f}")


# ══════════════════════════════════════════════════════════════
#   PAGE 3: Festival Impact
# ══════════════════════════════════════════════════════════════
elif page == 'Festival Impact':
    st.title('Festival Price Impact Analysis')
    st.markdown('Analyze how Indian festivals and sale events affect product prices.')

    fest_analyzer = load_festival_analyzer()
    available_cats = fest_analyzer.get_categories()

    if not available_cats:
        st.warning('No festival impact data available. Run prepare_dashboard.py first.')
    else:
        selected_cat = st.selectbox('Select Category', available_cats)
        price_input = st.number_input('Current Product Price (₹)', min_value=100.0, value=10000.0, step=500.0)

        st.markdown('---')

        # Festival estimates
        results = fest_analyzer.estimate_festival_price(price_input, selected_cat)

        if results:
            st.subheader('Expected Festival Prices')

            # Table view
            table_data = []
            for r in results:
                table_data.append({
                    'Festival': r['festival'],
                    'Current Price': f"₹{r['current_price']:,.0f}",
                    'Expected Price': f"₹{r['expected_price']:,.0f}",
                    'Discount': f"{r['expected_discount_pct']:.1f}%",
                    'Impact': f"{r.get('impact_pct', 0):+.1f}%",
                    'Confidence': r.get('confidence', 'low').title(),
                })
            st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)

            # Visual cards
            cols = st.columns(3)
            for i, r in enumerate(results):
                with cols[i % 3]:
                    conf_emoji = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}.get(r.get('confidence', ''), '⚪')
                    st.markdown(f"""<div class="festival-card">
                        <h4>{r['festival']}</h4>
                        <div style="font-size:0.8rem;color:#94a3b8;">{r.get('description', '')}</div>
                        <div class="price">₹{r['expected_price']:,.0f}</div>
                        <div class="discount">Expected Discount: {r['expected_discount_pct']:.1f}%</div>
                        <div style="font-size:0.8rem;color:#94a3b8;margin-top:0.3rem;">
                            Confidence: {conf_emoji} {r.get('confidence','').title()}
                        </div>
                    </div>""", unsafe_allow_html=True)

            # Chart
            st.markdown('---')
            fig = festival_impact_chart(results, title=f'Festival Price Impact — {selected_cat}')
            st.plotly_chart(fig, use_container_width=True)

        # Raw impact data
        with st.expander('Raw Festival Impact Data'):
            impact_data = fest_analyzer.get_festival_impact(selected_cat)
            if impact_data:
                for fest, data in impact_data.items():
                    st.markdown(f"**{fest}**: Impact = {data['impact_pct']:+.2f}% ({data['direction']})")
            else:
                st.info('No impact data available for this category.')


# ══════════════════════════════════════════════════════════════
#   PAGE 4: Dataset Analytics
# ══════════════════════════════════════════════════════════════
elif page == 'Dataset Analytics':
    st.title('Dataset Analytics')
    st.markdown('Statistical insights and visualizations from the Flipkart product dataset.')

    analyzer = load_dataset_analyzer()

    # Summary cards
    stats = analyzer.summary_stats()
    if stats:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric('Total Products', f"{stats.get('total_products', 0):,}")
        c2.metric('Categories', f"{stats.get('total_categories', 0)}")
        c3.metric('Brands', f"{stats.get('total_brands', 0)}")
        c4.metric('Avg Price', f"₹{stats.get('avg_price', 0):,.0f}")
        c5.metric('Avg Discount', f"{stats.get('avg_discount_pct', 0):.1f}%")

    st.markdown('---')

    # Charts in tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        'Avg Price by Category',
        'Price Distribution',
        'Top Expensive',
        'Discount Distribution',
        'Brand vs Rating',
        'Category Trends',
    ])

    with tab1:
        st.plotly_chart(analyzer.avg_price_by_category(), use_container_width=True)
        st.plotly_chart(analyzer.median_price_chart(), use_container_width=True)

    with tab2:
        st.plotly_chart(analyzer.price_distribution(), use_container_width=True)

    with tab3:
        st.plotly_chart(analyzer.top_expensive_categories(), use_container_width=True)

    with tab4:
        st.plotly_chart(analyzer.discount_distribution(), use_container_width=True)

    with tab5:
        st.plotly_chart(analyzer.brand_vs_rating(), use_container_width=True)

    with tab6:
        forecaster = load_forecaster()
        cats = forecaster.get_categories()
        if cats:
            selected = st.multiselect('Select categories to compare', cats, default=cats[:5])
            if selected:
                st.plotly_chart(
                    analyzer.category_price_trends(selected),
                    use_container_width=True,
                )
