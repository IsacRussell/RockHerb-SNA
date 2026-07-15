import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.colors as mcolors
import os
import math
import random
import joblib
from itertools import combinations
from pyvis.network import Network
import plotly.graph_objects as go
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. PLATFORM CONFIGURATION & UI SETUP (LIGHT MODE)
# ============================================================================
st.set_page_config(page_title="Social Network Analysis & Churn Prediction", layout="wide")

# Updated to Light Mode Palette
COLOR_BG = "#ffffff"          # Pure White background
COLOR_SURFACE = "#f8f9fa"     # Light gray for sidebars and hover states
COLOR_TEXT = "#121212"        # Near-black for highly readable text
COLOR_ACCENT = "#d32f2f"      # Deep red for accents and high-risk nodes
COLOR_EDGE = "#9e9e9e"        # Medium gray for network lines

st.markdown(f"""
    <style>
    /* Main App Background and Top Header */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{ 
        background-color: {COLOR_BG} !important; 
        color: {COLOR_TEXT} !important; 
    }}
    
    /* Sidebar Background */
    [data-testid="stSidebar"] {{ 
        background-color: {COLOR_SURFACE} !important; 
        border-right: 1px solid #e0e0e0 !important;
    }}
    
    /* Force text colors across common elements */
    h1, h2, h3, h4, h5, h6, p, span, div, label {{
        color: {COLOR_TEXT} !important;
    }}
    
    /* Metric Values */
    [data-testid="stMetricValue"] {{ 
        font-size: 2rem; 
        font-weight: bold; 
        color: {COLOR_ACCENT} !important; 
    }}
    
    /* Tabs styling */
    button[data-baseweb="tab"] {{
        background-color: {COLOR_BG} !important;
        color: {COLOR_TEXT} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# 2. DATA INGESTION ENGINE
# ============================================================================
@st.cache_data
def load_and_clean_data(file_path):
    try:
        if str(file_path).lower().endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        required_cols = ['Order ID', 'Seller Name', 'Product', 'Phone', 'Order Date', 'Grand Total', 'State']
        
        if 'Quantity X Product' in df.columns:
            required_cols.append('Quantity X Product')
            
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            return None

        df = df.dropna(subset=['Order ID', 'Seller Name', 'Phone', 'Order Date', 'Grand Total', 'State'])
        
        df['Seller Name'] = df['Seller Name'].astype(str).str.strip().str.replace(r'["\']', '', regex=True)
        
        if 'Quantity X Product' in df.columns:
            df['Base_Product'] = df['Quantity X Product'].astype(str).str.strip().str.replace(r'["\']', '', regex=True)
        else:
            df['Base_Product'] = df['Product'].astype(str).str.strip().str.replace(r'["\']', '', regex=True)
            
        df['Phone'] = df['Phone'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        
        df['State'] = df['State'].astype(str).str.strip().str.upper() 
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
        df = df.dropna(subset=['Order Date'])

        return df
    except Exception as e:
        st.error(f"Data Loading Error: {e}")
        return None

def explode_products(df):
    temp_df = df.copy()
    temp_df['Clean_Product'] = temp_df['Base_Product'].astype(str).str.split(',')
    temp_df = temp_df.explode('Clean_Product')
    temp_df['Clean_Product'] = temp_df['Clean_Product'].str.strip().str.upper()
    temp_df['Clean_Product'] = temp_df['Clean_Product'].str.replace(r'^\d+[xX]?\s*', '', regex=True)
    temp_df = temp_df[temp_df['Clean_Product'] != ""]
    temp_df = temp_df[temp_df['Clean_Product'] != "NAN"]
    temp_df = temp_df[temp_df['Clean_Product'].notna()]
    return temp_df

# ============================================================================
# 3. DYNAMIC SELLER PROFILE & SPECIFIC SELLER NETWORK
# ============================================================================
def show_seller_profile(seller_name, data):
    seller_data = data[data['Seller Name'] == seller_name]
    
    total_rev = seller_data['Grand Total'].sum()
    unique_cust = seller_data['Phone'].nunique()
    total_orders = seller_data['Order ID'].nunique()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Revenue", f"RM {total_rev:,.2f}")
    col2.metric("Customers", unique_cust)
    col3.metric("Orders", total_orders)
    
    st.markdown("---")
    st.markdown("**Customer List & Order Frequency**")
    
    cust_list = seller_data.groupby('Phone').agg(
        Orders=('Order ID', 'nunique'),
        Total_Spend=('Grand Total', 'sum')
    ).reset_index().sort_values(by='Total_Spend', ascending=False)
    
    st.dataframe(cust_list, width='stretch', hide_index=True)

def draw_single_seller_network(seller_name, df):
    sdf = df[df['Seller Name'] == seller_name]
    customers = sdf['Phone'].unique()
    
    if len(customers) > 300:
        customers = customers[:300]
        st.caption(f"*(Visualizing top 300 unique customers for {seller_name} to maintain performance)*")

    G = nx.Graph()
    G.add_node(seller_name)
    for cust in customers:
        G.add_edge(seller_name, str(cust))
        
    pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42)

    net = Network(height='450px', width='100%', bgcolor=COLOR_BG, font_color=COLOR_TEXT, directed=False)
    
    for node in G.nodes():
        x = float(pos[node][0]) * 3000
        y = float(pos[node][1]) * 3000
        
        if node == seller_name:
            net.add_node(
                node, label=node, size=75, x=x, y=y,
                color={"background": COLOR_ACCENT, "border": COLOR_ACCENT}
            )
        else:
            net.add_node(
                node, label=node, size=25, x=x, y=y,
                color={"background": "#e0e0e0", "border": COLOR_EDGE}
            )
            net.add_edge(seller_name, node, color={"color": COLOR_EDGE, "opacity": 0.6})

    net.set_options(f"""
    var options = {{
      "physics": {{ "enabled": false }},
      "nodes": {{ "font": {{ "color": "{COLOR_TEXT}", "size": 12, "strokeWidth": 2, "strokeColor": "{COLOR_BG}" }}, "shape": "dot" }},
      "edges": {{ "smooth": false }}
    }}
    """)
    
    html_file = f"net_single_seller.html"
    net.save_graph(html_file)
    
    with open(html_file, 'r', encoding='utf-8') as f:
        html_source = f.read()
    
    components.html(html_source, height=470)
    os.remove(html_file)

# ============================================================================
# 4. UNIVERSAL UNIPARTITE SNA BUILDER
# ============================================================================
def build_network(df, node_col, edge_group_col, max_cap=100, include_isolated=False):
    G = nx.Graph()
    
    if include_isolated:
        all_unique_nodes = df[node_col].unique()
        G.add_nodes_from(all_unique_nodes)
        
    grouped = df.groupby(edge_group_col)[node_col].unique()
    
    edge_weights = {}
    for items in grouped:
        if len(items) > 1:
            if len(items) > max_cap:
                items = np.random.choice(items, max_cap, replace=False)
            for n1, n2 in combinations(sorted(items), 2):
                edge_weights[(n1, n2)] = edge_weights.get((n1, n2), 0) + 1
                
    weighted_edges = [(k[0], k[1], v) for k, v in edge_weights.items()]
    G.add_weighted_edges_from(weighted_edges)
    return G

def draw_spiderweb_network(G, title, prefix):
    if len(G.nodes()) == 0:
        st.warning("Not enough overlapping data to draw connections.")
        return
        
    if len(G.nodes()) > 250:
        top_nodes = sorted(dict(G.degree()), key=dict(G.degree()).get, reverse=True)[:250]
        G = G.subgraph(top_nodes).copy()
        
    net = Network(height='600px', width='100%', bgcolor=COLOR_BG, font_color=COLOR_TEXT, directed=False)
    pos = nx.spring_layout(G, k=1.5, iterations=200, seed=42)
    
    degree_dict = dict(G.degree())
    max_degree = max(degree_dict.values()) if degree_dict else 1
    min_degree = min(degree_dict.values()) if degree_dict else 1
    
    cmap = mcolors.LinearSegmentedColormap.from_list("connection_heatmap", [COLOR_ACCENT, "#ffb3b3", "#ffffff"])
    
    for node in G.nodes():
        deg = degree_dict.get(node, 0)
        
        if deg == 0:
            node_color = "#000000"
            size = 40
        else:
            heat_ratio = (deg - min_degree) / (max_degree - min_degree) if max_degree > min_degree else 0.5
            node_color = mcolors.to_hex(cmap(heat_ratio))
            size = 25 + (62.5 * heat_ratio) 
            
        x = float(pos[node][0]) * 7500
        y = float(pos[node][1]) * 7500
        
        net.add_node(
            str(node), 
            label=f"{str(node)[:10]}.." if (deg > 0 and heat_ratio < 0.5) else f"{str(node)[:15]}",
            title=f"{prefix} {node}\nConnections: {deg}", 
            size=size, x=x, y=y,
            color={"background": node_color, "border": COLOR_EDGE} 
        )
        
    for u, v, d in G.edges(data=True):
        net.add_edge(str(u), str(v), value=d.get('weight', 1), color={"color": COLOR_EDGE, "opacity": 0.4})
        
    net.set_options(f"""
    var options = {{
      "physics": {{ "enabled": false }},
      "nodes": {{ "font": {{ "color": "{COLOR_TEXT}", "size": 12, "strokeWidth": 2, "strokeColor": "{COLOR_BG}" }}, "shape": "dot" }},
      "edges": {{ "smooth": false }} 
    }}
    """)
    
    html_file = f"net_{title}.html"
    net.save_graph(html_file)
    
    with open(html_file, 'r', encoding='utf-8') as f:
        html_source = f.read()
        
    components.html(html_source, height=620)
    os.remove(html_file)

# ============================================================================
# 5. GRAVITATIONAL WEB (State-Customer)
# ============================================================================
def draw_plotly_state_customer(df):
    unique_pairs = df[['State', 'Phone']].drop_duplicates()
    if unique_pairs.empty: return

    state_cust_counts = unique_pairs['State'].value_counts()
    max_state_count = state_cust_counts.max() if not state_cust_counts.empty else 1
    sorted_states = state_cust_counts.index.tolist()
    
    golden_angle = math.pi * (3 - math.sqrt(5))
    
    state_coords = {}
    for i, s in enumerate(sorted_states):
        radius = 75 * math.sqrt(i) 
        angle = i * golden_angle
        state_coords[s] = (radius * math.cos(angle), radius * math.sin(angle))
        
    cust_to_states = unique_pairs.groupby('Phone')['State'].apply(list).to_dict()

    edge_x, edge_y = [], []
    s_x, s_y, s_labels, s_hover, s_size = [], [], [], [], []
    c_x, c_y, c_hover = [], [], []

    for s in sorted_states:
        x, y = state_coords[s]
        s_x.append(x)
        s_y.append(y)
        count = state_cust_counts.get(s, 0)
        s_size.append(18 + ((count / max_state_count) * 50))
        s_labels.append(str(s)) 
        s_hover.append(f"<b>State:</b> {s}<br><b>Customers:</b> {count}")

    for cust, linked_states in cust_to_states.items():
        if len(linked_states) == 1:
            sx, sy = state_coords[linked_states[0]]
            r = random.uniform(10, 50) 
            theta = random.uniform(0, 2 * math.pi)
            cx = sx + r * math.cos(theta)
            cy = sy + r * math.sin(theta)
        else:
            cx = sum(state_coords[s][0] for s in linked_states) / len(linked_states)
            cy = sum(state_coords[s][1] for s in linked_states) / len(linked_states)
            cx += random.uniform(-7.5, 7.5)
            cy += random.uniform(-7.5, 7.5)
            
        c_x.append(cx)
        c_y.append(cy)
        c_hover.append(f"<b>Customer:</b> {cust}<br><b>States:</b> {', '.join([str(s) for s in linked_states])}")
        
        for s in linked_states:
            sx, sy = state_coords[s]
            edge_x.extend([sx, cx, None])
            edge_y.extend([sy, cy, None])

    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(width=0.25, color=COLOR_EDGE),
        hoverinfo='none', showlegend=False
    ))

    fig.add_trace(go.Scattergl(
        x=c_x, y=c_y, mode='markers',
        marker=dict(size=4, color="#666666", line=dict(width=0.5, color=COLOR_BG)),
        customdata=c_hover, hovertemplate="%{customdata}<extra></extra>", name='Customers'
    ))

    fig.add_trace(go.Scattergl(
        x=s_x, y=s_y, mode='markers+text',
        marker=dict(size=s_size, color=COLOR_ACCENT, line=dict(width=2, color=COLOR_TEXT)),
        text=s_labels, textposition="bottom center",
        textfont=dict(color=COLOR_TEXT, size=13, family="Arial Black"),
        customdata=s_hover, hovertemplate="%{customdata}<extra></extra>", name='States'
    ))

    fig.update_layout(
        plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG, font=dict(color=COLOR_TEXT),
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=750, dragmode='pan',
        hoverlabel=dict(bgcolor=COLOR_SURFACE, font_size=13, font_family="Arial", bordercolor=COLOR_ACCENT),
        hovermode='closest'
    )
    st.plotly_chart(fig, width='stretch', config={'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False})

# ============================================================================
# 6. GRAVITATIONAL WEB (Seller-Customer)
# ============================================================================
def draw_plotly_ecosystem(df):
    unique_pairs = df[['Seller Name', 'Phone']].drop_duplicates()
    if unique_pairs.empty: return

    seller_cust_counts = unique_pairs['Seller Name'].value_counts()
    max_seller_count = seller_cust_counts.max() if not seller_cust_counts.empty else 1
    sorted_sellers = seller_cust_counts.index.tolist()
    
    golden_angle = math.pi * (3 - math.sqrt(5))
    seller_coords = {}
    for i, s in enumerate(sorted_sellers):
        radius = 40 * math.sqrt(i) 
        angle = i * golden_angle
        seller_coords[s] = (radius * math.cos(angle), radius * math.sin(angle))
        
    cust_to_sellers = unique_pairs.groupby('Phone')['Seller Name'].apply(list).to_dict()

    edge_x, edge_y = [], []
    s_x, s_y, s_hover, s_size = [], [], [], []
    c_x, c_y, c_hover = [], [], []

    for s in sorted_sellers:
        x, y = seller_coords[s]
        s_x.append(x)
        s_y.append(y)
        count = seller_cust_counts.get(s, 0)
        s_size.append(12 + ((count / max_seller_count) * 45))
        s_hover.append(f"<b>Seller:</b> {s}<br><b>Customers:</b> {count}")

    for cust, linked_sellers in cust_to_sellers.items():
        if len(linked_sellers) == 1:
            sx, sy = seller_coords[linked_sellers[0]]
            r = random.uniform(10, 30) 
            theta = random.uniform(0, 2 * math.pi)
            cx = sx + r * math.cos(theta)
            cy = sy + r * math.sin(theta)
        else:
            cx = sum(seller_coords[s][0] for s in linked_sellers) / len(linked_sellers)
            cy = sum(seller_coords[s][1] for s in linked_sellers) / len(linked_sellers)
            cx += random.uniform(-7.5, 7.5)
            cy += random.uniform(-7.5, 7.5)
            
        c_x.append(cx)
        c_y.append(cy)
        c_hover.append(f"<b>Customer:</b> {cust}<br><b>Buys From:</b> {', '.join([str(s)[:15] for s in linked_sellers])}")
        
        for s in linked_sellers:
            sx, sy = seller_coords[s]
            edge_x.extend([sx, cx, None])
            edge_y.extend([sy, cy, None])

    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(width=0.35, color=COLOR_EDGE), hoverinfo='none', showlegend=False
    ))

    fig.add_trace(go.Scattergl(
        x=c_x, y=c_y, mode='markers',
        marker=dict(size=6, color="#666666", line=dict(width=0.5, color=COLOR_BG)),
        customdata=c_hover, hovertemplate="%{customdata}<extra></extra>", name='Customers'
    ))

    fig.add_trace(go.Scattergl(
        x=s_x, y=s_y, mode='markers',
        marker=dict(size=s_size, color=COLOR_ACCENT, line=dict(width=1.5, color=COLOR_TEXT)),
        customdata=s_hover, hovertemplate="%{customdata}<extra></extra>", name='Sellers'
    ))

    fig.update_layout(
        plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG, font=dict(color=COLOR_TEXT),
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=750, dragmode='pan',
        hoverlabel=dict(bgcolor=COLOR_SURFACE, font_size=13, font_family="Arial", bordercolor=COLOR_ACCENT),
        hovermode='closest'
    )
    st.plotly_chart(fig, width='stretch', config={'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False})

# ============================================================================
# 7. CUSTOMER CHURN ENGINE & ML PREDICTION (INFERENCE ONLY)
# ============================================================================
def calculate_churn_rfm(df):
    if df.empty: return pd.DataFrame(), 0.0
        
    snapshot_date = df['Order Date'].max() + pd.Timedelta(days=1)
    
    churn_df = df.groupby('Phone').agg(
        Frequency=('Order ID', 'nunique'),
        Last_Purchase=('Order Date', 'max'),
        First_Purchase=('Order Date', 'min'),
        Total_Spend=('Grand Total', 'sum')
    ).reset_index()
    
    churn_df['Recency_Days'] = (snapshot_date - churn_df['Last_Purchase']).dt.days
    churn_df['Account_Age_Days'] = (snapshot_date - churn_df['First_Purchase']).dt.days
    churn_df['Avg_Order_Value'] = churn_df['Total_Spend'] / churn_df['Frequency']
    
    def assign_risk(row):
        if row['Frequency'] >= 3 and row['Recency_Days'] <= 60:
            return "Active & Loyal"
        elif row['Frequency'] == 1 and row['Recency_Days'] > 90:
            return "High Risk (One-Off)"
        elif row['Frequency'] >= 2 and row['Recency_Days'] > 90:
            return "Churned (Lost Repeat)"
        else:
            return "At Risk (Needs Nurturing)"
            
    churn_df['Churn Risk Profile'] = churn_df.apply(assign_risk, axis=1)
    
    # ---------------------------------------------------------------------
    # LOAD PRE-TRAINED RANDOM FOREST MODEL (Memory Safe)
    # ---------------------------------------------------------------------
    model_path = 'churn_model.pkl'
    
    if os.path.exists(model_path):
        # Load the saved brain
        rf_model = joblib.load(model_path)
        
        # Prepare the exact same features the model was trained on
        X_predict = churn_df[['Frequency', 'Total_Spend', 'Account_Age_Days', 'Avg_Order_Value']]
        
        # Ask the model for the probability percentage of churn
        churn_df['AI_Churn_Probability'] = rf_model.predict_proba(X_predict)[:, 1]
        
        # Calculate the overall predicted churn rate mathematically based on AI
        predicted_churners = (churn_df['AI_Churn_Probability'] > 0.5).sum()
        overall_churn_rate = (predicted_churners / len(churn_df)) * 100
        
        # Format for UI display
        churn_df['AI_Churn_Probability'] = (churn_df['AI_Churn_Probability'] * 100).round(1).astype(str) + "%"
        
    else:
        # Fallback to standard math if the .pkl file isn't uploaded yet
        st.warning("⚠️ Pre-trained 'churn_model.pkl' not found. Falling back to mathematical RFM.")
        at_risk_count = churn_df[churn_df['Churn Risk Profile'].isin(['High Risk (One-Off)', 'Churned (Lost Repeat)'])].shape[0]
        overall_churn_rate = (at_risk_count / len(churn_df)) * 100 if len(churn_df) > 0 else 0.0
        churn_df['AI_Churn_Probability'] = "N/A"
    
    return churn_df, overall_churn_rate

# ============================================================================
# 8. MAIN DASHBOARD INTERFACE
# ============================================================================
def main():
    st.title("Social Network Analysis & Churn Prediction")
    
    DATA_FILE = "RockHerb_Full.xlsx"
    
    if not os.path.exists(DATA_FILE):
        st.error(f"Dataset '{DATA_FILE}' not found. Please make sure '{DATA_FILE}' is inside the same folder as this script.")
        return
        
    with st.spinner("Processing dataset..."):
        raw_df = load_and_clean_data(DATA_FILE)
        
    if raw_df is None: return

    st.sidebar.markdown("---")
    st.sidebar.subheader("Time Filters")
    
    available_years = sorted(raw_df['Order Date'].dt.year.unique().astype(int).tolist(), reverse=True)
    selected_year = st.sidebar.selectbox("Filter by Year", ["All Years"] + available_years)
    selected_quarter = "All Quarters"
    
    if selected_year != "All Years":
        df = raw_df[raw_df['Order Date'].dt.year == selected_year]
        selected_quarter = st.sidebar.selectbox("Filter by Quarter", ["All Quarters", "Q1", "Q2", "Q3", "Q4"])
        if selected_quarter != "All Quarters":
            q_map = {"Q1": [1, 2, 3], "Q2": [4, 5, 6], "Q3": [7, 8, 9], "Q4": [10, 11, 12]}
            df = df[df['Order Date'].dt.month.isin(q_map[selected_quarter])]
    else:
        df = raw_df.copy()
        
    # Display the Dynamic Filters below the title
    if selected_year != "All Years":
        q_text = f" - {selected_quarter}" if selected_quarter != "All Quarters" else ""
        st.markdown(f"**Viewing Data For:** {selected_year}{q_text}")
        
    if df.empty:
        st.error("No data available for the selected timeframe.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Orders", f"{df['Order ID'].nunique():,}")
    c2.metric("Unique Customers", f"{df['Phone'].nunique():,}")
    c3.metric("Total Sellers", f"{df['Seller Name'].nunique():,}")
    c4.metric("Gross Revenue", f"RM {df['Grand Total'].sum():,.2f}")
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "SNA: Seller Performance", 
        "SNA: Product Lifetime", 
        "SNA: Customer Loyalty", 
        "SNA: State-Customer Distribution",
        "SNA: Seller-Customer Ecosystem",
        "Customer Churn"
    ])
    
    with tab1:
        st.subheader("Seller Performance Network")
        seller_list = sorted(df['Seller Name'].unique().tolist())
        selected_seller = st.selectbox("Select a Seller to view specific performance:", ["-- Select --"] + seller_list)
        
        if selected_seller != "-- Select --":
            with st.expander(f"Performance Profile: {selected_seller}", expanded=True):
                show_seller_profile(selected_seller, df)
                st.markdown(f"**Customer Map: {selected_seller}**")
                draw_single_seller_network(selected_seller, df)
            
        st.caption("Logic: Fruchterman-Reingold Force-Directed Algorithm. White nodes indicate hubs. Red indicates fewer connections. Black dots are isolated sellers with 0 shared customers.")
        seller_net = build_network(df, node_col='Seller Name', edge_group_col='Phone', include_isolated=True)
        draw_spiderweb_network(seller_net, "seller", "Seller:")

        # --- SELLER LEADERBOARD ---
        st.markdown("---")
        st.subheader("Seller Leaderboard")
        
        # Aggregate data for the leaderboard
        seller_agg = df.groupby('Seller Name').agg(
            Revenue=('Grand Total', 'sum'),
            Customers=('Phone', 'nunique'),
            Orders=('Order ID', 'nunique')
        ).reset_index()
        
        # Leaderboard Filters
        col_view, col_count, col_sort = st.columns(3)
        view_type = col_view.selectbox("View", ["Top", "Bottom"])
        row_count_str = col_count.selectbox("Number of Sellers", ["5", "10", "20", "50", "All"])
        sort_metric = col_sort.selectbox("Sort By", ["Revenue", "Customers", "Orders"])
        
        # Sort and filter the dataframe
        ascending = True if view_type == "Bottom" else False
        seller_agg = seller_agg.sort_values(by=sort_metric, ascending=ascending)
        
        if row_count_str != "All":
            seller_agg = seller_agg.head(int(row_count_str))
            
        st.dataframe(
            seller_agg, 
            width='stretch', 
            hide_index=True,
            column_config={
                "Revenue": st.column_config.NumberColumn(
                    "Revenue (RM)",
                    format="RM %.2f"
                )
            }
        )

    with tab2:
        st.subheader("Product Lifetime Value Network")
        st.caption("Logic: Fruchterman-Reingold Force-Directed Algorithm. White nodes indicate hubs. Red indicates fewer connections.")
        exploded_df_prod = explode_products(df)
        exploded_df_prod['Month_Str'] = exploded_df_prod['Order Date'].dt.strftime('%Y-%m')
        
        prod_net = build_network(exploded_df_prod, node_col='Clean_Product', edge_group_col='Month_Str', max_cap=80)
        draw_spiderweb_network(prod_net, "product", "Prod:")

    with tab3:
        st.subheader("Customer Loyalty Network")
        st.caption("Logic: Fruchterman-Reingold Force-Directed Algorithm. White nodes indicate hubs. Red indicates fewer connections.")
        exploded_df_cust = explode_products(df)
        
        cust_net = build_network(exploded_df_cust, node_col='Phone', edge_group_col='Clean_Product', max_cap=80) 
        draw_spiderweb_network(cust_net, "customer", "Cust:")

    with tab4:
        st.subheader("State-Customer Distribution Network")
        draw_plotly_state_customer(df)
        
    with tab5:
        st.subheader("Seller-Customer Ecosystem")
        draw_plotly_ecosystem(df)

    with tab6:
        st.subheader("Customer Churn Forecasting (RFM & Machine Learning)")
        churn_df, overall_churn_rate = calculate_churn_rfm(df)
        
        st.metric("Estimated Churn / High Risk Rate", f"{overall_churn_rate:.1f}%", 
                  help="Calculated using either the Machine Learning model (if available) or standard mathematical RFM modeling.",
                  delta_color="inverse")
        
        c_left, c_right = st.columns([1, 2])
        
        with c_left:
            risk_counts = churn_df['Churn Risk Profile'].value_counts().reset_index()
            risk_counts.columns = ['Profile', 'Count']
            
            # Calculate Percentage
            total_customers = risk_counts['Count'].sum()
            risk_counts['Percentage'] = (risk_counts['Count'] / total_customers * 100).round(1)
            
            # Format text label for the bar chart
            text_labels = [f"{val} ({pct}%)" for val, pct in zip(risk_counts['Count'], risk_counts['Percentage'])]
            
            st.dataframe(risk_counts[['Profile', 'Count']], width='stretch', hide_index=True)
            
        with c_right:
            max_y_axis = risk_counts['Count'].max() * 1.2 # Expand Y axis so text fits 
            
            fig = go.Figure(data=[go.Bar(
                x=risk_counts['Profile'],
                y=risk_counts['Count'],
                marker_color=COLOR_ACCENT,
                text=text_labels,
                textposition='outside' # Puts the Value (Percentage) Above the Bar
            )])
            
            fig.update_layout(
                title="Customer Distribution by Risk Profile",
                plot_bgcolor=COLOR_BG,
                paper_bgcolor=COLOR_BG,
                font=dict(color=COLOR_TEXT),
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis=dict(showgrid=False, linecolor=COLOR_EDGE),
                yaxis=dict(showgrid=True, gridcolor=COLOR_SURFACE, linecolor=COLOR_EDGE, range=[0, max_y_axis])
            )
            
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
            
        st.markdown("---")
        st.markdown("**🔍 View Customers by Risk Profile & AI Probability**")
        
        profiles = ["All Profiles"] + churn_df['Churn Risk Profile'].unique().tolist()
        selected_profile = st.selectbox("Select a Risk Level to filter the customer list below:", profiles)
        
        if selected_profile == "All Profiles":
            display_df = churn_df.sort_values(by='Recency_Days', ascending=False)
        else:
            display_df = churn_df[churn_df['Churn Risk Profile'] == selected_profile].sort_values(by='Recency_Days', ascending=False)
            
        # Select the most useful columns for the final display
        columns_to_show = ['Phone', 'Frequency', 'Total_Spend', 'Last_Purchase', 'Recency_Days', 'Churn Risk Profile', 'AI_Churn_Probability']
        st.dataframe(display_df[columns_to_show], width='stretch', hide_index=True)

if __name__ == "__main__":
    main()