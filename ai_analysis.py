import streamlit as st
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import math
import random
from itertools import combinations
from pyvis.network import Network
import plotly.graph_objects as go
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. PLATFORM CONFIGURATION & UI SETUP
# ============================================================================
st.set_page_config(page_title="E-Commerce Intelligence", layout="wide")

COLOR_BG = "#050505"
COLOR_SURFACE = "#121212"
COLOR_TEXT = "#ffffff"
COLOR_ACCENT = "#990000"
COLOR_EDGE = "#333333"

st.markdown(f"""
    <style>
    html, body, [data-testid="stAppViewContainer"] {{ 
        background-color: {COLOR_BG} !important; 
        color: {COLOR_TEXT} !important; 
    }}
    
    [data-testid="stSidebar"] {{ 
        background-color: {COLOR_SURFACE} !important; 
        border-right: 1px solid {COLOR_ACCENT} !important;
    }}
    
    [data-testid="stSidebar"] *, 
    [data-testid="stMarkdownContainer"] *, 
    .stSelectbox label, 
    .stFileUploader label {{
        color: {COLOR_TEXT} !important;
    }}
    
    [data-testid="stMetricValue"] {{ 
        font-size: 2rem; 
        font-weight: bold; 
        color: {COLOR_ACCENT}; 
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
        
        df['Seller Name'] = df['Seller Name'].astype(str).str.strip()
        
        if 'Quantity X Product' in df.columns:
            df['Base_Product'] = df['Quantity X Product'].astype(str).str.strip()
        else:
            df['Base_Product'] = df['Product'].astype(str).str.strip()
            
        df['Phone'] = df['Phone'].astype(str).str.strip()
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
    col1.metric("Revenue", f"${total_rev:,.2f}")
    col2.metric("Customers", unique_cust)
    col3.metric("Orders", total_orders)
    
    st.markdown("---")
    st.markdown("**Customer List & Order Frequency**")
    
    cust_list = seller_data.groupby('Phone').agg(
        Orders=('Order ID', 'nunique'),
        Total_Spend=('Grand Total', 'sum')
    ).reset_index().sort_values(by='Total_Spend', ascending=False)
    
    st.dataframe(cust_list, use_container_width=True, hide_index=True)

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
                node, label=node, size=30, x=x, y=y,
                color={"background": COLOR_ACCENT, "border": COLOR_ACCENT}
            )
        else:
            net.add_node(
                node, label=node, size=10, x=x, y=y,
                color={"background": COLOR_TEXT, "border": COLOR_TEXT}
            )
            net.add_edge(seller_name, node, color={"color": COLOR_EDGE, "opacity": 0.5})

    net.set_options(f"""
    var options = {{
      "physics": {{ "enabled": false }},
      "nodes": {{ "font": {{ "color": "{COLOR_TEXT}", "size": 12, "strokeWidth": 3, "strokeColor": "{COLOR_BG}" }}, "shape": "dot" }},
      "edges": {{ "smooth": false }}
    }}
    """)
    
    html_file = f"net_single_seller.html"
    net.save_graph(html_file)
    with open(html_file, 'r', encoding='utf-8') as f:
        components.html(f.read(), height=470)
    os.remove(html_file)

# ============================================================================
# 4. UNIVERSAL UNIPARTITE SNA BUILDER (Fruchterman-Reingold Engine)
# ============================================================================
def build_network(df, node_col, edge_group_col, max_cap=100):
    G = nx.Graph()
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
    
    cmap = mcolors.LinearSegmentedColormap.from_list("connection_heatmap", [COLOR_ACCENT, "#ff5555", COLOR_TEXT])
    
    for node in G.nodes():
        deg = degree_dict.get(node, 0)
        
        if max_degree > min_degree:
            heat_ratio = (deg - min_degree) / (max_degree - min_degree)
        else:
            heat_ratio = 0.5
            
        node_color = mcolors.to_hex(cmap(heat_ratio))
        size = 10 + (25 * heat_ratio) 
        
        x = float(pos[node][0]) * 7500
        y = float(pos[node][1]) * 7500
        
        net.add_node(
            str(node), 
            label=f"{str(node)[:10]}.." if heat_ratio < 0.5 else f"{str(node)[:15]}",
            title=f"{prefix} {node}\nConnections: {deg}", 
            size=size,
            x=x, y=y,
            color={"background": node_color, "border": node_color}
        )
        
    for u, v, d in G.edges(data=True):
        net.add_edge(str(u), str(v), value=d.get('weight', 1), color={"color": COLOR_EDGE, "opacity": 0.6})
        
    net.set_options(f"""
    var options = {{
      "physics": {{ "enabled": false }},
      "nodes": {{ "font": {{ "color": "{COLOR_TEXT}", "size": 12, "strokeWidth": 3, "strokeColor": "{COLOR_BG}" }}, "shape": "dot" }},
      "edges": {{ "smooth": false }} 
    }}
    """)
    
    html_file = f"net_{title}.html"
    net.save_graph(html_file)
    with open(html_file, 'r', encoding='utf-8') as f:
        components.html(f.read(), height=620)
    os.remove(html_file)

# ============================================================================
# 5. GRAVITATIONAL WEB (State-Customer Ecosystem - Tab 4)
# ============================================================================
def draw_plotly_state_customer(df):
    unique_pairs = df[['State', 'Phone']].drop_duplicates()
    if unique_pairs.empty:
        st.warning("No data available to build the ecosystem.")
        return

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
        
        calculated_size = 18 + ((count / max_state_count) * 50)
        s_size.append(calculated_size)
        
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
        
        shared_text = ", ".join([str(s) for s in linked_states])
        c_hover.append(f"<b>Customer:</b> {cust}<br><b>States:</b> {shared_text}")
        
        for s in linked_states:
            sx, sy = state_coords[s]
            edge_x.extend([sx, cx, None])
            edge_y.extend([sy, cy, None])

    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=0.15, color=COLOR_EDGE),
        hoverinfo='none',
        showlegend=False
    ))

    fig.add_trace(go.Scattergl(
        x=c_x, y=c_y,
        mode='markers',
        marker=dict(size=4, color=COLOR_TEXT, line=dict(width=0.5, color=COLOR_BG)),
        customdata=c_hover,
        hovertemplate="%{customdata}<extra></extra>",
        name='Customers'
    ))

    fig.add_trace(go.Scattergl(
        x=s_x, y=s_y,
        mode='markers+text',
        marker=dict(size=s_size, color=COLOR_ACCENT, line=dict(width=2, color=COLOR_TEXT)),
        text=s_labels,
        textposition="bottom center",
        textfont=dict(color=COLOR_TEXT, size=13, family="Arial Black"),
        customdata=s_hover,
        hovertemplate="%{customdata}<extra></extra>",
        name='States'
    ))

    fig.update_layout(
        plot_bgcolor=COLOR_BG,
        paper_bgcolor=COLOR_BG,
        font=dict(color=COLOR_TEXT),
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=750,
        dragmode='pan',
        hoverlabel=dict(
            bgcolor=COLOR_SURFACE,
            font_size=13,
            font_family="Arial",
            bordercolor=COLOR_ACCENT
        ),
        hovermode='closest'
    )

    st.plotly_chart(fig, use_container_width=True, clear_figure=True, config={
        'scrollZoom': True,
        'displayModeBar': True,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'displaylogo': False
    })


# ============================================================================
# 6. GRAVITATIONAL WEB (Seller-Customer Ecosystem - Tab 5)
# ============================================================================
def draw_plotly_ecosystem(df):
    unique_pairs = df[['Seller Name', 'Phone']].drop_duplicates()
    if unique_pairs.empty:
        st.warning("No data available to build the ecosystem.")
        return

    seller_cust_counts = unique_pairs['Seller Name'].value_counts()
    max_seller_count = seller_cust_counts.max() if not seller_cust_counts.empty else 1
    sorted_sellers = seller_cust_counts.index.tolist()
    
    golden_angle = math.pi * (3 - math.sqrt(5))
    
    seller_coords = {}
    for i, s in enumerate(sorted_sellers):
        radius = 40 * math.sqrt(i) 
        angle = i * golden_angle
        sx = radius * math.cos(angle)
        sy = radius * math.sin(angle)
        seller_coords[s] = (sx, sy)
        
    cust_to_sellers = unique_pairs.groupby('Phone')['Seller Name'].apply(list).to_dict()

    edge_x, edge_y = [], []
    s_x, s_y, s_hover, s_size = [], [], [], []
    c_x, c_y, c_hover = [], [], []

    for s in sorted_sellers:
        x, y = seller_coords[s]
        s_x.append(x)
        s_y.append(y)
        count = seller_cust_counts.get(s, 0)
        
        calculated_size = 12 + ((count / max_seller_count) * 45)
        s_size.append(calculated_size)
        
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
        
        shared_text = ", ".join([str(s)[:15] for s in linked_sellers])
        c_hover.append(f"<b>Customer:</b> {cust}<br><b>Buys From:</b> {shared_text}")
        
        for s in linked_sellers:
            sx, sy = seller_coords[s]
            edge_x.extend([sx, cx, None])
            edge_y.extend([sy, cy, None])

    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=0.35, color=COLOR_EDGE),
        hoverinfo='none',
        showlegend=False
    ))

    fig.add_trace(go.Scattergl(
        x=c_x, y=c_y,
        mode='markers',
        marker=dict(size=6, color=COLOR_TEXT, line=dict(width=0.5, color=COLOR_BG)),
        customdata=c_hover,
        hovertemplate="%{customdata}<extra></extra>",
        name='Customers'
    ))

    fig.add_trace(go.Scattergl(
        x=s_x, y=s_y,
        mode='markers',
        marker=dict(size=s_size, color=COLOR_ACCENT, line=dict(width=1.5, color=COLOR_TEXT)),
        customdata=s_hover,
        hovertemplate="%{customdata}<extra></extra>",
        name='Sellers'
    ))

    fig.update_layout(
        plot_bgcolor=COLOR_BG,
        paper_bgcolor=COLOR_BG,
        font=dict(color=COLOR_TEXT),
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=750,
        dragmode='pan',
        hoverlabel=dict(
            bgcolor=COLOR_SURFACE,
            font_size=13,
            font_family="Arial",
            bordercolor=COLOR_ACCENT
        ),
        hovermode='closest'
    )

    st.plotly_chart(fig, use_container_width=True, clear_figure=True, config={
        'scrollZoom': True,
        'displayModeBar': True,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'displaylogo': False
    })


# ============================================================================
# 7. CUSTOMER CHURN ENGINE
# ============================================================================
def calculate_churn_rfm(df):
    if df.empty:
        return pd.DataFrame()
        
    snapshot_date = df['Order Date'].max() + pd.Timedelta(days=1)
    
    churn_df = df.groupby('Phone').agg(
        Frequency=('Order ID', 'nunique'),
        Last_Purchase=('Order Date', 'max'),
        Total_Spend=('Grand Total', 'sum')
    ).reset_index()
    
    churn_df['Recency_Days'] = (snapshot_date - churn_df['Last_Purchase']).dt.days
    
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
    return churn_df

# ============================================================================
# 8. MAIN DASHBOARD INTERFACE
# ============================================================================
def main():
    st.title("E-Commerce Intelligence: SNA & Churn")
    
    # ---------------------------------------------------------
    # HARDCODED DATASET PATH (NO UPLOADER)
    # ---------------------------------------------------------
    DATA_FILE = "RockHerb_Full.xlsx"
    
    if not os.path.exists(DATA_FILE):
        st.error(f"Dataset '{DATA_FILE}' not found. Please make sure '{DATA_FILE}' is inside the same folder as this script.")
        return
        
    with st.spinner("Processing dataset..."):
        raw_df = load_and_clean_data(DATA_FILE)
        
    if raw_df is None: return
    # ---------------------------------------------------------

    st.sidebar.markdown("---")
    st.sidebar.subheader("Time Filters")
    
    available_years = sorted(raw_df['Order Date'].dt.year.unique().astype(int).tolist(), reverse=True)
    selected_year = st.sidebar.selectbox("Filter by Year", ["All Years"] + available_years)
    
    if selected_year != "All Years":
        df = raw_df[raw_df['Order Date'].dt.year == selected_year]
        
        selected_quarter = st.sidebar.selectbox("Filter by Quarter", ["All Quarters", "Q1", "Q2", "Q3", "Q4"])
        
        if selected_quarter != "All Quarters":
            q_map = {"Q1": [1, 2, 3], "Q2": [4, 5, 6], "Q3": [7, 8, 9], "Q4": [10, 11, 12]}
            df = df[df['Order Date'].dt.month.isin(q_map[selected_quarter])]
    else:
        df = raw_df.copy()
        
    if df.empty:
        st.error("No data available for the selected timeframe.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Orders", f"{df['Order ID'].nunique():,}")
    c2.metric("Unique Customers", f"{df['Phone'].nunique():,}")
    c3.metric("Total Sellers", f"{df['Seller Name'].nunique():,}")
    c4.metric("Gross Revenue", f"${df['Grand Total'].sum():,.2f}")
    
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
            
        st.caption("Logic: Fruchterman-Reingold Force-Directed Algorithm. White nodes indicate higher connections, Red indicates fewer.")
        seller_net = build_network(df, node_col='Seller Name', edge_group_col='Phone')
        draw_spiderweb_network(seller_net, "seller", "Seller:")

    with tab2:
        st.subheader("Product Lifetime Value Network")
        st.caption("Logic: Fruchterman-Reingold Force-Directed Algorithm. White nodes indicate higher connections, Red indicates fewer.")
        
        exploded_df_prod = explode_products(df)
        exploded_df_prod['Month_Str'] = exploded_df_prod['Order Date'].dt.strftime('%Y-%m')
        
        prod_net = build_network(exploded_df_prod, node_col='Clean_Product', edge_group_col='Month_Str', max_cap=80)
        draw_spiderweb_network(prod_net, "product", "Prod:")

    with tab3:
        st.subheader("Customer Loyalty Network")
        st.caption("Logic: Fruchterman-Reingold Force-Directed Algorithm. White nodes indicate higher connections, Red indicates fewer.")
        
        exploded_df_cust = explode_products(df)
        
        cust_net = build_network(exploded_df_cust, node_col='Phone', edge_group_col='Clean_Product', max_cap=80) 
        draw_spiderweb_network(cust_net, "customer", "Cust:")

    with tab4:
        st.subheader("State-Customer Distribution Network")
        st.caption("Logic: Gravitational Web mapping Demographic States (Red Hubs) directly to buying Customers (White Satellites). Hub size is driven by total customers.")
        draw_plotly_state_customer(df)
        
    with tab5:
        st.subheader("Seller-Customer Ecosystem")
        st.caption("Logic: Organic Gravitational Spiral. Seller Hub size is driven by total customers. Click or hover on red hubs to see details.")
        draw_plotly_ecosystem(df)

    with tab6:
        st.subheader("Customer Churn Forecasting (Frequency Based)")
        churn_df = calculate_churn_rfm(df)
        
        c_left, c_right = st.columns([1, 2])
        
        with c_left:
            risk_counts = churn_df['Churn Risk Profile'].value_counts().reset_index()
            risk_counts.columns = ['Profile', 'Count']
            
            st.dataframe(risk_counts, use_container_width=True, hide_index=True)
            
        with c_right:
            fig, ax = plt.subplots(figsize=(8, 4), facecolor=COLOR_BG)
            ax.set_facecolor(COLOR_BG)
            
            ax.bar(risk_counts['Profile'], risk_counts['Count'], color=COLOR_ACCENT, edgecolor=COLOR_TEXT)
            
            ax.set_title("Customer Distribution by Risk Profile", color=COLOR_TEXT)
            ax.tick_params(colors=COLOR_TEXT)
            for spine in ax.spines.values(): spine.set_edgecolor(COLOR_EDGE)
            plt.xticks(rotation=15)
            
            st.pyplot(fig)
            
        st.markdown("**Raw Customer Churn Data:**")
        st.dataframe(churn_df.sort_values(by='Recency_Days', ascending=False).head(100), use_container_width=True)

if __name__ == "__main__":
    main()