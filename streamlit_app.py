import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(layout="wide", page_title="PCA Analyzer")
st.title("PCA Analyzer — Upload Excel/CSV and get interactive plots + interpretation")

uploaded = st.file_uploader("Upload an Excel (.xlsx) or CSV file with numeric features", type=["xlsx","csv"])

def read_file(uploaded_file):
    if uploaded_file is None:
        return None
    fname = uploaded_file.name.lower()
    if fname.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file, engine='openpyxl')

if uploaded:
    df = read_file(uploaded)
    if df is None:
        st.error("Could not read file")
        st.stop()

    st.subheader("Preview of data")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found. Make sure your Excel has numeric feature columns.")
        st.stop()

    st.sidebar.header("Settings")
    features = st.sidebar.multiselect("Select feature columns (numeric)", numeric_cols, default=numeric_cols)
    label_col = st.sidebar.selectbox("Optional label/categorical column for coloring", [None] + list(df.columns))
    n_comp = st.sidebar.slider("Number of principal components", min_value=2, max_value=min(10, max(2, len(features))), value=min(5, max(2, len(features))))

    X = df[features].dropna()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=n_comp)
    pcs = pca.fit_transform(Xs)
    cols = [f"PC{i+1}" for i in range(pcs.shape[1])]
    pca_df = pd.DataFrame(pcs, columns=cols, index=X.index)

    # Attach original feature values for hover data (align indices)
    # This ensures plotly hover_data columns exist in the PCA dataframe
    try:
        pca_df = pd.concat([pca_df, df.loc[pca_df.index, features]], axis=1)
    except Exception:
        # fallback: add features column by column
        for feat in features:
            pca_df[feat] = df.loc[pca_df.index, feat]

    if label_col:
        pca_df[label_col] = df.loc[pca_df.index, label_col]

    st.subheader("PCA Results")
    st.write("Explained variance ratio:")
    evr = pca.explained_variance_ratio_
    evr_df = pd.DataFrame({"PC": cols, "Explained Variance Ratio": evr, "Cumulative": np.cumsum(evr)})
    st.dataframe(evr_df)

    # Scree plot and cumulative variance
    fig_scree = px.bar(evr_df, x='PC', y='Explained Variance Ratio', title='Scree plot (explained variance ratio)', text='Explained Variance Ratio')
    fig_scree.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_cum = px.line(evr_df, x='PC', y='Cumulative', title='Cumulative explained variance', markers=True)

    # Correlation heatmap
    corr = X.corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1, title='Feature correlation heatmap')

    # Loadings heatmap (features vs PCs)
    try:
        fig_loadings = px.imshow(loadings.T, x=cols, y=features, color_continuous_scale='RdBu', title='Loadings heatmap (features vs PCs)')
    except Exception:
        fig_loadings = None

    # Sessions (save/load)
    import os, pickle, datetime
    sessions_dir = 'sessions'
    os.makedirs(sessions_dir, exist_ok=True)
    st.sidebar.header('Sessions')
    session_name = st.sidebar.text_input('Session name (optional)')
    if st.sidebar.button('Save session'):
        sname = (session_name.strip() or datetime.datetime.now().strftime('session_%Y%m%d_%H%M%S'))
        filepath = os.path.join(sessions_dir, f"{sname}.pkl")
        data = {'pca_df': pca_df, 'loadings': loadings, 'evr_df': evr_df, 'features': features, 'label_col': label_col}
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        st.sidebar.success(f"Saved session to {filepath}")
    files = sorted([f for f in os.listdir(sessions_dir) if f.endswith('.pkl')])
    selected = st.sidebar.selectbox('Load saved session', [''] + files)
    if st.sidebar.button('Load session') and selected:
        with open(os.path.join(sessions_dir, selected), 'rb') as f:
            data = pickle.load(f)
        st.session_state['loaded_session'] = data
        st.experimental_rerun()

    # If a session was loaded, override variables
    if 'loaded_session' in st.session_state:
        ld = st.session_state['loaded_session']
        pca_df = ld.get('pca_df', pca_df)
        loadings = ld.get('loadings', loadings)
        evr_df = ld.get('evr_df', evr_df)
        features = ld.get('features', features)
        label_col = ld.get('label_col', label_col)

    # PC scatter
    st.subheader('Interactive PCA scatter')
    pc_x = st.selectbox('X axis', cols, index=0)
    pc_y = st.selectbox('Y axis', cols, index=1)
    size = st.slider('Marker size', 5, 30, 12)
    show_matrix = st.sidebar.checkbox('Show scatter matrix (PCs)')
    show_3d = st.sidebar.checkbox('Show 3D PCA scatter (PC1,PC2,PC3)')

    if label_col:
        fig_scatter = px.scatter(pca_df, x=pc_x, y=pc_y, color=label_col, hover_data=features, title=f"{pc_x} vs {pc_y}")
    else:
        fig_scatter = px.scatter(pca_df, x=pc_x, y=pc_y, hover_data=features, title=f"{pc_x} vs {pc_y}")
    fig_scatter.update_traces(marker=dict(size=size))

    # 3D scatter (if requested and available)
    fig_3d = None
    if show_3d and len(cols) >= 3:
        try:
            if label_col:
                fig_3d = px.scatter_3d(pca_df, x=cols[0], y=cols[1], z=cols[2], color=label_col, hover_data=features, title='3D PCA scatter')
            else:
                fig_3d = px.scatter_3d(pca_df, x=cols[0], y=cols[1], z=cols[2], hover_data=features, title='3D PCA scatter')
            fig_3d.update_traces(marker=dict(size=max(5, size-4)))
        except Exception:
            fig_3d = None

    if show_matrix:
        try:
            matrix_df = pca_df[cols + ([label_col] if label_col else [])]
        except Exception:
            matrix_df = pca_df[cols]
        fig_matrix = px.scatter_matrix(matrix_df, dimensions=cols, color=label_col if label_col else None, title='PCA scatter matrix')
    else:
        fig_matrix = None

    # PC loadings bar for selected PC
    st.sidebar.header('Loadings')
    sel_pc = st.sidebar.selectbox('Select PC for loadings bar', cols, index=0)
    try:
        load_series = loadings[sel_pc].sort_values(key=lambda x: x.abs(), ascending=False)
        fig_loadings_bar = px.bar(load_series.reset_index().rename(columns={'index':'feature', sel_pc:'loading'}), x='feature', y='loading', title=f'Feature loadings for {sel_pc}')
    except Exception:
        fig_loadings_bar = None

    # Violin plots for features by label
    show_violin = st.sidebar.checkbox('Show violin plots by group')
    violin_fig = None
    if show_violin and label_col:
        try:
            top_feats = features[:6]
            import math
            from plotly.subplots import make_subplots
            rows = math.ceil(len(top_feats)/2)
            fig = make_subplots(rows=rows, cols=2, subplot_titles=top_feats)
            r = c = 1
            for i, feat in enumerate(top_feats):
                sub = px.violin(df, x=label_col, y=feat, box=True, points='all', title=feat)
                for trace in sub.data:
                    fig.add_trace(trace, row=r, col=c)
                c += 1
                if c > 2:
                    c = 1
                    r += 1
            fig.update_layout(height=300*rows, title='Feature distributions by group')
            violin_fig = fig
        except Exception:
            violin_fig = None

    # Loadings
    loadings = pd.DataFrame(pca.components_.T, index=features, columns=cols)

    st.subheader("Loadings (feature contributions to PCs)")
    st.dataframe(loadings)

    # Biplot-like arrows (scaled)
    def make_biplot(fig, pcx, pcy, loadings, scale=3):
        lx = loadings[pcx]
        ly = loadings[pcy]
        for i, feat in enumerate(loadings.index):
            fig.add_trace(go.Scatter(x=[0, lx[i]*scale], y=[0, ly[i]*scale], mode='lines+markers+text', marker=dict(size=1), text=[None, feat], textposition='top center', showlegend=False))
        return fig

    biplot_fig = go.Figure(fig_scatter)
    biplot_fig = make_biplot(biplot_fig, pc_x, pc_y, loadings, scale=3)
    biplot_fig.update_layout(title=f"Biplot overlay: {pc_x} and {pc_y}")

    # Interpretation
    def interpret(pc_index, loadings, evr):
        vec = loadings.iloc[:, pc_index]
        top_pos = vec.abs().sort_values(ascending=False).head(5).index.tolist()
        top_vals = vec.loc[top_pos]
        parts = []
        for f, v in zip(top_pos, top_vals):
            parts.append(f"{f} ({v:.2f})")
        pct = evr[pc_index]*100
        return f"PC{pc_index+1} explains {pct:.2f}% of variance. Top contributing features: " + ", ".join(parts)

    interpretations = []
    for i in range(min(3, len(cols))):
        interpretations.append(interpret(i, loadings, evr))

    # Layout
    col1, col2 = st.columns([1,1])
    with col1:
        st.plotly_chart(fig_scree, use_container_width=True)
        st.plotly_chart(biplot_fig, use_container_width=True)
    with col2:
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.header("Interpretation (auto-generated)")
        for t in interpretations:
            st.write("- ", t)

    # Download results
    out = BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        pca_df.to_excel(writer, sheet_name='PCA_scores')
        loadings.to_excel(writer, sheet_name='Loadings')
        evr_df.to_excel(writer, sheet_name='ExplainedVariance')
    st.download_button("Download PCA results as Excel", data=out.getvalue(), file_name='pca_results.xlsx')

else:
    st.info("Upload an Excel (.xlsx) or CSV file to start PCA analysis. See README for how to prepare the Excel file.")



# Helpful quick instructions + small example
if st.sidebar.checkbox("Show example/data format instructions"):
    st.sidebar.markdown("""
    Steps to create an Excel file for PCA:
    1. Put variable (feature) names in the first row as column headers.
    2. Each following row is an observation/sample.
    3. All feature columns must be numeric. Optionally add one categorical column (e.g., 'Group') to color points.
    4. Example (CSV copy-paste):\n
    SampleID,Feature_A,Feature_B,Feature_C,Group\n
    S1,1.2,3.4,2.1,A\n
    S2,2.3,3.8,1.8,B\n
    S3,1.9,4.0,2.5,A\n
    Save as .xlsx or .csv and upload above.
    """)

# Footer
st.markdown("---")
st.caption("Lalit Pandurang Patil, PhD student IARI, New Delhi")

