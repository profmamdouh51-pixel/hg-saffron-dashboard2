import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Mercury Toxicity Dashboard", layout="wide", page_icon="🧪")

st.markdown("""
    <h1 style='text-align:center; color:#2c3e50;'>🧪 Mercury Toxicity & Saffron Recovery</h1>
    <p style='text-align:center; color:#7f8c8d;'>Full Interactive Dashboard — 6 Groups × 21 Biomarkers</p>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv('clean_data.csv')

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ ملف clean_data.csv غير موجود. تأكد من رفعه إلى مستودع GitHub.")
    st.stop()

st.sidebar.header("🎛️ التحكم")
category = st.sidebar.selectbox("اختر القسم", sorted(df['Category'].unique()))
available_markers = sorted(df[df['Category'] == category]['Marker'].unique())
marker = st.sidebar.selectbox("اختر المؤشر الحيوي", available_markers)
st.sidebar.divider()
show_ai = st.sidebar.checkbox("🤖 AI Predictions", value=True)
show_recovery = st.sidebar.checkbox("🎯 Recovery Index", value=True)

marker_data = df[df['Marker'] == marker].copy()
group_order = ['Control', 'Saffron+ normal', 'Hg-induced',
               'Hg-Saffron (Protective)', 'VitE Treatment', 'Saffron Treatment',
               'Control (-Ve.)', 'Normal + Saffron', 'Protective',
               '+ Ve + HgCl2', '+ Ve + Saffron', '+ Ve + Vit E']
present_groups = [g for g in group_order if g in marker_data['Group'].unique()]
marker_data['Group'] = pd.Categorical(marker_data['Group'], categories=present_groups, ordered=True)

summary = marker_data.groupby('Group', observed=True)['Value'].agg(['mean', 'std', 'count']).reset_index()

st.subheader(f"📊 {marker} — Average per Group")
fig, ax = plt.subplots(figsize=(12, 5))
colors = ['#2ecc71', '#27ae60', '#e74c3c', '#f39c12', '#3498db', '#9b59b6',
          '#1abc9c', '#16a085', '#c0392b', '#d35400', '#2980b9', '#8e44ad'][:len(summary)]
bars = ax.bar(summary['Group'], summary['mean'], yerr=summary['std'],
              capsize=5, color=colors, edgecolor='black', alpha=0.85)
ax.set_ylabel(marker, fontsize=12)
ax.set_xticklabels(summary['Group'], rotation=15, ha='right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
for bar, val in zip(bars, summary['mean']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
st.pyplot(fig)
plt.close(fig)

st.subheader("📋 الإحصاءات التفصيلية")
stats = summary.copy()
stats.columns = ['Group', 'Mean', 'SD', 'N']
stats['Mean ± SD'] = stats.apply(lambda r: f"{r['Mean']:.2f} ± {r['SD']:.2f}", axis=1)
st.dataframe(stats[['Group', 'Mean ± SD', 'N']], use_container_width=True)

if show_recovery:
    st.subheader("🎯 Recovery Index")
    def get_mean(g):
        row = summary[summary['Group'] == g]
        return row['mean'].values[0] if len(row) > 0 else np.nan
    control_mean = get_mean('Control') if pd.notna(get_mean('Control')) else get_mean('Control (-Ve.)')
    hg_mean = get_mean('Hg-induced') if pd.notna(get_mean('Hg-induced')) else get_mean('+ Ve + HgCl2')
    treatments = ['Hg-Saffron (Protective)', 'VitE Treatment', 'Saffron Treatment',
                  '+ Ve + Saffron', '+ Ve + Vit E']
    recovery_rows = []
    for g in treatments:
        g_mean = get_mean(g)
        if pd.notna(g_mean) and pd.notna(hg_mean) and pd.notna(control_mean) and hg_mean != control_mean:
            pct = ((hg_mean - g_mean) / (hg_mean - control_mean)) * 100
            recovery_rows.append({'Treatment': g, 'Mean': g_mean, 'Recovery %': pct})
    if recovery_rows:
        rec_df = pd.DataFrame(recovery_rows)
        st.dataframe(rec_df.style.format({'Mean': '{:.2f}', 'Recovery %': '{:.1f}%'}),
                     use_container_width=True)
        best = rec_df.loc[rec_df['Recovery %'].idxmax()]
        st.success(f"🏆 أفضل علاج: **{best['Treatment']}** — تعافي **{best['Recovery %']:.1f}%**")

if show_ai:
    st.subheader("🤖 AI What-If Simulator")
    # إنشاء أكواد ديناميكية لكل المجموعات الموجودة في هذا المؤشر
    unique_groups = marker_data['Group'].unique().tolist()
    group_codes = {g: i for i, g in enumerate(unique_groups)}
    marker_data['Code'] = marker_data['Group'].map(group_codes).astype(int)
    X = marker_data[['Code']]
    y = marker_data['Value']
    if len(X) > 0:
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X, y)
        col1, col2 = st.columns(2)
        with col1:
            g = st.selectbox("اختر المجموعة للتنبؤ", unique_groups)
            pred = model.predict([[group_codes[g]]])[0]
            actual = get_mean(g) if g in present_groups else None
            st.metric(f"AI Prediction — {g}", f"{pred:.2f}",
                      delta=f"الفعلي: {actual:.2f}" if actual and pd.notna(actual) else None)
        with col2:
            st.write("**Feature Importance:**")
            fi = pd.DataFrame({'Importance': model.feature_importances_}, index=['Group Code'])
            st.bar_chart(fi)

st.divider()
st.caption("🧪 Built from Toxicity of HgCl2 & Saffron Study")
