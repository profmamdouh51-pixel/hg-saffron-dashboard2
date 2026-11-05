import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Mercury Toxicity Dashboard", layout="wide", page_icon="🧪")

# ============================================================
# 1. قاموس اللغات (22 لغة)
# ============================================================
LANG = {
    "en": {
        "name": "English",
        "title": "🧪 Mercury Toxicity & Saffron Recovery",
        "subtitle": "Full Interactive Dashboard — 6 Groups × 21 Biomarkers",
        "sidebar_header": "🎛️ Controls",
        "language": "🌐 Language",
        "category": "Select Category",
        "marker": "Select Biomarker",
        "ai_toggle": "🤖 AI Predictions",
        "recovery_toggle": "🎯 Recovery Index",
        "chart_title": "📊 {marker} — Average per Group",
        "stats_title": "📋 Detailed Statistics",
        "recovery_title": "🎯 Recovery Index",
        "best_treatment": "🏆 Best Treatment: **{treatment}** — Recovery **{pct:.1f}%**",
        "ai_title": "🤖 AI What-If Simulator",
        "select_group": "Select Group for Prediction",
        "ai_prediction": "AI Prediction — {group}",
        "actual_value": "Actual: {actual:.2f}",
        "feature_importance": "**Feature Importance:**",
        "error_file": "⚠️ `clean_data.csv` not found. Please upload it to your GitHub repository.",
        "table_group": "Group",
        "table_mean": "Mean ± SD",
        "table_n": "N",
        "table_treatment": "Treatment",
        "table_mean_val": "Mean",
        "table_recovery": "Recovery %",
        "caption": "🧪 Built from Toxicity of HgCl2 & Saffron Study",
        "about_title": "📖 About this Dashboard",
        "categories": {"Liver": "Liver", "Oxidative Stress": "Oxidative Stress",
                       "Neurotransmitters": "Neurotransmitters", "Anti-inflammatory": "Anti-inflammatory",
                       "Kidney": "Kidney", "Lipid Profile": "Lipid Profile"},
    },
    "ar": {
        "name": "العربية",
        "title": "🧪 سمية الزئبق وتعافي الزعفران",
        "subtitle": "لوحة تحكم تفاعلية شاملة — 6 مجموعات × 21 مؤشراً حيوياً",
        "sidebar_header": "🎛️ التحكم",
        "language": "🌐 اللغة",
        "category": "اختر القسم",
        "marker": "اختر المؤشر الحيوي",
        "ai_toggle": "🤖 تنبؤات الذكاء الاصطناعي",
        "recovery_toggle": "🎯 مؤشر التعافي",
        "chart_title": "📊 {marker} — المتوسط لكل مجموعة",
        "stats_title": "📋 الإحصاءات التفصيلية",
        "recovery_title": "🎯 مؤشر التعافي",
        "best_treatment": "🏆 أفضل علاج: **{treatment}** — تعافي **{pct:.1f}%**",
        "ai_title": "🤖 محاكي الذكاء الاصطناعي",
        "select_group": "اختر المجموعة للتنبؤ",
        "ai_prediction": "تنبؤ AI — {group}",
        "actual_value": "الفعلي: {actual:.2f}",
        "feature_importance": "**أهمية الميزات:**",
        "error_file": "⚠️ ملف `clean_data.csv` غير موجود. تأكد من رفعه إلى مستودع GitHub.",
        "table_group": "المجموعة",
        "table_mean": "المتوسط ± الانحراف",
        "table_n": "العدد",
        "table_treatment": "العلاج",
        "table_mean_val": "المتوسط",
        "table_recovery": "نسبة التعافي %",
        "caption": "🧪 مبني من دراسة سمية HgCl2 وتعافي الزعفران",
        "about_title": "📖 عن هذا النظام",
        "categories": {"Liver": "وظائف الكبد", "Oxidative Stress": "الإجهاد التأكسدي",
                       "Neurotransmitters": "النواقل العصبية", "Anti-inflammatory": "مضادات الالتهاب",
                       "Kidney": "وظائف الكلى", "Lipid Profile": "ملف الدهون"},
    },
    "fr": {
        "name": "Français",
        "title": "🧪 Toxicité du mercure et récupération du safran",
        "subtitle": "Tableau de bord interactif — 6 groupes × 21 biomarqueurs",
        "sidebar_header": "🎛️ Contrôles",
        "language": "🌐 Langue",
        "category": "Choisir une catégorie",
        "marker": "Choisir un biomarqueur",
        "ai_toggle": "🤖 Prédictions IA",
        "recovery_toggle": "🎯 Indice de récupération",
        "chart_title": "📊 {marker} — Moyenne par groupe",
        "stats_title": "📋 Statistiques détaillées",
        "recovery_title": "🎯 Indice de récupération",
        "best_treatment": "🏆 Meilleur traitement: **{treatment}** — Récupération **{pct:.1f}%**",
        "ai_title": "🤖 Simulateur IA",
        "select_group": "Choisir un groupe",
        "ai_prediction": "Prédiction IA — {group}",
        "actual_value": "Réel: {actual:.2f}",
        "feature_importance": "**Importance des caractéristiques:**",
        "error_file": "⚠️ `clean_data.csv` introuvable.",
        "table_group": "Groupe", "table_mean": "Moyenne ± ÉT", "table_n": "N",
        "table_treatment": "Traitement", "table_mean_val": "Moyenne", "table_recovery": "Récupération %",
        "caption": "🧪 Basé sur l'étude de toxicité du HgCl2 et du safran",
        "about_title": "📖 À propos de ce tableau de bord",
        "categories": {"Liver": "Foie", "Oxidative Stress": "Stress oxydatif",
                       "Neurotransmitters": "Neurotransmetteurs", "Anti-inflammatory": "Anti-inflammatoire",
                       "Kidney": "Rein", "Lipid Profile": "Profil lipidique"},
    },
    "es": {
        "name": "Español",
        "title": "🧪 Toxicidad del mercurio y recuperación del azafrán",
        "subtitle": "Panel interactivo — 6 grupos × 21 biomarcadores",
        "sidebar_header": "🎛️ Controles",
        "language": "🌐 Idioma",
        "category": "Seleccionar categoría",
        "marker": "Seleccionar biomarcador",
        "ai_toggle": "🤖 Predicciones IA",
        "recovery_toggle": "🎯 Índice de recuperación",
        "chart_title": "📊 {marker} — Promedio por grupo",
        "stats_title": "📋 Estadísticas detalladas",
        "recovery_title": "🎯 Índice de recuperación",
        "best_treatment": "🏆 Mejor tratamiento: **{treatment}** — Recuperación **{pct:.1f}%**",
        "ai_title": "🤖 Simulador IA",
        "select_group": "Seleccionar grupo",
        "ai_prediction": "Predicción IA — {group}",
        "actual_value": "Real: {actual:.2f}",
        "feature_importance": "**Importancia de características:**",
        "error_file": "⚠️ `clean_data.csv` no encontrado.",
        "table_group": "Grupo", "table_mean": "Media ± DE", "table_n": "N",
        "table_treatment": "Tratamiento", "table_mean_val": "Media", "table_recovery": "Recuperación %",
        "caption": "🧪 Basado en el estudio de toxicidad del HgCl2 y azafrán",
        "about_title": "📖 Acerca de este panel",
        "categories": {"Liver": "Hígado", "Oxidative Stress": "Estrés oxidativo",
                       "Neurotransmitters": "Neurotransmisores", "Anti-inflammatory": "Antiinflamatorio",
                       "Kidney": "Riñón", "Lipid Profile": "Perfil lipídico"},
    },
    "zh": {
        "name": "中文",
        "title": "🧪 汞毒性与藏红花恢复",
        "subtitle": "交互式仪表板 — 6组 × 21个生物标志物",
        "sidebar_header": "🎛️ 控制",
        "language": "🌐 语言",
        "category": "选择类别",
        "marker": "选择生物标志物",
        "ai_toggle": "🤖 AI预测",
        "recovery_toggle": "🎯 恢复指数",
        "chart_title": "📊 {marker} — 每组平均值",
        "stats_title": "📋 详细统计",
        "recovery_title": "🎯 恢复指数",
        "best_treatment": "🏆 最佳治疗: **{treatment}** — 恢复 **{pct:.1f}%**",
        "ai_title": "🤖 AI模拟器",
        "select_group": "选择组",
        "ai_prediction": "AI预测 — {group}",
        "actual_value": "实际: {actual:.2f}",
        "feature_importance": "**特征重要性:**",
        "error_file": "⚠️ 未找到 `clean_data.csv`。",
        "table_group": "组", "table_mean": "平均值 ± 标准差", "table_n": "N",
        "table_treatment": "治疗", "table_mean_val": "平均值", "table_recovery": "恢复 %",
        "caption": "🧪 基于HgCl2毒性与藏红花研究",
        "about_title": "📖 关于此仪表板",
        "categories": {"Liver": "肝脏", "Oxidative Stress": "氧化应激",
                       "Neurotransmitters": "神经递质", "Anti-inflammatory": "抗炎",
                       "Kidney": "肾脏", "Lipid Profile": "血脂"},
    },
}

# ============================================================
# 2. نصوص قسم About بالخمس لغات
# ============================================================
ABOUT = {
    "en": """
### 🧪 Mercury Toxicity & Saffron Recovery Dashboard

**What is this?** An interactive dashboard for analyzing a scientific study on mercury (HgCl₂) toxicity and saffron's protective effects in adult rats.

**Purpose:**
- Visualize the toxic effects of mercury on liver, kidney, brain, and lipid profile.
- Compare saffron's protective/therapeutic effects with Vitamin E (standard treatment).
- Calculate recovery percentages for 21 biomarkers.
- Provide AI-based predictions using Random Forest.

**How to use:**
1. **Choose language** from the sidebar.
2. **Select a category** (Liver, Kidney, Oxidative Stress, etc.).
3. **Select a biomarker** (AST, MDA, BDNF, etc.).
4. **Read the chart** — green = healthy, red = mercury, orange/purple = treatments.
5. **Check the statistics table** for means and standard deviations.
6. **Use the Recovery Index** to see which treatment worked best.
7. **Try the AI simulator** to explore "what-if" scenarios.

**Note:** All data is real from the original study. This tool supports research and education — it does not replace expert toxicological interpretation.
""",
    "ar": """
### 🧪 لوحة تحكم سمية الزئبق وتعافي الزعفران

**ما هذا؟** لوحة تحكم تفاعلية لتحليل دراسة علمية عن سمية الزئبق (HgCl₂) وفاعلية الزعفران الوقائية في فئران التجارب البالغة.

**الهدف:**
- عرض التأثيرات السامة للزئبق على الكبد، الكلى، المخ، وملف الدهون بشكل مرئي.
- مقارنة تأثير الزعفران الوقائي/العلاجي بفيتامين E (العلاج القياسي).
- حساب نسب التعافي لـ 21 مؤشراً حيوياً.
- توفير تنبؤات بالذكاء الاصطناعي باستخدام Random Forest.

**كيف تستخدم النظام:**
1. **اختر اللغة** من الشريط الجانبي.
2. **اختر القسم** (الكبد، الكلى، الإجهاد التأكسدي، إلخ).
3. **اختر المؤشر الحيوي** (AST, MDA, BDNF, إلخ).
4. **اقرأ الرسم البياني** — أخضر = طبيعي، أحمر = زئبق، برتقالي/بنفسجي = علاجات.
5. **راجع جدول الإحصاءات** للمتوسطات والانحرافات المعيارية.
6. **استخدم مؤشر التعافي** لمعرفة أفضل علاج.
7. **جرب محاكي الذكاء الاصطناعي** لاستكشاف سيناريوهات "ماذا لو".

**ملاحظة:** جميع البيانات حقيقية من الدراسة الأصلية. هذه الأداة تدعم البحث والتعليم — ولا تغني عن التفسير المتخصص.
""",
    "fr": """
### 🧪 Tableau de bord : Toxicité du mercure et récupération du safran

**Qu'est-ce que c'est ?** Un tableau de bord interactif pour analyser une étude scientifique sur la toxicité du mercure (HgCl₂) et les effets protecteurs du safran chez le rat adulte.

**Objectif :**
- Visualiser les effets toxiques du mercure sur le foie, les reins, le cerveau et le profil lipidique.
- Comparer les effets protecteurs/thérapeutiques du safran avec la vitamine E (traitement standard).
- Calculer les pourcentages de récupération pour 21 biomarqueurs.
- Fournir des prédictions par IA (Random Forest).

**Comment utiliser :**
1. **Choisissez la langue** dans la barre latérale.
2. **Sélectionnez une catégorie** (Foie, Rein, Stress oxydatif, etc.).
3. **Sélectionnez un biomarqueur** (AST, MDA, BDNF, etc.).
4. **Lisez le graphique** — vert = sain, rouge = mercure, orange/violet = traitements.
5. **Consultez le tableau statistique** pour les moyennes et écarts-types.
6. **Utilisez l'indice de récupération** pour voir le meilleur traitement.
7. **Essayez le simulateur IA** pour explorer des scénarios "et si".

**Remarque :** Toutes les données sont réelles, issues de l'étude originale. Cet outil soutient la recherche et l'éducation — il ne remplace pas l'interprétation toxicologique experte.
""",
    "es": """
### 🧪 Panel: Toxicidad del mercurio y recuperación del azafrán

**¿Qué es esto?** Un panel interactivo para analizar un estudio científico sobre la toxicidad del mercurio (HgCl₂) y los efectos protectores del azafrán en ratas adultas.

**Objetivo:**
- Visualizar los efectos tóxicos del mercurio en hígado, riñón, cerebro y perfil lipídico.
- Comparar los efectos protectores/terapéuticos del azafrán con la vitamina E (tratamiento estándar).
- Calcular porcentajes de recuperación para 21 biomarcadores.
- Proporcionar predicciones con IA (Random Forest).

**Cómo usar:**
1. **Elige el idioma** en la barra lateral.
2. **Selecciona una categoría** (Hígado, Riñón, Estrés oxidativo, etc.).
3. **Selecciona un biomarcador** (AST, MDA, BDNF, etc.).
4. **Lee el gráfico** — verde = sano, rojo = mercurio, naranja/morado = tratamientos.
5. **Consulta la tabla estadística** para medias y desviaciones estándar.
6. **Usa el Índice de Recuperación** para ver el mejor tratamiento.
7. **Prueba el simulador de IA** para explorar escenarios "qué pasaría si".

**Nota:** Todos los datos son reales del estudio original. Esta herramienta apoya la investigación y la educación — no reemplaza la interpretación toxicológica experta.
""",
    "zh": """
### 🧪 仪表板：汞毒性与藏红花恢复

**这是什么？** 一个交互式仪表板，用于分析一项关于汞（HgCl₂）毒性以及藏红花对成年大鼠保护作用的科学研究。

**目的：**
- 可视化汞对肝脏、肾脏、大脑和血脂的毒性作用。
- 将藏红花的保护/治疗效果与维生素E（标准治疗）进行比较。
- 计算21个生物标志物的恢复百分比。
- 使用随机森林提供基于AI的预测。

**如何使用：**
1. **从侧边栏选择语言**。
2. **选择一个类别**（肝脏、肾脏、氧化应激等）。
3. **选择一个生物标志物**（AST、MDA、BDNF等）。
4. **阅读图表** — 绿色 = 健康，红色 = 汞，橙色/紫色 = 治疗。
5. **查看统计表**了解平均值和标准差。
6. **使用恢复指数**查看最佳治疗。
7. **尝试AI模拟器**探索"如果"场景。

**注意：** 所有数据均来自原始研究的真实数据。此工具支持研究和教育 — 不能替代专业的毒理学解释。
""",
}

# ============================================================
# 3. اختيار اللغة
# ============================================================
if "lang" not in st.session_state:
    st.session_state.lang = "en"

with st.sidebar:
    lang_options = list(LANG.keys())
    lang_choice = st.selectbox(
        "🌐 Language / اللغة / Langue",
        options=lang_options,
        format_func=lambda x: LANG[x]["name"],
        index=lang_options.index(st.session_state.lang),
    )
    st.session_state.lang = lang_choice

t = LANG[st.session_state.lang]

# ============================================================
# 4. العنوان
# ============================================================
st.markdown(f"""
    <h1 style='text-align:center; color:#2c3e50;'>{t['title']}</h1>
    <p style='text-align:center; color:#7f8c8d;'>{t['subtitle']}</p>
""", unsafe_allow_html=True)

# ============================================================
# 5. تحميل البيانات
# ============================================================
@st.cache_data
def load_data():
    return pd.read_csv('clean_data.csv')

try:
    df = load_data()
except FileNotFoundError:
    st.error(t["error_file"])
    st.stop()

# ============================================================
# 6. الشريط الجانبي
# ============================================================
st.sidebar.header(t["sidebar_header"])

raw_categories = sorted(df['Category'].unique())
category_display = [t["categories"].get(c, c) for c in raw_categories]
category_map = dict(zip(category_display, raw_categories))
selected_category_display = st.sidebar.selectbox(t["category"], category_display)
category = category_map[selected_category_display]

available_markers = sorted([str(m) for m in df[df['Category'] == category]['Marker'].unique()])
marker = st.sidebar.selectbox(t["marker"], available_markers)

st.sidebar.divider()
show_ai = st.sidebar.checkbox(t["ai_toggle"], value=True)
show_recovery = st.sidebar.checkbox(t["recovery_toggle"], value=True)

# ============================================================
# 7. تصفية البيانات
# ============================================================
marker_data = df[df['Marker'] == marker].copy()
group_order = ['Control', 'Saffron+ normal', 'Hg-induced',
               'Hg-Saffron (Protective)', 'VitE Treatment', 'Saffron Treatment',
               'Control (-Ve.)', 'Normal + Saffron', 'Protective',
               '+ Ve + HgCl2', '+ Ve + Saffron', '+ Ve + Vit E']
present_groups = [g for g in group_order if g in marker_data['Group'].unique()]
marker_data['Group'] = pd.Categorical(marker_data['Group'], categories=present_groups, ordered=True)

summary = marker_data.groupby('Group', observed=True)['Value'].agg(['mean', 'std', 'count']).reset_index()

# ============================================================
# 8. الرسم البياني
# ============================================================
st.subheader(t["chart_title"].format(marker=marker))
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

# ============================================================
# 9. الإحصاءات
# ============================================================
st.subheader(t["stats_title"])
stats = summary.copy()
stats.columns = [t["table_group"], "Mean", "SD", t["table_n"]]
stats[t["table_mean"]] = stats.apply(lambda r: f"{r['Mean']:.2f} ± {r['SD']:.2f}", axis=1)
st.dataframe(stats[[t["table_group"], t["table_mean"], t["table_n"]]], use_container_width=True)

# ============================================================
# 10. مؤشر التعافي
# ============================================================
if show_recovery:
    st.subheader(t["recovery_title"])
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
            recovery_rows.append({t["table_treatment"]: g, t["table_mean_val"]: g_mean, t["table_recovery"]: pct})
    if recovery_rows:
        rec_df = pd.DataFrame(recovery_rows)
        st.dataframe(rec_df.style.format({t["table_mean_val"]: '{:.2f}', t["table_recovery"]: '{:.1f}%'}),
                     use_container_width=True)
        best = rec_df.loc[rec_df[t["table_recovery"]].idxmax()]
        st.success(t["best_treatment"].format(treatment=best[t["table_treatment"]], pct=best[t["table_recovery"]]))

# ============================================================
# 11. الذكاء الاصطناعي
# ============================================================
if show_ai:
    st.subheader(t["ai_title"])
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
            g = st.selectbox(t["select_group"], unique_groups)
            pred = model.predict([[group_codes[g]]])[0]
            actual = get_mean(g) if g in present_groups else None
            delta_text = t["actual_value"].format(actual=actual) if actual and pd.notna(actual) else None
            st.metric(t["ai_prediction"].format(group=g), f"{pred:.2f}", delta=delta_text)
        with col2:
            st.write(t["feature_importance"])
            fi = pd.DataFrame({'Importance': model.feature_importances_}, index=['Group Code'])
            st.bar_chart(fi)

# ============================================================
# 12. قسم About (بخمس لغات)
# ============================================================
st.divider()
with st.expander(t["about_title"], expanded=False):
    st.markdown(ABOUT[st.session_state.lang])

st.caption(t["caption"])
