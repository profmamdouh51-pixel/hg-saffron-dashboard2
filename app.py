import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Mercury Toxicity Dashboard", layout="wide", page_icon="🧪")

# ============================================================
# قاموس اللغات (22 لغة عالمية)
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
        "categories": {"Liver": "Hígado", "Oxidative Stress": "Estrés oxidativo",
                       "Neurotransmitters": "Neurotransmisores", "Anti-inflammatory": "Antiinflamatorio",
                       "Kidney": "Riñón", "Lipid Profile": "Perfil lipídico"},
    },
    "de": {
        "name": "Deutsch",
        "title": "🧪 Quecksilbertoxizität & Safran-Erholung",
        "subtitle": "Interaktives Dashboard — 6 Gruppen × 21 Biomarker",
        "sidebar_header": "🎛️ Steuerung",
        "language": "🌐 Sprache",
        "category": "Kategorie wählen",
        "marker": "Biomarker wählen",
        "ai_toggle": "🤖 KI-Vorhersagen",
        "recovery_toggle": "🎯 Erholungsindex",
        "chart_title": "📊 {marker} — Durchschnitt pro Gruppe",
        "stats_title": "📋 Detaillierte Statistik",
        "recovery_title": "🎯 Erholungsindex",
        "best_treatment": "🏆 Beste Behandlung: **{treatment}** — Erholung **{pct:.1f}%**",
        "ai_title": "🤖 KI-Simulator",
        "select_group": "Gruppe wählen",
        "ai_prediction": "KI-Vorhersage — {group}",
        "actual_value": "Tatsächlich: {actual:.2f}",
        "feature_importance": "**Merkmalsbedeutung:**",
        "error_file": "⚠️ `clean_data.csv` nicht gefunden.",
        "table_group": "Gruppe", "table_mean": "Mittelwert ± SD", "table_n": "N",
        "table_treatment": "Behandlung", "table_mean_val": "Mittelwert", "table_recovery": "Erholung %",
        "caption": "🧪 Basierend auf HgCl2-Toxizitäts- und Safranstudie",
        "categories": {"Liver": "Leber", "Oxidative Stress": "Oxidativer Stress",
                       "Neurotransmitters": "Neurotransmitter", "Anti-inflammatory": "Entzündungshemmend",
                       "Kidney": "Niere", "Lipid Profile": "Lipidprofil"},
    },
    "it": {
        "name": "Italiano",
        "title": "🧪 Tossicità del mercurio e recupero dello zafferano",
        "subtitle": "Dashboard interattiva — 6 gruppi × 21 biomarcatori",
        "sidebar_header": "🎛️ Controlli",
        "language": "🌐 Lingua",
        "category": "Seleziona categoria",
        "marker": "Seleziona biomarcatore",
        "ai_toggle": "🤖 Predizioni IA",
        "recovery_toggle": "🎯 Indice di recupero",
        "chart_title": "📊 {marker} — Media per gruppo",
        "stats_title": "📋 Statistiche dettagliate",
        "recovery_title": "🎯 Indice di recupero",
        "best_treatment": "🏆 Miglior trattamento: **{treatment}** — Recupero **{pct:.1f}%**",
        "ai_title": "🤖 Simulatore IA",
        "select_group": "Seleziona gruppo",
        "ai_prediction": "Predizione IA — {group}",
        "actual_value": "Reale: {actual:.2f}",
        "feature_importance": "**Importanza caratteristiche:**",
        "error_file": "⚠️ `clean_data.csv` non trovato.",
        "table_group": "Gruppo", "table_mean": "Media ± DS", "table_n": "N",
        "table_treatment": "Trattamento", "table_mean_val": "Media", "table_recovery": "Recupero %",
        "caption": "🧪 Basato sullo studio di tossicità HgCl2 e zafferano",
        "categories": {"Liver": "Fegato", "Oxidative Stress": "Stress ossidativo",
                       "Neurotransmitters": "Neurotrasmettitori", "Anti-inflammatory": "Antinfiammatorio",
                       "Kidney": "Rene", "Lipid Profile": "Profilo lipidico"},
    },
    "pt": {
        "name": "Português",
        "title": "🧪 Toxicidade do mercúrio e recuperação do açafrão",
        "subtitle": "Painel interativo — 6 grupos × 21 biomarcadores",
        "sidebar_header": "🎛️ Controles",
        "language": "🌐 Idioma",
        "category": "Selecionar categoria",
        "marker": "Selecionar biomarcador",
        "ai_toggle": "🤖 Previsões IA",
        "recovery_toggle": "🎯 Índice de recuperação",
        "chart_title": "📊 {marker} — Média por grupo",
        "stats_title": "📋 Estatísticas detalhadas",
        "recovery_title": "🎯 Índice de recuperação",
        "best_treatment": "🏆 Melhor tratamento: **{treatment}** — Recuperação **{pct:.1f}%**",
        "ai_title": "🤖 Simulador IA",
        "select_group": "Selecionar grupo",
        "ai_prediction": "Previsão IA — {group}",
        "actual_value": "Real: {actual:.2f}",
        "feature_importance": "**Importância das características:**",
        "error_file": "⚠️ `clean_data.csv` não encontrado.",
        "table_group": "Grupo", "table_mean": "Média ± DP", "table_n": "N",
        "table_treatment": "Tratamento", "table_mean_val": "Média", "table_recovery": "Recuperação %",
        "caption": "🧪 Baseado no estudo de toxicidade do HgCl2 e açafrão",
        "categories": {"Liver": "Fígado", "Oxidative Stress": "Estresse oxidativo",
                       "Neurotransmitters": "Neurotransmissores", "Anti-inflammatory": "Anti-inflamatório",
                       "Kidney": "Rim", "Lipid Profile": "Perfil lipídico"},
    },
    "ru": {
        "name": "Русский",
        "title": "🧪 Токсичность ртути и восстановление шафрана",
        "subtitle": "Интерактивная панель — 6 групп × 21 биомаркер",
        "sidebar_header": "🎛️ Управление",
        "language": "🌐 Язык",
        "category": "Выбрать категорию",
        "marker": "Выбрать биомаркер",
        "ai_toggle": "🤖 Прогнозы ИИ",
        "recovery_toggle": "🎯 Индекс восстановления",
        "chart_title": "📊 {marker} — Среднее по группе",
        "stats_title": "📋 Подробная статистика",
        "recovery_title": "🎯 Индекс восстановления",
        "best_treatment": "🏆 Лучшее лечение: **{treatment}** — Восстановление **{pct:.1f}%**",
        "ai_title": "🤖 Симулятор ИИ",
        "select_group": "Выбрать группу",
        "ai_prediction": "Прогноз ИИ — {group}",
        "actual_value": "Фактически: {actual:.2f}",
        "feature_importance": "**Важность признаков:**",
        "error_file": "⚠️ `clean_data.csv` не найден.",
        "table_group": "Группа", "table_mean": "Среднее ± СО", "table_n": "N",
        "table_treatment": "Лечение", "table_mean_val": "Среднее", "table_recovery": "Восстановление %",
        "caption": "🧪 На основе исследования токсичности HgCl2 и шафрана",
        "categories": {"Liver": "Печень", "Oxidative Stress": "Окислительный стресс",
                       "Neurotransmitters": "Нейротрансмиттеры", "Anti-inflammatory": "Противовоспалительный",
                       "Kidney": "Почки", "Lipid Profile": "Липидный профиль"},
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
        "categories": {"Liver": "肝脏", "Oxidative Stress": "氧化应激",
                       "Neurotransmitters": "神经递质", "Anti-inflammatory": "抗炎",
                       "Kidney": "肾脏", "Lipid Profile": "血脂"},
    },
    "ja": {
        "name": "日本語",
        "title": "🧪 水銀毒性とサフラン回復",
        "subtitle": "インタラクティブダッシュボード — 6グループ × 21バイオマーカー",
        "sidebar_header": "🎛️ コントロール",
        "language": "🌐 言語",
        "category": "カテゴリを選択",
        "marker": "バイオマーカーを選択",
        "ai_toggle": "🤖 AI予測",
        "recovery_toggle": "🎯 回復指数",
        "chart_title": "📊 {marker} — グループ別平均",
        "stats_title": "📋 詳細統計",
        "recovery_title": "🎯 回復指数",
        "best_treatment": "🏆 最良の治療: **{treatment}** — 回復 **{pct:.1f}%**",
        "ai_title": "🤖 AIシミュレーター",
        "select_group": "グループを選択",
        "ai_prediction": "AI予測 — {group}",
        "actual_value": "実際: {actual:.2f}",
        "feature_importance": "**特徴量重要度:**",
        "error_file": "⚠️ `clean_data.csv` が見つかりません。",
        "table_group": "グループ", "table_mean": "平均 ± SD", "table_n": "N",
        "table_treatment": "治療", "table_mean_val": "平均", "table_recovery": "回復 %",
        "caption": "🧪 HgCl2毒性とサフラン研究に基づく",
        "categories": {"Liver": "肝臓", "Oxidative Stress": "酸化ストレス",
                       "Neurotransmitters": "神経伝達物質", "Anti-inflammatory": "抗炎症",
                       "Kidney": "腎臓", "Lipid Profile": "脂質プロファイル"},
    },
    "ko": {
        "name": "한국어",
        "title": "🧪 수은 독성 및 사프란 회복",
        "subtitle": "대화형 대시보드 — 6개 그룹 × 21개 바이오마커",
        "sidebar_header": "🎛️ 제어",
        "language": "🌐 언어",
        "category": "카테고리 선택",
        "marker": "바이오마커 선택",
        "ai_toggle": "🤖 AI 예측",
        "recovery_toggle": "🎯 회복 지수",
        "chart_title": "📊 {marker} — 그룹별 평균",
        "stats_title": "📋 상세 통계",
        "recovery_title": "🎯 회복 지수",
        "best_treatment": "🏆 최고 치료: **{treatment}** — 회복 **{pct:.1f}%**",
        "ai_title": "🤖 AI 시뮬레이터",
        "select_group": "그룹 선택",
        "ai_prediction": "AI 예측 — {group}",
        "actual_value": "실제: {actual:.2f}",
        "feature_importance": "**특성 중요도:**",
        "error_file": "⚠️ `clean_data.csv`를 찾을 수 없습니다.",
        "table_group": "그룹", "table_mean": "평균 ± SD", "table_n": "N",
        "table_treatment": "치료", "table_mean_val": "평균", "table_recovery": "회복 %",
        "caption": "🧪 HgCl2 독성 및 사프란 연구 기반",
        "categories": {"Liver": "간", "Oxidative Stress": "산화 스트레스",
                       "Neurotransmitters": "신경전달물질", "Anti-inflammatory": "항염증",
                       "Kidney": "신장", "Lipid Profile": "지질 프로필"},
    },
    "hi": {
        "name": "हिन्दी",
        "title": "🧪 पारा विषाक्तता और केसर पुनर्प्राप्ति",
        "subtitle": "इंटरैक्टिव डैशबोर्ड — 6 समूह × 21 बायोमार्कर",
        "sidebar_header": "🎛️ नियंत्रण",
        "language": "🌐 भाषा",
        "category": "श्रेणी चुनें",
        "marker": "बायोमार्कर चुनें",
        "ai_toggle": "🤖 AI भविष्यवाणियां",
        "recovery_toggle": "🎯 पुनर्प्राप्ति सूचकांक",
        "chart_title": "📊 {marker} — प्रति समूह औसत",
        "stats_title": "📋 विस्तृत आंकड़े",
        "recovery_title": "🎯 पुनर्प्राप्ति सूचकांक",
        "best_treatment": "🏆 सर्वोत्तम उपचार: **{treatment}** — पुनर्प्राप्ति **{pct:.1f}%**",
        "ai_title": "🤖 AI सिम्युलेटर",
        "select_group": "समूह चुनें",
        "ai_prediction": "AI भविष्यवाणी — {group}",
        "actual_value": "वास्तविक: {actual:.2f}",
        "feature_importance": "**विशेषता महत्व:**",
        "error_file": "⚠️ `clean_data.csv` नहीं मिला।",
        "table_group": "समूह", "table_mean": "औसत ± SD", "table_n": "N",
        "table_treatment": "उपचार", "table_mean_val": "औसत", "table_recovery": "पुनर्प्राप्ति %",
        "caption": "🧪 HgCl2 विषाक्तता और केसर अध्ययन पर आधारित",
        "categories": {"Liver": "यकृत", "Oxidative Stress": "ऑक्सीडेटिव तनाव",
                       "Neurotransmitters": "न्यूरोट्रांसमीटर", "Anti-inflammatory": "विरोधी भड़काऊ",
                       "Kidney": "गुर्दा", "Lipid Profile": "लिपिड प्रोफाइल"},
    },
    "tr": {
        "name": "Türkçe",
        "title": "🧪 Cıva Toksisitesi ve Safran İyileşmesi",
        "subtitle": "Etkileşimli Panel — 6 grup × 21 biyobelirteç",
        "sidebar_header": "🎛️ Kontroller",
        "language": "🌐 Dil",
        "category": "Kategori seçin",
        "marker": "Biyobelirteç seçin",
        "ai_toggle": "🤖 AI Tahminleri",
        "recovery_toggle": "🎯 İyileşme Endeksi",
        "chart_title": "📊 {marker} — Grup başına ortalama",
        "stats_title": "📋 Ayrıntılı İstatistikler",
        "recovery_title": "🎯 İyileşme Endeksi",
        "best_treatment": "🏆 En İyi Tedavi: **{treatment}** — İyileşme **{pct:.1f}%**",
        "ai_title": "🤖 AI Simülatörü",
        "select_group": "Grup seçin",
        "ai_prediction": "AI Tahmini — {group}",
        "actual_value": "Gerçek: {actual:.2f}",
        "feature_importance": "**Özellik Önemi:**",
        "error_file": "⚠️ `clean_data.csv` bulunamadı.",
        "table_group": "Grup", "table_mean": "Ortalama ± SS", "table_n": "N",
        "table_treatment": "Tedavi", "table_mean_val": "Ortalama", "table_recovery": "İyileşme %",
        "caption": "🧪 HgCl2 toksisitesi ve safran çalışmasına dayanmaktadır",
        "categories": {"Liver": "Karaciğer", "Oxidative Stress": "Oksidatif Stres",
                       "Neurotransmitters": "Nörotransmitterler", "Anti-inflammatory": "Anti-enflamatuar",
                       "Kidney": "Böbrek", "Lipid Profile": "Lipid Profili"},
    },
    "fa": {
        "name": "فارسی",
        "title": "🧪 سمیت جیوه و بازیابی زعفران",
        "subtitle": "داشبورد تعاملی — ۶ گروه × ۲۱ نشانگر زیستی",
        "sidebar_header": "🎛️ کنترل‌ها",
        "language": "🌐 زبان",
        "category": "انتخاب دسته",
        "marker": "انتخاب نشانگر زیستی",
        "ai_toggle": "🤖 پیش‌بینی هوش مصنوعی",
        "recovery_toggle": "🎯 شاخص بازیابی",
        "chart_title": "📊 {marker} — میانگین هر گروه",
        "stats_title": "📋 آمار تفصیلی",
        "recovery_title": "🎯 شاخص بازیابی",
        "best_treatment": "🏆 بهترین درمان: **{treatment}** — بازیابی **{pct:.1f}%**",
        "ai_title": "🤖 شبیه‌ساز هوش مصنوعی",
        "select_group": "انتخاب گروه",
        "ai_prediction": "پیش‌بینی AI — {group}",
        "actual_value": "واقعی: {actual:.2f}",
        "feature_importance": "**اهمیت ویژگی:**",
        "error_file": "⚠️ فایل `clean_data.csv` یافت نشد.",
        "table_group": "گروه", "table_mean": "میانگین ± انحراف", "table_n": "N",
        "table_treatment": "درمان", "table_mean_val": "میانگین", "table_recovery": "بازیابی %",
        "caption": "🧪 بر اساس مطالعه سمیت HgCl2 و زعفران",
        "categories": {"Liver": "کبد", "Oxidative Stress": "استرس اکسیداتیو",
                       "Neurotransmitters": "ناقل‌های عصبی", "Anti-inflammatory": "ضد التهاب",
                       "Kidney": "کلیه", "Lipid Profile": "پروفایل چربی"},
    },
    "ur": {
        "name": "اردو",
        "title": "🧪 پارے کی زہریلا پن اور زعفران کی بحالی",
        "subtitle": "انٹرایکٹو ڈیش بورڈ — 6 گروپس × 21 بایو مارکرز",
        "sidebar_header": "🎛️ کنٹرولز",
        "language": "🌐 زبان",
        "category": "زمرہ منتخب کریں",
        "marker": "بایو مارکر منتخب کریں",
        "ai_toggle": "🤖 AI پیشین گوئیاں",
        "recovery_toggle": "🎯 بحالی انڈیکس",
        "chart_title": "📊 {marker} — فی گروپ اوسط",
        "stats_title": "📋 تفصیلی اعداد و شمار",
        "recovery_title": "🎯 بحالی انڈیکس",
        "best_treatment": "🏆 بہترین علاج: **{treatment}** — بحالی **{pct:.1f}%**",
        "ai_title": "🤖 AI سمیولیٹر",
        "select_group": "گروپ منتخب کریں",
        "ai_prediction": "AI پیشین گوئی — {group}",
        "actual_value": "اصل: {actual:.2f}",
        "feature_importance": "**خصوصیت کی اہمیت:**",
        "error_file": "⚠️ `clean_data.csv` نہیں ملا۔",
        "table_group": "گروپ", "table_mean": "اوسط ± SD", "table_n": "N",
        "table_treatment": "علاج", "table_mean_val": "اوسط", "table_recovery": "بحالی %",
        "caption": "🧪 HgCl2 زہریلا پن اور زعفران کے مطالعے پر مبنی",
        "categories": {"Liver": "جگر", "Oxidative Stress": "آکسیڈیٹو تناؤ",
                       "Neurotransmitters": "نیورو ٹرانسمیٹرز", "Anti-inflammatory": "اینٹی سوزش",
                       "Kidney": "گردہ", "Lipid Profile": "لپڈ پروفائل"},
    },
    "nl": {
        "name": "Nederlands",
        "title": "🧪 Kwiktoxiciteit & Saffraanherstel",
        "subtitle": "Interactief dashboard — 6 groepen × 21 biomarkers",
        "sidebar_header": "🎛️ Bediening",
        "language": "🌐 Taal",
        "category": "Selecteer categorie",
        "marker": "Selecteer biomarker",
        "ai_toggle": "🤖 AI-voorspellingen",
        "recovery_toggle": "🎯 Herstelindex",
        "chart_title": "📊 {marker} — Gemiddelde per groep",
        "stats_title": "📋 Gedetailleerde statistieken",
        "recovery_title": "🎯 Herstelindex",
        "best_treatment": "🏆 Beste behandeling: **{treatment}** — Herstel **{pct:.1f}%**",
        "ai_title": "🤖 AI-simulator",
        "select_group": "Selecteer groep",
        "ai_prediction": "AI-voorspelling — {group}",
        "actual_value": "Werkelijk: {actual:.2f}",
        "feature_importance": "**Functie-importantie:**",
        "error_file": "⚠️ `clean_data.csv` niet gevonden.",
        "table_group": "Groep", "table_mean": "Gemiddelde ± SD", "table_n": "N",
        "table_treatment": "Behandeling", "table_mean_val": "Gemiddelde", "table_recovery": "Herstel %",
        "caption": "🧪 Gebaseerd op HgCl2-toxiciteit en saffraanonderzoek",
        "categories": {"Liver": "Lever", "Oxidative Stress": "Oxidatieve stress",
                       "Neurotransmitters": "Neurotransmitters", "Anti-inflammatory": "Ontstekingsremmend",
                       "Kidney": "Nier", "Lipid Profile": "Lipidenprofiel"},
    },
    "pl": {
        "name": "Polski",
        "title": "🧪 Toksyczność rtęci i regeneracja szafranu",
        "subtitle": "Interaktywny panel — 6 grup × 21 biomarkerów",
        "sidebar_header": "🎛️ Sterowanie",
        "language": "🌐 Język",
        "category": "Wybierz kategorię",
        "marker": "Wybierz biomarker",
        "ai_toggle": "🤖 Predykcje AI",
        "recovery_toggle": "🎯 Indeks regeneracji",
        "chart_title": "📊 {marker} — Średnia na grupę",
        "stats_title": "📋 Szczegółowe statystyki",
        "recovery_title": "🎯 Indeks regeneracji",
        "best_treatment": "🏆 Najlepsze leczenie: **{treatment}** — Regeneracja **{pct:.1f}%**",
        "ai_title": "🤖 Symulator AI",
        "select_group": "Wybierz grupę",
        "ai_prediction": "Predykcja AI — {group}",
        "actual_value": "Rzeczywiste: {actual:.2f}",
        "feature_importance": "**Ważność cech:**",
        "error_file": "⚠️ Nie znaleziono `clean_data.csv`.",
        "table_group": "Grupa", "table_mean": "Średnia ± SD", "table_n": "N",
        "table_treatment": "Leczenie", "table_mean_val": "Średnia", "table_recovery": "Regeneracja %",
        "caption": "🧪 Na podstawie badania toksyczności HgCl2 i szafranu",
        "categories": {"Liver": "Wątroba", "Oxidative Stress": "Stres oksydacyjny",
                       "Neurotransmitters": "Neuroprzekaźniki", "Anti-inflammatory": "Przeciwzapalne",
                       "Kidney": "Nerka", "Lipid Profile": "Profil lipidowy"},
    },
    "sv": {
        "name": "Svenska",
        "title": "🧪 Kvicksilvertoxicitet & Saffransåterhämtning",
        "subtitle": "Interaktiv instrumentpanel — 6 grupper × 21 biomarkörer",
        "sidebar_header": "🎛️ Kontroller",
        "language": "🌐 Språk",
        "category": "Välj kategori",
        "marker": "Välj biomarkör",
        "ai_toggle": "🤖 AI-förutsägelser",
        "recovery_toggle": "🎯 Återhämtningsindex",
        "chart_title": "📊 {marker} — Medelvärde per grupp",
        "stats_title": "📋 Detaljerad statistik",
        "recovery_title": "🎯 Återhämtningsindex",
        "best_treatment": "🏆 Bästa behandling: **{treatment}** — Återhämtning **{pct:.1f}%**",
        "ai_title": "🤖 AI-simulator",
        "select_group": "Välj grupp",
        "ai_prediction": "AI-förutsägelse — {group}",
        "actual_value": "Faktiskt: {actual:.2f}",
        "feature_importance": "**Funktionsvikt:**",
        "error_file": "⚠️ `clean_data.csv` hittades inte.",
        "table_group": "Grupp", "table_mean": "Medelvärde ± SD", "table_n": "N",
        "table_treatment": "Behandling", "table_mean_val": "Medelvärde", "table_recovery": "Återhämtning %",
        "caption": "🧪 Baserat på HgCl2-toxicitet och saffransstudie",
        "categories": {"Liver": "Lever", "Oxidative Stress": "Oxidativ stress",
                       "Neurotransmitters": "Neurotransmittorer", "Anti-inflammatory": "Antiinflammatorisk",
                       "Kidney": "Njure", "Lipid Profile": "Lipidprofil"},
    },
    "el": {
        "name": "Ελληνικά",
        "title": "🧪 Τοξικότητα υδραργύρου & ανάκαμψη σαφράν",
        "subtitle": "Διαδραστικός πίνακας — 6 ομάδες × 21 βιοδείκτες",
        "sidebar_header": "🎛️ Έλεγχοι",
        "language": "🌐 Γλώσσα",
        "category": "Επιλογή κατηγορίας",
        "marker": "Επιλογή βιοδείκτη",
        "ai_toggle": "🤖 Προβλέψεις AI",
        "recovery_toggle": "🎯 Δείκτης ανάκαμψης",
        "chart_title": "📊 {marker} — Μέσος όρος ανά ομάδα",
        "stats_title": "📋 Λεπτομερή στατιστικά",
        "recovery_title": "🎯 Δείκτης ανάκαμψης",
        "best_treatment": "🏆 Καλύτερη θεραπεία: **{treatment}** — Ανάκαμψη **{pct:.1f}%**",
        "ai_title": "🤖 Προσομοιωτής AI",
        "select_group": "Επιλογή ομάδας",
        "ai_prediction": "Πρόβλεψη AI — {group}",
        "actual_value": "Πραγματικό: {actual:.2f}",
        "feature_importance": "**Σημασία χαρακτηριστικών:**",
        "error_file": "⚠️ Το `clean_data.csv` δεν βρέθηκε.",
        "table_group": "Ομάδα", "table_mean": "Μέσος ± ΤΑ", "table_n": "N",
        "table_treatment": "Θεραπεία", "table_mean_val": "Μέσος", "table_recovery": "Ανάκαμψη %",
        "caption": "🧪 Βασισμένο στη μελέτη τοξικότητας HgCl2 και σαφράν",
        "categories": {"Liver": "Ήπαρ", "Oxidative Stress": "Οξειδωτικό στρες",
                       "Neurotransmitters": "Νευροδιαβιβαστές", "Anti-inflammatory": "Αντιφλεγμονώδες",
                       "Kidney": "Νεφρός", "Lipid Profile": "Λιπιδικό προφίλ"},
    },
    "id": {
        "name": "Bahasa Indonesia",
        "title": "🧪 Toksisitas Merkuri & Pemulihan Saffron",
        "subtitle": "Dasbor Interaktif — 6 grup × 21 biomarker",
        "sidebar_header": "🎛️ Kontrol",
        "language": "🌐 Bahasa",
        "category": "Pilih kategori",
        "marker": "Pilih biomarker",
        "ai_toggle": "🤖 Prediksi AI",
        "recovery_toggle": "🎯 Indeks Pemulihan",
        "chart_title": "📊 {marker} — Rata-rata per Grup",
        "stats_title": "📋 Statistik Detail",
        "recovery_title": "🎯 Indeks Pemulihan",
        "best_treatment": "🏆 Perawatan Terbaik: **{treatment}** — Pemulihan **{pct:.1f}%**",
        "ai_title": "🤖 Simulator AI",
        "select_group": "Pilih grup",
        "ai_prediction": "Prediksi AI — {group}",
        "actual_value": "Aktual: {actual:.2f}",
        "feature_importance": "**Kepentingan Fitur:**",
        "error_file": "⚠️ `clean_data.csv` tidak ditemukan.",
        "table_group": "Grup", "table_mean": "Rata-rata ± SD", "table_n": "N",
        "table_treatment": "Perawatan", "table_mean_val": "Rata-rata", "table_recovery": "Pemulihan %",
        "caption": "🧪 Berdasarkan studi toksisitas HgCl2 dan saffron",
        "categories": {"Liver": "Hati", "Oxidative Stress": "Stres Oksidatif",
                       "Neurotransmitters": "Neurotransmitter", "Anti-inflammatory": "Anti-inflamasi",
                       "Kidney": "Ginjal", "Lipid Profile": "Profil Lipid"},
    },
    "th": {
        "name": "ไทย",
        "title": "🧪 ความเป็นพิษของปรอทและการฟื้นตัวของหญ้าฝรั่น",
        "subtitle": "แดชบอร์ดแบบโต้ตอบ — 6 กลุ่ม × 21 ไบโอมาร์คเกอร์",
        "sidebar_header": "🎛️ การควบคุม",
        "language": "🌐 ภาษา",
        "category": "เลือกหมวดหมู่",
        "marker": "เลือกไบโอมาร์คเกอร์",
        "ai_toggle": "🤖 การคาดการณ์ AI",
        "recovery_toggle": "🎯 ดัชนีการฟื้นตัว",
        "chart_title": "📊 {marker} — ค่าเฉลี่ยต่อกลุ่ม",
        "stats_title": "📋 สถิติโดยละเอียด",
        "recovery_title": "🎯 ดัชนีการฟื้นตัว",
        "best_treatment": "🏆 การรักษาที่ดีที่สุด: **{treatment}** — การฟื้นตัว **{pct:.1f}%**",
        "ai_title": "🤖 ตัวจำลอง AI",
        "select_group": "เลือกกลุ่ม",
        "ai_prediction": "การคาดการณ์ AI — {group}",
        "actual_value": "จริง: {actual:.2f}",
        "feature_importance": "**ความสำคัญของคุณลักษณะ:**",
        "error_file": "⚠️ ไม่พบ `clean_data.csv`",
        "table_group": "กลุ่ม", "table_mean": "ค่าเฉลี่ย ± SD", "table_n": "N",
        "table_treatment": "การรักษา", "table_mean_val": "ค่าเฉลี่ย", "table_recovery": "การฟื้นตัว %",
        "caption": "🧪 อ้างอิงจากการศึกษาความเป็นพิษของ HgCl2 และหญ้าฝรั่น",
        "categories": {"Liver": "ตับ", "Oxidative Stress": "ความเครียดออกซิเดชัน",
                       "Neurotransmitters": "สารสื่อประสาท", "Anti-inflammatory": "ต้านการอักเสบ",
                       "Kidney": "ไต", "Lipid Profile": "โปรไฟล์ไขมัน"},
    },
    "vi": {
        "name": "Tiếng Việt",
        "title": "🧪 Độc tính thủy ngân & phục hồi nghệ tây",
        "subtitle": "Bảng điều khiển tương tác — 6 nhóm × 21 dấu ấn sinh học",
        "sidebar_header": "🎛️ Điều khiển",
        "language": "🌐 Ngôn ngữ",
        "category": "Chọn danh mục",
        "marker": "Chọn dấu ấn sinh học",
        "ai_toggle": "🤖 Dự đoán AI",
        "recovery_toggle": "🎯 Chỉ số phục hồi",
        "chart_title": "📊 {marker} — Trung bình mỗi nhóm",
        "stats_title": "📋 Thống kê chi tiết",
        "recovery_title": "🎯 Chỉ số phục hồi",
        "best_treatment": "🏆 Điều trị tốt nhất: **{treatment}** — Phục hồi **{pct:.1f}%**",
        "ai_title": "🤖 Trình mô phỏng AI",
        "select_group": "Chọn nhóm",
        "ai_prediction": "Dự đoán AI — {group}",
        "actual_value": "Thực tế: {actual:.2f}",
        "feature_importance": "**Tầm quan trọng tính năng:**",
        "error_file": "⚠️ Không tìm thấy `clean_data.csv`.",
        "table_group": "Nhóm", "table_mean": "Trung bình ± SD", "table_n": "N",
        "table_treatment": "Điều trị", "table_mean_val": "Trung bình", "table_recovery": "Phục hồi %",
        "caption": "🧪 Dựa trên nghiên cứu độc tính HgCl2 và nghệ tây",
        "categories": {"Liver": "Gan", "Oxidative Stress": "Stress oxy hóa",
                       "Neurotransmitters": "Chất dẫn truyền thần kinh", "Anti-inflammatory": "Chống viêm",
                       "Kidney": "Thận", "Lipid Profile": "Hồ sơ lipid"},
    },
}

# ============================================================
# اختيار اللغة
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
# العنوان
# ============================================================
st.markdown(f"""
    <h1 style='text-align:center; color:#2c3e50;'>{t['title']}</h1>
    <p style='text-align:center; color:#7f8c8d;'>{t['subtitle']}</p>
""", unsafe_allow_html=True)

# ============================================================
# تحميل البيانات
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
# الشريط الجانبي
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
# تصفية البيانات
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
# الرسم البياني
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
# الإحصاءات
# ============================================================
st.subheader(t["stats_title"])
stats = summary.copy()
stats.columns = [t["table_group"], "Mean", "SD", t["table_n"]]
stats[t["table_mean"]] = stats.apply(lambda r: f"{r['Mean']:.2f} ± {r['SD']:.2f}", axis=1)
st.dataframe(stats[[t["table_group"], t["table_mean"], t["table_n"]]], use_container_width=True)

# ============================================================
# مؤشر التعافي
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
# الذكاء الاصطناعي
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

st.divider()
st.caption(t["caption"])
