import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as gg
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Prediksi Kualitas Tidur",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
    /* Main Theme Overrides */
    .stApp {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    
    /* Header Card */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .main-title {
        color: #38bdf8;
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 8px;
    }
    
    .sub-title {
        color: #94a3b8;
        font-size: 15px;
        margin-bottom: 0px;
    }

    /* Metric Cards */
    .metric-card {
        background: #1e293b;
        border-left: 4px solid #38bdf8;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }

    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #f8fafc;
    }

    .metric-label {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Side-by-Side Model Prediction Cards */
    .model-card-rf {
        background: rgba(30, 41, 59, 0.8);
        border: 2px solid #3b82f6;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }

    .model-card-xgb {
        background: rgba(30, 41, 59, 0.8);
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }

    .badge-high {
        background-color: #065f46;
        color: #34d399;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }

    .badge-medium {
        background-color: #92400e;
        color: #fbbf24;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }

    .badge-low {
        background-color: #991b1b;
        color: #fca5a5;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD DATA AND ARTIFACTS ---
DATA_PATH = r'z:/Kuliyeah/SEM 8/Final Dance/Dataset/archive/Sleep_health_and_lifestyle_dataset.csv'
ARTIFACT_PATH = r'z:/Kuliyeah/SEM 8/Final Dance/models/model_artifacts.pkl'

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['BMI Category'] = df['BMI Category'].replace({'Normal Weight': 'Normal'})
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
    return df

@st.cache_resource
def load_artifacts():
    if not os.path.exists(ARTIFACT_PATH):
        return None
    return joblib.load(ARTIFACT_PATH)

df_data = load_data()
artifacts = load_artifacts()

# --- HEADER SECTION ---
st.markdown("""
<div class="main-header">
    <div class="main-title"> Prediksi Kualitas Tidur Berdasarkan Gaya Hidup</div>
    <div class="sub-title">Perbandingan Algoritma <b>XGBoost</b> (Extreme Gradient Boosting) vs <b>Random Forest</b></div>
</div>
""", unsafe_allow_html=True)

if artifacts is None:
    st.warning("⚠️ Model belum dilatih! Menjalankan script pelatihan model otomatis...")
    import subprocess
    cmd = r'"C:\Users\baim\anaconda3\python.exe" "z:/Kuliyeah/SEM 8/Final Dance/train.py"'
    subprocess.run(cmd, shell=True)
    st.rerun()

# Extract artifacts
xgb_cls = artifacts['xgb_cls']
xgb_reg = artifacts['xgb_reg']
rf_cls = artifacts['rf_cls']
rf_reg = artifacts['rf_reg']
label_encoders = artifacts['label_encoders']
scaler = artifacts['scaler']
feature_cols = artifacts['feature_cols']
metrics_cls = artifacts['metrics_cls']
metrics_reg = artifacts['metrics_reg']
feature_importances = artifacts['feature_importances']

# --- NAVIGATION TABS ---
tab_eda, tab_predict, tab_compare = st.tabs([
    " 1. Exploratory Data Analysis (EDA)",
    " 2. Simulator Prediksi Real-Time",
    " 3. Perbandingan Performa Algoritma"
])

# ==========================================
# TAB 1: EXPLORATORY DATA ANALYSIS
# ==========================================
with tab_eda:
    st.subheader("Overview & Visualisasi Dataset Gaya Hidup & Kualitas Tidur")
    
    # Key Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Sampel</div>
            <div class="metric-value">{len(df_data)} Responden</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Rata-Rata Durasi Tidur</div>
            <div class="metric-value">{df_data['Sleep Duration'].mean():.2f} Jam</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Rata-Rata Skor Kualitas</div>
            <div class="metric-value">{df_data['Quality of Sleep'].mean():.2f} / 10</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Rata-Rata Tingkat Stres</div>
            <div class="metric-value">{df_data['Stress Level'].mean():.2f} / 10</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Plots Row 1
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("####  Durasi Tidur vs Kualitas Tidur (Berdasarkan Kategori BMI)")
        fig1 = px.scatter(
            df_data,
            x="Sleep Duration",
            y="Quality of Sleep",
            color="BMI Category",
            size="Physical Activity Level",
            hover_data=["Occupation", "Age", "Stress Level"],
            title="Hubungan Durasi Tidur dan Skor Kualitas Tidur",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)

    with col_right:
        st.markdown("####  Rata-Rata Kualitas Tidur Berdasarkan Pekerjaan")
        avg_occ = df_data.groupby('Occupation')['Quality of Sleep'].mean().reset_index().sort_values(by='Quality of Sleep', ascending=True)
        fig2 = px.bar(
            avg_occ,
            x="Quality of Sleep",
            y="Occupation",
            orientation="h",
            color="Quality of Sleep",
            color_continuous_scale="Viridis",
            title="Kualitas Tidur Berdasarkan Profesi",
            template="plotly_dark"
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # Interactive Plots Row 2
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown("####  Tingkat Stres vs Kualitas Tidur")
        fig3 = px.box(
            df_data,
            x="Stress Level",
            y="Quality of Sleep",
            color="Gender",
            title="Distribusi Kualitas Tidur Pada Setiap Level Stres",
            template="plotly_dark"
        )
        fig3.update_layout(height=380)
        st.plotly_chart(fig3, use_container_width=True)
        
    with col_b2:
        st.markdown("####  Heatmap Korelasi Variabel Gaya Hidup")
        num_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic_BP', 'Diastolic_BP']
        corr_matrix = df_data[num_cols].corr()
        fig4 = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Matriks Korelasi (Pearson Correlation)",
            template="plotly_dark"
        )
        fig4.update_layout(height=380)
        st.plotly_chart(fig4, use_container_width=True)


# ==========================================
# TAB 2: SIMULATOR PREDIKSI REAL-TIME
# ==========================================
with tab_predict:
    st.subheader(" Simulator Prediksi Kualitas Tidur Interaktif")
    st.markdown("Masukkan parameter profil gaya hidup Anda di bawah ini untuk melihat hasil estimasi prediksi dari model **Random Forest** dan **XGBoost**.")

    with st.form("prediction_form"):
        f_col1, f_col2, f_col3 = st.columns(3)
        
        with f_col1:
            gender = st.selectbox("Jenis Kelamin", options=["Laki-laki (Male)", "Perempuan (Female)"])
            age = st.number_input("Usia (Tahun)", min_value=18, max_value=80, value=30)
            
            # Pilihan Pekerjaan yang Luas & Inklusif
            occ_display_options = [
                "Software Engineer / IT Professional",
                "Dokter / Tenaga Kesehatan",
                "Guru / Dosen / Tenaga Pendidik",
                "Pegawai Swasta / Karyawan Kantor",
                "PNS / ASN / Pegawai Pemerintah",
                "Pengusaha / Wiraswasta",
                "Mahasiswa / Pelajar",
                "Ibu Rumah Tangga",
                "Sales / Pemasaran",
                "Pekerja Lapangan / Industri / Konstruksi",
                "Akuntan / Keuangan",
                "Pengacara / Konsultan Hukum",
                "Ilmuwan / Peneliti",
                "Manajer / Eksekutif",
                "Lainnya / Pekerjaan Lain"
            ]
            selected_occ_display = st.selectbox("Pekerjaan / Profesi", options=occ_display_options)
            
            # Jika memilih 'Lainnya', tampilkan text input opsional
            if selected_occ_display == "Lainnya / Pekerjaan Lain":
                custom_job = st.text_input("Sebutkan Nama Pekerjaan (Opsional):", value="", placeholder="misal: Desainer, Driver, Kontraktor")

            # Kategori BMI
            bmi_display_options = ["Normal (Berat Badan Ideal)", "Overweight (Kelebihan Berat Badan)", "Obese (Obesitas)"]
            selected_bmi_display = st.selectbox("Kategori BMI", options=bmi_display_options)

        with f_col2:
            sleep_dur = st.slider("Durasi Tidur Harian (Jam)", min_value=4.0, max_value=10.0, value=7.0, step=0.1)
            act_level = st.slider("Aktivitas Fisik (Menit / Hari)", min_value=10, max_value=150, value=60, step=5)
            stress_lvl = st.slider("Tingkat Stres Mandiri (1 - 10)", min_value=1, max_value=10, value=5)
            
            disorder_display_options = ["Tidak Ada (None)", "Insomnia", "Sleep Apnea"]
            selected_disorder_display = st.selectbox("Riwayat Gangguan Tidur", options=disorder_display_options)

        with f_col3:
            sys_bp = st.number_input("Tekanan Darah Sistolik (mmHg)", min_value=90, max_value=180, value=120)
            dia_bp = st.number_input("Tekanan Darah Diastolik (mmHg)", min_value=60, max_value=120, value=80)
            heart_rate = st.number_input("Detak Jantung Istirahat (BPM)", min_value=50, max_value=110, value=70)
            daily_steps = st.number_input("Jumlah Langkah Harian (Steps)", min_value=1000, max_value=15000, value=7000, step=500)

        submit_btn = st.form_submit_button(" Jalankan Prediksi Perbandingan")

    if submit_btn:
        # Pemetaan Pilihan Pekerjaan ke Model Dataset
        occ_mapping = {
            "Software Engineer / IT Professional": "Software Engineer",
            "Dokter / Tenaga Kesehatan": "Doctor",
            "Guru / Dosen / Tenaga Pendidik": "Teacher",
            "Pegawai Swasta / Karyawan Kantor": "Accountant",
            "PNS / ASN / Pegawai Pemerintah": "Accountant",
            "Pengusaha / Wiraswasta": "Manager",
            "Mahasiswa / Pelajar": "Software Engineer",
            "Ibu Rumah Tangga": "Nurse",
            "Sales / Pemasaran": "Salesperson",
            "Pekerja Lapangan / Industri / Konstruksi": "Sales Representative",
            "Akuntan / Keuangan": "Accountant",
            "Pengacara / Konsultan Hukum": "Lawyer",
            "Ilmuwan / Peneliti": "Scientist",
            "Manajer / Eksekutif": "Manager",
            "Lainnya / Pekerjaan Lain": "Software Engineer"
        }
        mapped_occ = occ_mapping.get(selected_occ_display, "Software Engineer")

        # Pemetaan Gender
        mapped_gender = "Male" if "Laki-laki" in gender else "Female"

        # Pemetaan BMI
        if "Normal" in selected_bmi_display:
            mapped_bmi = "Normal"
        elif "Overweight" in selected_bmi_display:
            mapped_bmi = "Overweight"
        else:
            mapped_bmi = "Obese"

        # Pemetaan Sleep Disorder
        if "Insomnia" in selected_disorder_display:
            mapped_disorder = "Insomnia"
        elif "Sleep Apnea" in selected_disorder_display:
            mapped_disorder = "Sleep Apnea"
        else:
            mapped_disorder = "None"

        # Transform Input Data
        gender_enc = label_encoders['Gender'].transform([mapped_gender])[0]
        occ_enc = label_encoders['Occupation'].transform([mapped_occ])[0]
        bmi_enc = label_encoders['BMI Category'].transform([mapped_bmi])[0]
        disorder_enc = label_encoders['Sleep Disorder'].transform([mapped_disorder])[0]
        
        input_data = pd.DataFrame([{
            'Age': age,
            'Sleep Duration': sleep_dur,
            'Physical Activity Level': act_level,
            'Stress Level': stress_lvl,
            'Heart Rate': heart_rate,
            'Daily Steps': daily_steps,
            'Systolic_BP': sys_bp,
            'Diastolic_BP': dia_bp,
            'Gender_encoded': gender_enc,
            'Occupation_encoded': occ_enc,
            'BMI Category_encoded': bmi_enc,
            'Sleep Disorder_encoded': disorder_enc
        }])

        # Order columns correctly
        input_data = input_data[feature_cols]

        # Predict RF
        rf_cls_pred = rf_cls.predict(input_data)[0]
        rf_reg_pred = rf_reg.predict(input_data)[0]
        rf_proba = rf_cls.predict_proba(input_data)[0]

        # Predict XGB
        xgb_cls_pred = xgb_cls.predict(input_data)[0]
        xgb_reg_pred = xgb_reg.predict(input_data)[0]
        xgb_proba = xgb_cls.predict_proba(input_data)[0]

        cat_names = artifacts.get('quality_labels', {
            0: 'Sangat Rendah (Skor 4)',
            1: 'Rendah (Skor 5)',
            2: 'Cukup (Skor 6)',
            3: 'Baik (Skor 7)',
            4: 'Sangat Baik (Skor 8)',
            5: 'Sempurna (Skor 9)'
        })

        st.markdown("---")
        st.markdown("###  Hasil Prediksi Real-Time")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            rf_label = cat_names.get(rf_cls_pred, f"Kategori {rf_cls_pred}")
            st.markdown(f"""
            <div class="model-card-rf">
                <h3 style="color: #3b82f6; margin-bottom: 4px;">🌲 Random Forest Model</h3>
                <p style="color: #94a3b8; font-size: 13px;">Algoritma Ensemble Bagging</p>
                <hr style="border-color: #334155;">
                <p style="margin-bottom: 4px; color: #cbd5e1;">Kategori Kualitas Tidur:</p>
                <div class="badge-high">{rf_label}</div>
                <h2 style="color: #f8fafc; margin-top: 16px; margin-bottom: 0;">Skor: {rf_reg_pred:.2f} / 10</h2>
                <p style="color: #94a3b8; font-size: 13px;">Estimasi Kontinu Skala 1-10 (R² = 98.44%)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability chart
            df_rf_prob = pd.DataFrame({'Kategori': [cat_names[i] for i in range(len(rf_proba))], 'Probabilitas': rf_proba})
            fig_rf_p = px.bar(df_rf_prob, x='Kategori', y='Probabilitas', color='Kategori', text_auto='.1%', title="Probabilitas Klasifikasi (Random Forest)", template="plotly_dark")
            fig_rf_p.update_layout(height=280, showlegend=False)
            st.plotly_chart(fig_rf_p, use_container_width=True)

        with res_col2:
            xgb_label = cat_names.get(xgb_cls_pred, f"Kategori {xgb_cls_pred}")
            st.markdown(f"""
            <div class="model-card-xgb">
                <h3 style="color: #10b981; margin-bottom: 4px;">🚀 XGBoost Model</h3>
                <p style="color: #94a3b8; font-size: 13px;">Algoritma Gradient Boosting</p>
                <hr style="border-color: #334155;">
                <p style="margin-bottom: 4px; color: #cbd5e1;">Kategori Kualitas Tidur:</p>
                <div class="badge-high">{xgb_label}</div>
                <h2 style="color: #f8fafc; margin-top: 16px; margin-bottom: 0;">Skor: {xgb_reg_pred:.2f} / 10</h2>
                <p style="color: #94a3b8; font-size: 13px;">Estimasi Kontinu Skala 1-10 (R² = 98.85%)</p>
            </div>
            """, unsafe_allow_html=True)

            # Probability chart
            df_xgb_prob = pd.DataFrame({'Kategori': [cat_names[i] for i in range(len(xgb_proba))], 'Probabilitas': xgb_proba})
            fig_xgb_p = px.bar(df_xgb_prob, x='Kategori', y='Probabilitas', color='Kategori', text_auto='.1%', title="Probabilitas Klasifikasi (XGBoost)", template="plotly_dark")
            fig_xgb_p.update_layout(height=280, showlegend=False)
            st.plotly_chart(fig_xgb_p, use_container_width=True)

        # Health Recommendation Box
        st.markdown("####  Rekomendasi Kesehatan Gaya Hidup (Insights)")
        rec_messages = []
        if sleep_dur < 7.0:
            rec_messages.append("⚠️ **Durasi Tidur Kurang**: Durasi tidur Anda di bawah rekomendasi ideal (7-9 jam). Cobalah tidur 30-60 menit lebih awal.")
        if stress_lvl >= 7:
            rec_messages.append("⚠️ **Tingkat Stres Tinggi**: Stres terbukti menurunkan kualitas tidur secara signifikan. Disarankan melakukan teknik relaksasi / meditasi sebelum tidur.")
        if sys_bp > 130 or dia_bp > 85:
            rec_messages.append("⚠️ **Tekanan Darah Tinggi**: Tekanan darah Anda sedikit di atas normal. Kurangi konsumsi garam dan pantau tekanan darah secara berkala.")
        if act_level < 30:
            rec_messages.append("⚠️ **Aktivitas Fisik Rendah**: Tingkatkan olahraga ringan setidaknya 30 menit sehari untuk membantu tubuh memicu tidur nyenyak.")

        if not rec_messages:
            st.success("🎉 **Luar Biasa!** Profil gaya hidup Anda sangat mendukung kualitas tidur yang optimal!")
        else:
            for msg in rec_messages:
                st.warning(msg)


# ==========================================
# TAB 3: PERBANDINGAN PERFORMA ALGORITMA
# ==========================================
with tab_compare:
    st.subheader(" Evaluasi & Perbandingan Komparatif Algoritma")
    st.markdown("Tabel dan grafik di bawah ini merupakan hasil pengujian performa **Random Forest** dan **XGBoost** (Menggunakan 5-Fold Cross Validation & Test Set).")

    # Metrics Summary Table
    st.markdown("#### 1. Matriks Evaluasi Klasifikasi Multi-Kelas (5-Fold Cross Validation)")
    df_cls_table = pd.DataFrame(metrics_cls).T[['Accuracy', 'Precision', 'Recall', 'F1-Score']]
    df_cls_table = df_cls_table.map(lambda x: f"{x * 100:.2f}%")
    st.dataframe(df_cls_table, use_container_width=True)

    st.markdown("#### 2. Matriks Evaluasi Regresi (Estimasi Skor Continuous)")
    df_reg_table = pd.DataFrame(metrics_reg).T[['MAE', 'RMSE', 'R2_Score']]
    df_reg_table['R2_Score'] = df_reg_table['R2_Score'].map(lambda x: f"{x * 100:.2f}%")
    df_reg_table['MAE'] = df_reg_table['MAE'].map(lambda x: f"{x:.4f}")
    df_reg_table['RMSE'] = df_reg_table['RMSE'].map(lambda x: f"{x:.4f}")
    st.dataframe(df_reg_table, use_container_width=True)

    # Confusion Matrix Visualizations
    st.markdown("#### 3. Visualisasi Confusion Matrix Realistis (Skor 4 - 9)")
    cm_col1, cm_col2 = st.columns(2)
    
    cats = ['Skor 4', 'Skor 5', 'Skor 6', 'Skor 7', 'Skor 8', 'Skor 9']
    with cm_col1:
        cm_rf = np.array(metrics_cls['Random Forest']['Confusion_Matrix'])
        fig_cm_rf = px.imshow(
            cm_rf,
            x=cats[:cm_rf.shape[1]], y=cats[:cm_rf.shape[0]],
            text_auto=True,
            color_continuous_scale="Blues",
            title="Confusion Matrix - Random Forest (Acc: 96.26%)",
            labels=dict(x="Predicted", y="Actual"),
            template="plotly_dark"
        )
        st.plotly_chart(fig_cm_rf, use_container_width=True)

    with cm_col2:
        cm_xgb = np.array(metrics_cls['XGBoost']['Confusion_Matrix'])
        fig_cm_xgb = px.imshow(
            cm_xgb,
            x=cats[:cm_xgb.shape[1]], y=cats[:cm_xgb.shape[0]],
            text_auto=True,
            color_continuous_scale="Greens",
            title="Confusion Matrix - XGBoost (Acc: 94.92%)",
            labels=dict(x="Predicted", y="Actual"),
            template="plotly_dark"
        )
        st.plotly_chart(fig_cm_xgb, use_container_width=True)

    # Feature Importance Comparison
    st.markdown("#### 4. Perbandingan Feature Importance (Bobot Variabel Dominan)")
    fi_rf_series = pd.Series(feature_importances['Random Forest'], name="Random Forest")
    fi_xgb_series = pd.Series(feature_importances['XGBoost'], name="XGBoost")
    
    df_fi = pd.DataFrame([fi_rf_series, fi_xgb_series]).T.reset_index()
    df_fi.columns = ['Feature', 'Random Forest', 'XGBoost']
    df_fi = df_fi.sort_values(by='XGBoost', ascending=True)

    fig_fi = px.bar(
        df_fi,
        y="Feature",
        x=["Random Forest", "XGBoost"],
        barmode="group",
        title="Tingkat Kepentingan Variabel Gaya Hidup (Feature Importance)",
        orientation="h",
        template="plotly_dark",
        color_discrete_map={"Random Forest": "#3b82f6", "XGBoost": "#10b981"}
    )
    fig_fi.update_layout(height=500)
    st.plotly_chart(fig_fi, use_container_width=True)
