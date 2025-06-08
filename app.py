import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Dashboard Analisis Nasabah Bank",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mengatur gaya visualisasi
sns.set_theme(style="whitegrid")

# --- FUNGSI UNTUK MEMUAT ARTEFAK DENGAN CACHING ---
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error saat memuat file model di '{path}': {e}")
        return None

@st.cache_data
def load_data(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Error saat memuat file CSV di '{path}': {e}")
        return None

@st.cache_data
def load_object(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error saat memuat file objek di '{path}': {e}")
        return None

# --- MEMUAT SEMUA ARTEFAK ---
# Gunakan nama file final _v2 yang sudah kita pastikan benar
LOG_REG_MODEL_PATH = 'artifacts/log_reg_model_final_v2.pkl'
SCALER_PATH = 'artifacts/scaler_lr_final_v2.pkl'
X_COLUMNS_PATH = 'artifacts/X_reg_columns_final_v2.pkl'
CLUSTERED_DATA_PATH = 'artifacts/df_clustered_final_k4.csv' # Pastikan nama ini sesuai
CAT_OPTIONS_PATH = 'artifacts/categorical_options_final_v2.pkl'

log_reg_model = load_model(LOG_REG_MODEL_PATH)
scaler = load_model(SCALER_PATH)
X_reg_columns = load_object(X_COLUMNS_PATH)
df_profile_kmeans = load_data(CLUSTERED_DATA_PATH)
categorical_options = load_object(CAT_OPTIONS_PATH)

# Fallback jika categorical_options tidak dimuat
if categorical_options is None:
    st.warning("File 'categorical_options.pkl' tidak dimuat. Menggunakan opsi default.")
    categorical_options = {
        'job': ['management', 'technician', 'entrepreneur', 'blue-collar', 'retired', 'admin.', 'services', 'self-employed', 'unemployed', 'housemaid', 'student'],
        'marital': ['married', 'single', 'divorced'],
        'education': ['secondary', 'tertiary', 'primary', 'unknown'],
        'default': ['no', 'yes'], 'housing': ['no', 'yes'], 'loan': ['no', 'yes'],
        'contact': ['cellular', 'telephone', 'unknown'],
        'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
        'poutcome': ['unknown', 'failure', 'success', 'other']
    }

# --- Sidebar ---
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Analisis Segmentasi (K-Means)", "Alat Prediksi (Regresi Logistik)"])
st.sidebar.info(
    "Dashboard ini dirancang untuk menganalisis dan memprediksi "
    "probabilitas langganan deposito berjangka oleh nasabah."
)

# --- JUDUL UTAMA ---
st.title("🏦 Dashboard Analisis Nasabah Bank")
st.markdown("Gunakan panel navigasi di sebelah kiri untuk memilih antara analisis segmentasi dan alat prediksi.")
st.markdown("---")

# ==============================================================================
#                          HALAMAN 1: ANALISIS SEGMENTASI
# ==============================================================================
if page == "Analisis Segmentasi (K-Means)":
    st.header("Analisis Segmentasi Nasabah (k=4)")
    st.markdown("Bagian ini menyajikan karakteristik dari 4 segmen nasabah yang ditemukan menggunakan algoritma K-Means.")

    if df_profile_kmeans is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Distribusi Ukuran Segmen")
            cluster_counts = df_profile_kmeans['cluster'].value_counts().sort_index()
            fig_pie, ax_pie = plt.subplots(figsize=(7, 6))
            ax_pie.pie(cluster_counts, labels=[f'Cluster {i}' for i in cluster_counts.index], autopct='%1.1f%%',
                       startangle=90, colors=sns.color_palette("viridis", len(cluster_counts)))
            ax_pie.axis('equal')
            st.pyplot(fig_pie)

        with col2:
            st.subheader("Tingkat Konversi per Segmen")
            if 'deposit_encoded' in df_profile_kmeans.columns:
                conversion_rates = df_profile_kmeans.groupby('cluster')['deposit_encoded'].mean().sort_index()
                fig_bar, ax_bar = plt.subplots(figsize=(7, 6))
                sns.barplot(x=conversion_rates.index, y=conversion_rates.values, ax=ax_bar, palette="viridis")
                ax_bar.set_ylabel('Rata-rata Tingkat Konversi')
                ax_bar.set_xlabel('Cluster')
                ax_bar.set_ylim(0, 1.0)
                st.pyplot(fig_bar)

        st.markdown("---")
        st.subheader("Telusuri Profil Detail Setiap Segmen")
        selected_cluster = st.selectbox("Pilih Segmen:", options=sorted(df_profile_kmeans['cluster'].unique()))

        with st.container(border=True):
            profile_data = df_profile_kmeans[df_profile_kmeans['cluster'] == selected_cluster]
            st.markdown(f"#### Profil Detail untuk **Cluster {selected_cluster}**")

            c1, c2, c3 = st.columns(3)
            c1.metric("Tingkat Konversi", f"{profile_data['deposit_encoded'].mean():.1%}")
            c2.metric("Jumlah Nasabah", f"{len(profile_data):,}")
            c3.metric("Rata-rata Usia", f"{profile_data['age'].mean():.1f} tahun")

            num_col, cat_col = st.columns(2)
            with num_col:
                st.markdown("**Statistik Numerik (Rata-rata):**")
                # === PERBAIKAN DI SINI ===
                # Membuat daftar kolom numerik yang mungkin ada, lalu memfilternya
                possible_numerical_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
                available_cols = [col for col in possible_numerical_cols if col in profile_data.columns]

                if available_cols:
                    st.dataframe(profile_data[available_cols].mean().round(2).reset_index().rename(columns={0: 'Rata-rata', 'index':'Fitur'}), use_container_width=True)
                else:
                    st.warning("Tidak ada kolom numerik ('age', 'balance', dll.) yang ditemukan di data profil.")

            with cat_col:
                st.markdown("**Pekerjaan Dominan (Top 3):**")
                st.dataframe(profile_data['job'].value_counts(normalize=True).head(3).mul(100).round(1).astype(str) + '%', use_container_width=True)
    else:
        st.error("Data profil K-Means tidak dapat dimuat. Pastikan file 'df_clustered_final_k4_v2.csv' sudah diunggah.")

# ==============================================================================
#                       HALAMAN 2: ALAT PREDIKSI
# ==============================================================================
elif page == "Alat Prediksi (Regresi Logistik)":
    st.header("⚙️ Alat Prediksi Langganan Deposito")
    st.markdown("Masukkan data nasabah di bawah ini untuk menghitung probabilitas berlangganan.")

    if not (log_reg_model and scaler and X_reg_columns):
        st.error("Model prediksi tidak dapat dimuat. Pastikan semua file artefak (.pkl) sudah benar.")
    else:
        with st.form("prediction_form_final"):
            st.subheader("Data Diri & Finansial Nasabah")
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("Usia", 18, 100, 40)
                job = st.selectbox("Pekerjaan", categorical_options.get('job', []))
            with col2:
                marital = st.selectbox("Status Pernikahan", categorical_options.get('marital', []))
                education = st.selectbox("Pendidikan", categorical_options.get('education', []))
            with col3:
                balance = st.number_input("Saldo Saat Ini", value=500, step=100)

            st.markdown("---")
            st.subheader("Data Riwayat Pinjaman & Kontak")
            col4, col5, col6 = st.columns(3)
            with col4:
                default = st.selectbox("Gagal Bayar?", ['no', 'yes'])
                housing = st.selectbox("Punya KPR?", ['no', 'yes'])
                loan = st.selectbox("Punya Pinjaman Pribadi?", ['no', 'yes'])
            with col5:
                contact = st.selectbox("Tipe Kontak Terakhir", categorical_options.get('contact', []))
                day = st.number_input("Tanggal Kontak", 1, 31, 15)
                month = st.selectbox("Bulan Kontak", categorical_options.get('month', []))
            with col6:
                campaign = st.number_input("Jumlah Kontak (Kampanye Ini)", min_value=1, value=1)
                pdays = st.number_input("Hari Sejak Kontak Terakhir (Lalu)", value=-1)
                previous = st.number_input("Jumlah Kontak (Sebelumnya)", min_value=0, value=0)
                poutcome = st.selectbox("Hasil Kampanye Sebelumnya", categorical_options.get('poutcome', []))

            submitted = st.form_submit_button("Hitung Probabilitas")

        if submitted:
            # --- Pipeline Preprocessing untuk Input Tunggal (Dijalankan di background) ---
            input_data = {'age': age, 'job': job, 'marital': marital, 'education': education, 'default': default, 'balance': balance, 'housing': housing, 'loan': loan, 'contact': contact, 'day': day, 'month': month, 'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome}
            input_df = pd.DataFrame([input_data])

            # Langkah 1: Binning 'balance'
            bins = [-np.inf, -0.0001, 550, 1708, np.inf]; labels = ['Negative', 'Low', 'Medium', 'High']
            input_df['balance_status'] = pd.cut(input_df['balance'], bins=bins, labels=labels, right=True)
            input_df.drop('balance', axis=1, inplace=True)

            # Langkah 2: Transformasi Log
            features_to_log_transform = ['campaign', 'previous']
            for col in features_to_log_transform:
                if col in input_df.columns:
                    input_df[col + '_log'] = np.log1p(input_df[col])
                    input_df.drop(col, axis=1, inplace=True)

            # Langkah 3: One-Hot Encoding
            all_categorical_cols = list(categorical_options.keys())
            if 'balance_status' not in all_categorical_cols:
                all_categorical_cols.append('balance_status')
            cols_to_ohe = [col for col in all_categorical_cols if col in input_df.columns]
            input_df_ohe = pd.get_dummies(input_df, columns=cols_to_ohe, drop_first=True)

            # Langkah 4: Reindex kolom
            input_df_final = input_df_ohe.reindex(columns=X_reg_columns, fill_value=0)

            # Langkah 5: Scaling fitur numerik
            numerical_features_to_scale = ['age', 'day', 'pdays', 'campaign_log', 'previous_log']
            cols_to_scale = [col for col in numerical_features_to_scale if col in input_df_final.columns]
            input_df_final[cols_to_scale] = scaler.transform(input_df_final[cols_to_scale])

            # Langkah 6: Prediksi
            probability = log_reg_model.predict_proba(input_df_final)[0][1]

            st.markdown("---")
            st.subheader("Hasil Prediksi")
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric(label="Probabilitas Berlangganan", value=f"{probability:.2%}")
            with col_res2:
                if probability >= 0.5:
                    st.success("✔️ Prediksi: Yes (Berpotensi Berlangganan)")
                else:
                    st.error("❌ Prediksi: No (Kurang Berpotensi)")
