import streamlit as st
import pandas as pd
import os
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="RANI Monitoring",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
    }
    [data-testid="stMetric"] label { color: #333333 !important; }
    [data-testid="stMetric"] div { color: #000000 !important; }
    h1, h2, h3 { color: #1a1a1a; }
</style>
""", unsafe_allow_html=True)

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs/interactions.csv")

def load_data():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame(columns=['timestamp', 'source', 'session_id', 'pertanyaan', 'jawaban'])
    df = pd.read_csv(LOG_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

st.title("⚖️ RANI Monitoring Dashboard")
st.markdown("Monitor interaksi dan penggunaan asisten hukum digital Indonesia.")

st.sidebar.title("Kontrol")
if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

df = load_data()

if df.empty:
    st.warning("Belum ada data interaksi. Gunakan RANI untuk mulai memantau.")
else:
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Interaksi", len(df))
    with m2:
        st.metric("Sesi Unik", df['session_id'].nunique())
    with m3:
        avg_len = df['pertanyaan'].str.len().mean()
        st.metric("Rata-rata Panjang Pertanyaan", f"{avg_len:.0f} chars")
    with m4:
        today = df[df['timestamp'].dt.date == datetime.now().date()]
        st.metric("Interaksi Hari Ini", len(today))

    st.divider()

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("📈 Interaksi per Jam")
        df_time = df.set_index('timestamp').resample('H').size().reset_index(name='counts')
        fig = px.line(df_time, x='timestamp', y='counts', title='Interaksi per Jam',
                      line_shape='spline', render_mode='svg')
        fig.update_layout(template="plotly_white", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("📊 Aktivitas per Sumber")
        source_counts = df['source'].value_counts().reset_index()
        source_counts.columns = ['source', 'count']
        fig_pie = px.pie(source_counts, names='source', values='count', title='Distribusi per Sumber',
                         hole=0.4, color_discrete_sequence=['#4a90d9', '#00ff88', '#ff6b6b', '#ffd93d'])
        fig_pie.update_layout(template="plotly_white", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    st.subheader("💬 Riwayat Percakapan")
    search = st.text_input("🔍 Cari pertanyaan atau jawaban...")
    if search:
        df_display = df[df['pertanyaan'].str.contains(search, case=False, na=False) |
                        df['jawaban'].str.contains(search, case=False, na=False)]
    else:
        df_display = df

    st.dataframe(
        df_display.sort_values(by='timestamp', ascending=False).head(100),
        use_container_width=True,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Waktu"),
            "source": st.column_config.TextColumn("Sumber"),
            "session_id": st.column_config.TextColumn("Session ID"),
            "pertanyaan": st.column_config.TextColumn("Pertanyaan", width="medium"),
            "jawaban": st.column_config.TextColumn("Jawaban", width="large")
        }
    )

    st.divider()

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("🏆 Top 10 Sesi Aktif")
        top_sessions = df.groupby('session_id').size().reset_index(name='count').sort_values(by='count', ascending=False).head(10)
        top_sessions.columns = ['Session ID', 'Jumlah Interaksi']
        st.dataframe(top_sessions, use_container_width=True, hide_index=True)

    with c4:
        st.subheader("⏰ Aktivitas per Jam")
        df['hour'] = df['timestamp'].dt.hour
        hourly = df['hour'].value_counts().sort_index().reset_index()
        hourly.columns = ['Jam', 'Jumlah']
        fig_bar = px.bar(hourly, x='Jam', y='Jumlah', title='Distribusi per Jam',
                         color_discrete_sequence=['#4a90d9'])
        fig_bar.update_layout(template="plotly_white", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")
st.caption("RANI Monitoring | Asisten Hukum Digital Indonesia © 2026")