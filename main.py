# main.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 0. 한글 폰트 깨짐 방지 (Streamlit Cloud 대응)
# ──────────────────────────────────────────────
def setup_korean_font():
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/opentype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/google-fonts/NanumGothic.ttf",
    ]
    chosen = None
    for path in candidates:
        if os.path.exists(path):
            chosen = path
            break
    if chosen is None:
        for f in fm.fontManager.ttflist:
            if "Nano" in f.name or "Nanum" in f.name:
                chosen = f.fname
                break
    if chosen is None:
        for f in fm.fontManager.ttflist:
            if any(kw in f.name for kw in ["Gothic", "Dotum", "Malgun", "AppleGothic"]):
                chosen = f.fname
                break
    if chosen:
        fm.fontManager.addfont(chosen)
        font_name = fm.FontProperties(fname=chosen).get_name()
        plt.rcParams["font.family"] = font_name
        plt.rcParams["axes.unicode_minus"] = False
    else:
        st.warning("한글 폰트를 찾지 못했습니다. 차트 레이블이 영어로 표시될 수 있습니다.")

setup_korean_font()

# ──────────────────────────────────────────────
# 1. 유틸: 인코딩 자동 시도 + 열 이름 정규화
# ──────────────────────────────────────────────
def read_csv_flexible(path):
    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            if df.shape[0] > 0:
                return df
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    return pd.DataFrame()

def normalize_growth_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    mapping = {
        "개체번호": ["개체번호", "개체 번호", "개체", "ID", "번호", "plant_id", "id"],
        "잎 수(장)": ["잎 수(장)", "잎 수", "잎수", "leaf_count", "leaf count"],
        "지상부 길이(mm)": ["지상부 길이(mm)", "지상부길이(mm)",
                            "지상부 길이", "지상부길이", "shoot_length", "shoot length"],
        "지하부길이(mm)": ["지하부길이(mm)", "지하부 길이(mm)",
                            "지하부길이", "지하부 길이", "root_length", "root length"],
        "지하부질량(g)": ["지하부질량(g)", "지하부 질량(g)",
                           "지하부질량", "지하부 질량", "root_mass", "root mass",
                           "지하부 무게(g)"],
        "지상부질량(g)": ["지상부질량(g)", "지상부 질량(g)",
                           "지상부질량", "지상부 질량", "shoot_mass", "shoot mass",
                           "지상부 무게(g)"],
    }
    reverse = {}
    for std_key, variants in mapping.items():
        for v in variants:
            reverse[v] = std_key
    col_map = {}
    for c in df.columns:
        if c in reverse:
            col_map[c] = reverse[c]
    return df.rename(columns=col_map), col_map

def normalize_env_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    groups = {
        "time": ["time", "날짜", "시각", "시간", "timestamp", "datetime", "date",
                 "시간열", "측정시간"],
        "temperature": ["temperature", "온도", "기온", "temp", "공기온도"],
        "humidity": ["humidity", "습도", "humi"],
        "ph": ["ph", "PH", "산도"],
        "ec": ["ec", "EC", "염도", "전기전도도"],
        "water_temp": ["water_temp", "수온", "water temperature"],
        "co2": ["co2", "CO2", "이산화탄소"],
        "category": ["categories", "카테고리", "구분", "category"],
    }
    col_map = {}
    for std_key, variants in groups.items():
        for v in variants:
            col_map[v] = std_key
    return df.rename(columns=col_map), col_map

def parse_env_time(series):
    s = series.astype(str).str.strip()
    m = s.str.match(r"^\d{8}_\d+$")
    if m.any():
        parsed = pd.to_datetime(
            s.str.extract(r"^(\d{4})(\d{2})(\d{2})_(\d+)$")
            .rename(columns={0: "Y", 1: "M", 2: "D", 3: "h"})
            .assign(h=lambda x: x["h"].astype(int)),
            format="%Y%m%d_%H",
            errors="coerce",
        )
        result = series.astype(object).copy()
        result[m] = parsed[m]
        result[~m] = pd.to_datetime(s[~m], errors="coerce")
        return result
    return pd.to_datetime(s, errors="coerce")

# ──────────────────────────────────────────────
# 2. 통계 용어 설명
# ──────────────────────────────────────────────
STAT_EXPLANATIONS = {
    "평균": "모든 값을 더해서 개수로 나눈 값. 데이터들의 '대표값'.",
    "표준편차": "데이터가 평균에서 얼마나 흩어져 있는지 나타내는 값. 클수록 들쭉날쭉.",
    "상관계수": "두 변수가 함께 움직이는 정도를 -1~1 사이 숫자로 나타낸 것. "
                "1에 가까우면 같이 증가, -1에 가까우면 반대로 움직인다.",
    "중앙값": "데이터를 크기순으로 나열했을 때 정확히 가운데 있는 값. "
              "극단값의 영향을 덜 받는다.",
    "최솟값": "데이터 중 가장 작은 값.",
    "최댓값": "데이터 중 가장 큰 값.",
}

# ──────────────────────────────────────────────
# 3. 페이지 설정
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="극지식물 생육 대시보드",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("🌱 극지식물 개체별 생육 차이 분석 대시보드")
st.caption("인천송천고등학교 극지식물 생육 데이터 분석")
st.divider()

# ──────────────────────────────────────────────
# 4. 사이드바: 파일 업로드 + 용어 설명
# ──────────────────────────────────────────────
st.sidebar.title("📂 데이터 업로드")
st.sidebar.caption("학교마다 열 이름이 다르더라도 자동으로 맞춰줍니다.")

uploaded_growth = st.sidebar.file_uploader(
    "📊 생육 결과 CSV",
    type=["csv"],
    help=("열 이름에 단위·띄어쓰기가 섞여 있어도 괜찮습니다 "
          "(예: '잎 수(장)', '지상부 길이(mm)')."),
)

uploaded_env = st.sidebar.file_uploader(
    "🌡️ 환경 데이터 CSV (있는 경우만)",
    type=["csv"],
    help=("형식 A(time, temperature, humidity, ph, ec, water_temp, co2) 또는 "
          "형식 B(categories, 온도, 습도, EC, PH) 모두 지원."),
    key="env_uploader",
)

with st.sidebar:
    st.divider()
    st.markdown("### 📖 용어 설명")
    for term, desc in STAT_EXPLANATIONS.items():
        st.markdown(f"**{term}** : {desc}")
    st.divider()
    st.caption("💡 업로드한 파일은 브라우저에만 일시 저장되며 서버로 전송되지 않습니다.")

# ──────────────────────────────────────────────
# 5. 데이터 로딩
# ──────────────────────────────────────────────
growth_df = pd.DataFrame()
env_df = pd.DataFrame()
growth_warnings = []
env_warnings = []

if uploaded_growth is not None:
    tmp_path = "/tmp/uploaded_growth.csv"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_growth.getvalue())
    growth_df = read_csv_flexible(tmp_path)
    if growth_df.empty:
        growth_warnings.append(
            "생육 CSV를 읽을 수 없습니다. 인코딩(utf-8-sig/cp949) 또는 "
            "파일 형식을 확인해주세요."
        )
    else:
        growth_df, growth_col_map = normalize_growth_columns(growth_df)
        if "개체번호" in growth_df.columns:
            growth_df["개체번호"] = growth_df["개체번호"].astype(str).str.strip()
        required_growth = [
            "개체번호", "잎 수(장)", "지상부 길이(mm)",
            "지하부길이(mm)", "지하부질량(g)", "지상부질량(g)",
        ]
        missing = [c for c in required_growth if c not in growth_df.columns]
        if missing:
            growth_warnings.append(
                f"다음 열이 없습니다: {', '.join(missing)}. "
                "열 이름이 학교와 다른 경우 파일 샘플을 확인하세요."
            )
        for c in [
            "잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)",
            "지하부질량(g)", "지상부질량(g)",
        ]:
            if c in growth_df.columns:
                growth_df[c] = pd.to_numeric(growth_df[c], errors="coerce")
        growth_df = growth_df.dropna(subset=["개체번호"])
        if growth_col_map:
            growth_warnings.append(
                "열 이름 매핑: "
                + ", ".join(f"{k} → {v}" for k, v in growth_col_map.items())
            )

if uploaded_env is not None:
    tmp_path = "/tmp/uploaded_env.csv"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_env.getvalue())
    env_df = read_csv_flexible(tmp_path)
    if env_df.empty:
        env_warnings.append("환경 CSV를 읽을 수 없습니다.")
    else:
        env_df, env_col_map = normalize_env_columns(env_df)
        if "time" in env_df.columns:
            env_df["time_dt"] = parse_env_time(env_df["time"])
        if env_col_map:
            env_warnings.append(
                "열 이름 매핑: "
                + ", ".join(f"{k} → {v}" for k, v in env_col_map.items())
            )

# ──────────────────────────────────────────────
# 6. 경고 표시
# ──────────────────────────────────────────────
if growth_warnings:
    with st.expander("⚠️ 생육 데이터 경고", expanded=True):
        for w in growth_warnings:
            st.warning(w)
if env_warnings:
    with st.expander("⚠️ 환경 데이터 경고", expanded=True):
        for w in env_warnings:
            st.warning(w)

# ──────────────────────────────────────────────
# 7. 데이터 없으면 중단
# ──────────────────────────────────────────────
if growth_df.empty and not uploaded_growth:
    st.info("왼쪽 사이드바에서 **생육 결과 CSV**를 업로드해주세요.")
    st.stop()

# ──────────────────────────────────────────────
# 8. 기본 정보 카드
# ──────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("개체 수", f"{len(growth_df)}개")
with c2:
    st.metric("열 개수", f"{len(growth_df.columns)}개")
with c3:
    st.metric("데이터 출처", "업로드 CSV")
with c4:
    st.metric("형식", "표준화된 생육 데이터")
st.divider()

# ──────────────────────────────────────────────
# 9. 원본 데이터 미리보기
# ──────────────────────────────────────────────
st.subheader("📋 원본 데이터 미리보기")
st.caption("업로드한 생육 데이터입니다. 열 이름이 자동으로 표준화되었습니다.")
if growth_df.empty:
    st.warning("표시할 데이터가 없습니다.")
else:
    st.dataframe(growth_df, use_container_width=True, height=300)
st.divider()

# ──────────────────────────────────────────────
# 10. 기초 통계 요약
# ──────────────────────────────────────────────
st.subheader("📊 기초 통계 요약")
st.caption(
    "각 생육 지표별 평균·표준편차·중앙값·최소·최대 값을 보여줍니다. "
    "통계 용어 설명은 사이드바를 참고하세요."
)
num_cols = [
    c for c in [
        "잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)",
        "지하부질량(g)", "지상부질량(g)",
    ]
    if c in growth_df.columns
]
if num_cols:
    desc_df = growth_df[num_cols].describe().T
    desc_df = desc_df[["count", "mean", "std", "50%", "min", "max"]]
    desc_df.columns = ["관측수", "평균", "표준편차", "중앙값", "최솟값", "최댓값"]
    st.dataframe(
        desc_df.style.format({
            "관측수": "{:.0f}",
            "평균": "{:.2f}",
            "표준편차": "{:.2f}",
            "중앙값": "{:.2f}",
            "최솟값": "{:.2f}",
            "최댓값": "{:.2f}",
        }),
        use_container_width=True,
    )
else:
    st.warning("수치형 열이 없어 통계를 계산할 수 없습니다.")
st.divider()

# ──────────────────────────────────────────────
# 11. 상관계수 히트맵
# ──────────────────────────────────────────────
st.subheader("🔗 생육 지표 간 상관관계")
st.caption(
    "상관계수: 두 지표가 함께 움직이는 정도를 -1 ~ 1 사이로 나타낸 값. "
    "1에 가까우면 함께 증가, -1에 가까우면 반대로 움직인다."
)
if len(num_cols) >= 2:
    corr = growth_df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    short_labels = [c.replace("(mm)", "").replace("(g)", "").replace("(장)", "")
                    for c in corr.columns]
    sns.heatmap(
        corr,
        annot=True,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        xticklabels=short_labels,
        yticklabels=short_labels,
    )
    ax.set_title("생육 지표 간 상관계수")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.warning("상관관계를 계산하려면 2개 이상의 수치형 열이 필요합니다.")
st.divider()

# ──────────────────────────────────────────────
# 12. 개체별 막대차트
# ──────────────────────────────────────────────
st.subheader("📈 개체별 생육 비교")
st.caption("각 개체의 생육 지표를 막대차트로 비교합니다.")
if num_cols:
    view_col = st.selectbox(
        "비교할 지표 선택",
        options=num_cols,
        index=0,
        help="각 개체의 값을 막대차트로 보여줍니다.",
    )
    if view_col:
        df_plot = growth_df[["개체번호", view_col]].sort_values(
            view_col, ascending=False
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(
            df_plot["개체번호"],
            df_plot[view_col],
            color=plt.cm.viridis(np.linspace(0.1, 0.9, len(df_plot))),
            edgecolor="white",
        )
        ax.set_xlabel("개체번호")
        ax.set_ylabel(view_col)
        ax.set_title(f"개체별 {view_col}")
        ax.tick_params(axis="x", rotation=90)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
st.divider()

# ──────────────────────────────────────────────
# 13. 산점도 (두 지표 관계)
# ──────────────────────────────────────────────
st.subheader("🔍 두 지표 간 관계 (산점도)")
st.caption(
    "가로축·세로축에 각각 지표를 선택하면 개체별 위치를 점으로 표시합니다. "
    "점들이 우상향으로 모여 있으면 두 지표가 함께 증가하는 양의 상관관계입니다."
)
if len(num_cols) >= 2:
    col_x, col_y = st.columns(2)
    with col_x:
        sel_x = st.selectbox("X축 (가로)", options=num_cols, index=0)
    with col_y:
        sel_y = st.selectbox(
            "Y축 (세로)",
            options=num_cols,
            index=min(1, len(num_cols) - 1),
        )
    if sel_x and sel_y:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(
            growth_df[sel_x],
            growth_df[sel_y],
            s=60,
            alpha=0.7,
            edgecolors="k",
            linewidths=0.5,
        )
        for _, row in growth_df.iterrows():
            ax.annotate(
                row["개체번호"],
                (row[sel_x], row[sel_y]),
                fontsize=7,
                alpha=0.6,
                ha="center",
                va="bottom",
            )
        ax.set_xlabel(sel_x)
        ax.set_ylabel(sel_y)
        ax.set_title(f"{sel_x} vs {sel_y}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        r = growth_df[sel_x].corr(growth_df[sel_y])
        st.metric("상관계수 (r)", f"{r:.3f}")
        if r > 0.5:
            label = "양의 상관관계 (함께 증가)"
        elif r > 0:
            label = "약한 양의 상관관계"
        elif r > -0.5:
            label = "약한 음의 상관관계"
        else:
            label = "음의 상관관계 (반대로 움직임)"
        st.caption(f"상관계수 {r:+.3f} — {label}")
else:
    st.warning("산점도를 그리려면 2개 이상의 수치형 열이 필요합니다.")
st.divider()

# ──────────────────────────────────────────────
# 14. 환경 데이터 시계열 (환경 CSV 업로드 시)
# ──────────────────────────────────────────────
if not env_df.empty:
    st.subheader("🌡️ 환경 데이터 시계열")
    st.caption("환경 센서 데이터를 시간 순서대로 선 그래프로 표시합니다.")
    env_metric_options = [
        c for c in ["temperature", "humidity", "ph", "ec",
                     "water_temp", "co2"]
        if c in env_df.columns
    ]
    env_metric = st.selectbox(
        "표시할 환경 지표",
        options=env_metric_options,
        index=0 if "temperature" in env_metric_options else None,
    )
    if env_metric and "time_dt" in env_df.columns:
        df_env_plot = env_df[["time_dt", env_metric]].dropna().sort_values(
            "time_dt"
        )
        if not df_env_plot.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(
                df_env_plot["time_dt"],
                df_env_plot[env_metric],
                marker="o",
                markersize=3,
                linewidth=1,
                alpha=0.8,
            )
            ax.set_xlabel("시간")
            ax.set_ylabel(env_metric)
            ax.set_title(f"시간별 {env_metric}")
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.metric("평균", f"{df_env_plot[env_metric].mean():.2f}")
            st.metric("표준편차", f"{df_env_plot[env_metric].std():.2f}")
        else:
            st.warning("그래프 그릴 데이터가 부족합니다.")
    else:
        st.warning("시간 열이 없거나 데이터가 부족합니다.")
    if "time_dt" in env_df.columns:
        times_sorted = env_df["time_dt"].dropna().sort_values()
        if len(times_sorted) >= 2:
            gaps = times_sorted.diff().dropna()
            big_gaps = gaps[gaps > pd.Timedelta(hours=6)]
            if not big_gaps.empty:
                st.warning(
                    f"센서 측정이 없는 구간(6시간 초과)이 {len(big_gaps)}곳 있습니다. "
                    "해당 구간은 그래프에서 빠져 있습니다."
                )
    st.divider()

# ──────────────────────────────────────────────
# 15. 생육 × 환경 병합 분석
# ──────────────────────────────────────────────
if not growth_df.empty and not env_df.empty:
    st.subheader("🌿 생육 × 환경 병합 분석")
    st.caption(
        "환경 데이터와 생육 데이터를 시간 기준으로 연결해 분석합니다."
    )
    if "time_dt" in env_df.columns and "개체번호" in growth_df.columns:
        env_avg = (
            env_df.set_index("time_dt")
            .resample("h")
            .mean(numeric_only=True)
            .reset_index()
            .dropna(
                subset=[
                    c for c in env_df.columns
                    if c not in ["time", "time_dt"]
                ]
            )
        )
        st.info(
            "환경 데이터는 시간별 평균 내어 생육 데이터와 같은 시간 축에서 비교합니다. "
            "개체 번호가 시간 정보와 매핑되지 않으면 병합 분석이 제한될 수 있습니다."
        )
        st.dataframe(env_avg.head(10), use_container_width=True)
    else:
        st.warning(
            "환경 데이터의 시간 열 또는 생육 데이터의 개체번호 열이 없어 "
            "병합 분석을 수행할 수 없습니다."
        )
    st.divider()

# ──────────────────────────────────────────────
# 16. 정제 데이터 다운로드
# ──────────────────────────────────────────────
if not growth_df.empty:
    csv_out = growth_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 정제된 생육 데이터 다운로드 (CSV)",
        data=csv_out,
        file_name="정제생육결과.csv",
        mime="text/csv",
    )
