"""신규 담보 자동 분류 도구 (Streamlit 화면). 실행: streamlit run app.py"""
import pandas as pd
import streamlit as st

from model import read_master, train, classify, accuracy
from rules import LEVELS, check_master, check_same_name

st.set_page_config(page_title="신규 담보 자동 분류 도구", layout="wide")
st.title("신규 담보 자동 분류 도구")

# 화면이 새로고침되어도 값이 남도록 session_state에 보관한다
for key in ["checked", "models", "new_df"]:
    st.session_state.setdefault(key, None)

page = st.sidebar.radio(
    "메뉴", ["① 기준표 업로드·학습", "② 신규 담보 분류", "③ 수정·승인", "④ 반영·다운로드"]
)


def color_review(row):
    """검토 필요 행은 빨간색으로 표시"""
    if row["상태"] == "검토 필요":
        style = "background-color: #ffd6d6; color: #b00020"
    elif row["상태"] == "기존 기준표(참고)":
        style = "background-color: #eeeeee; color: #555555"
    else:
        style = ""
    return [style] * len(row)


# ---------- ① 기준표 업로드·학습 ----------
if page == "① 기준표 업로드·학습":
    st.header("① 기준표 업로드·학습")
    file = st.file_uploader("기준표 엑셀 (coverage_master.xlsx, 첫 번째 시트 사용)", type="xlsx")
    if file:
        master = read_master(file)
        # 학습 전에 검사만 먼저 해서 건수를 보여준다
        checked = check_master(master)
        counts = checked["상태"].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("전체", f"{len(checked)}건")
        c2.metric("누락", f"{counts.get('누락', 0)}건")
        c3.metric("계층 오류", f"{counts.get('계층 오류', 0)}건")

        bad = checked[checked["상태"] != "정상"]
        if len(bad):
            st.warning("아래 행은 학습에서 제외됩니다.")
            st.dataframe(bad, hide_index=True, width="stretch")

        if st.button("학습하기", type="primary"):
            with st.spinner("학습 중입니다..."):
                models, checked = train(master)
            st.session_state["models"] = models
            st.session_state["checked"] = checked
            st.success(f"학습 완료! ({(checked['상태'] == '정상').sum()}건으로 학습했습니다)")
    elif st.session_state["models"] is not None:
        st.info("이미 학습된 모델이 있습니다. 다시 학습하려면 기준표를 업로드하세요.")

# ---------- ② 신규 담보 분류 ----------
elif page == "② 신규 담보 분류":
    st.header("② 신규 담보 분류")
    if st.session_state["models"] is None:
        st.warning("먼저 ① 화면에서 기준표를 올리고 학습해 주세요.")
        st.stop()

    file = st.file_uploader("신규 담보 엑셀 (담보코드, 담보명)", type="xlsx")
    if file:
        st.session_state["new_df"] = pd.read_excel(file, dtype=str)
    threshold = st.slider("신뢰도 기준 (이 값 이상이면 자동 분류)", 0.0, 1.0, 0.7, 0.05)

    new_df = st.session_state["new_df"]
    if new_df is not None:
        result = classify(st.session_state["models"], new_df, threshold)
        result = check_same_name(result, st.session_state["checked"])
        st.session_state["result"] = result

        counts = result["상태"].value_counts()
        st.write("　".join(f"**{k}** {v}건" for k, v in counts.items()))

        # 동일 담보명이면 그 행 바로 아래에 기준표의 기존 행(회색)을 끼워 넣는다
        show = ["담보코드", "담보명"] + LEVELS + ["신뢰도", "상태"]
        rows = []
        for _, r in result.iterrows():
            rows.append(r[show])
            if r["상태"] == "동일 담보명(코드 상이)":
                old = {"담보코드": r["기존_담보코드"], "담보명": r["담보명"], "신뢰도": None,
                       "상태": "기존 기준표(참고)"}
                old.update({lv: r["기존_" + lv] for lv in LEVELS})
                rows.append(pd.Series(old)[show])
        table = pd.DataFrame(rows).reset_index(drop=True)
        st.dataframe(table.style.apply(color_review, axis=1),
                     hide_index=True, width="stretch")

        with st.expander("정확도 확인 (정답지가 있을 때만)"):
            key_file = st.file_uploader("정답지 엑셀 (answer_key.xlsx)", type="xlsx", key="answer")
            if key_file:
                per_level, by_type = accuracy(result, pd.read_excel(key_file, dtype=str))
                st.write("**단계별 정확도**")
                st.table(pd.Series(per_level, name="정확도").map("{:.1%}".format))
                st.write("**유형별 정확도 (상세분류 기준)**")
                st.table(pd.Series(by_type, name="정확도").map("{:.1%}".format))

# ---------- ③, ④ 는 다음 단계에서 만든다 ----------
else:
    st.info("이 화면은 4단계에서 만듭니다.")
