"""신규 담보(특약) 자동 분류 도구 (Streamlit 화면). 실행: streamlit run app.py"""
import pandas as pd
import streamlit as st

from io import BytesIO

import history
from model import read_master, train, classify, accuracy
from rules import (LEVELS, INPUT_COLS, MASTER_COLS, missing_columns, check_master, check_same_name, pick_approved, is_new_class, loss_ratio_by_major)

st.set_page_config(page_title="신규 담보(특약) 자동 분류 도구", layout="wide")
st.title("신규 담보(특약) 자동 분류 도구")

# 화면이 새로고침되어도 값이 남도록 session_state에 보관한다
for key in ["master", "checked", "models", "new_df", "new_file_id"]:
    st.session_state.setdefault(key, None)

# 처음 열 때 가장 최근에 반영한 버전이 있으면 자동으로 불러와 학습한다
if st.session_state["models"] is None and not st.session_state.get("autoload_done"):
    st.session_state["autoload_done"] = True
    latest = history.latest_master_path()
    if latest is not None:
        with st.spinner("최근 반영 버전을 불러오는 중입니다..."):
            master = read_master(latest)
            models, checked = train(master)
        st.session_state.update(master=master, models=models, checked=checked,
                                loaded_from=latest.parent.name)

page = st.sidebar.radio(
    "메뉴", ["① 기준표 업로드·학습", "② 신규 특약 분류", "③ 수정·승인", "④ 반영·다운로드", "⑤ 손해율 집계(선택)"]
)


STATUS_ORDER = ["검토 필요", "계층 불일치", "동일 특약명(코드 상이)", "자동 분류"]


def sort_by_status(df):
    """상태별로 묶어서 정렬한다. 같은 상태 안에서는 원래 순서를 유지한다."""
    rank = df["상태"].map(lambda x: STATUS_ORDER.index(x) if x in STATUS_ORDER else len(STATUS_ORDER))
    return df.assign(_순서=rank).sort_values("_순서", kind="stable").drop(columns="_순서").reset_index(drop=True)


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
    if st.session_state.get("loaded_from"):
        st.info(f"최근 반영 버전({st.session_state['loaded_from']})을 자동으로 불러와 학습했습니다. "
                "새 기준표로 바꾸려면 아래에서 업로드하세요.")
    file = st.file_uploader("기준표 엑셀 (coverage_master.xlsx, 첫 번째 시트 사용)", type="xlsx")
    if file:
        master = read_master(file)
        lack = missing_columns(master, MASTER_COLS)
        if lack:
            st.error(f"기준표에 없는 열이 있습니다: {', '.join(lack)}  (필요한 열: {', '.join(MASTER_COLS)})")
            st.stop()
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
            st.session_state["master"] = master
            st.session_state["models"] = models
            st.session_state["checked"] = checked
            st.success(f"학습 완료! ({(checked['상태'] == '정상').sum()}건으로 학습했습니다)")
    elif st.session_state["models"] is not None:
        st.info("이미 학습된 모델이 있습니다. 다시 학습하려면 기준표를 업로드하세요.")

# ---------- ② 신규 특약 분류 ----------
elif page == "② 신규 특약 분류":
    st.header("② 신규 특약 분류")
    if st.session_state["models"] is None:
        st.warning("먼저 ① 화면에서 기준표를 올리고 학습해 주세요.")
        st.stop()

    file = st.file_uploader(f"신규 특약 엑셀 ({', '.join(INPUT_COLS)})", type="xlsx")
    if file and st.session_state["new_file_id"] != file.file_id:
        new_df = pd.read_excel(file, dtype=str)
        lack = missing_columns(new_df, INPUT_COLS)
        if lack:
            st.error(f"신규 특약 파일에 없는 열이 있습니다: {', '.join(lack)}")
            st.stop()
        st.session_state["new_df"] = new_df
        st.session_state["new_file_id"] = file.file_id
        st.session_state.pop("edit_base", None)  # 새 파일이면 ③의 수정 내용 초기화
        st.session_state.pop("editor", None)
    c1, c2 = st.columns(2)
    threshold = c1.slider("신뢰도 기준 (이 값 이상이면 자동 분류)", 0.0, 1.0, 0.7, 0.05)
    min_sim = c2.slider("기준표와의 유사도 기준 (이 값 미만이면 검토 필요)", 0.0, 1.0, 0.8, 0.05,
                        help="기준표에 없던 표현이 많은 특약은 신뢰도가 높아도 사람이 확인하도록 합니다.")

    new_df = st.session_state["new_df"]
    if new_df is not None:
        result = classify(st.session_state["models"], new_df, threshold, min_sim)
        result = sort_by_status(check_same_name(result, st.session_state["checked"]))
        st.session_state["result"] = result

        counts = result["상태"].value_counts()
        st.write("　".join(f"**{k}** {v}건" for k, v in counts.items()))

        # 동일 특약명이면 그 행 바로 아래에 기준표의 기존 행(회색)을 끼워 넣는다
        show = ["상태", "담보코드", "담보명", "특약코드", "특약명"] + LEVELS + ["신뢰도", "유사도"]
        rows = []
        for _, r in result.iterrows():
            rows.append(r[show])
            if r["상태"] == "동일 특약명(코드 상이)":
                old = {"담보코드": r["담보코드"], "담보명": r["담보명"], "특약코드": r["기존_특약코드"],
                       "특약명": r["특약명"], "신뢰도": None, "유사도": None,
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
                st.write("**유형별 정확도 (중분류·소분류 모두 일치 기준)**")
                st.table(pd.Series(by_type, name="정확도").map("{:.1%}".format))

# ---------- ③ 수정·승인 ----------
elif page == "③ 수정·승인":
    st.header("③ 수정·승인")
    if st.session_state.get("result") is None:
        st.warning("먼저 ② 화면에서 신규 특약을 분류해 주세요.")
        st.stop()

    # 표가 없거나, 열 구성이나 특약 순서가 ②의 결과와 달라졌으면 새로 만든다
    columns = (["상태", "승인", "새 분류 확인", "특약코드", "특약명", "담보코드", "담보명"] + LEVELS
               + ["신뢰도", "유사도", "동일한 기존 특약코드", "기존 분류 비교"])
    result = st.session_state["result"]
    old = st.session_state.get("edit_base")
    if old is None or list(old.columns) != columns or list(old["특약코드"]) != list(result["특약코드"]):
        base = result[
            ["상태", "특약코드", "특약명", "담보코드", "담보명"] + LEVELS + ["신뢰도", "유사도", "기존_특약코드", "기존 분류 비교"]
        ].rename(columns={"기존_특약코드": "동일한 기존 특약코드"})
        base.insert(1, "승인", False)
        base.insert(2, "새 분류 확인", False)
        st.session_state["edit_base"] = base
        st.session_state.pop("editor", None)

    def set_approval(kind):
        """승인 칸을 전체 선택 / 전체 해제 / 자동 분류만 선택한다. 고친 분류 내용은 그대로 둔다."""
        current = st.session_state.get("edited", st.session_state["edit_base"]).copy()
        if kind == "auto":
            current["승인"] = current["상태"] == "자동 분류"
        else:
            current["승인"] = kind == "all"
        st.session_state["edit_base"] = current
        st.session_state.pop("editor", None)  # 표를 새 값으로 다시 그리도록 초기화

    b1, b2, b3, _ = st.columns([1, 1, 1.4, 5])
    b1.button("전체 선택", on_click=set_approval, args=("all",))
    b2.button("전체 해제", on_click=set_approval, args=("none",))
    b3.button("자동 분류만 선택", on_click=set_approval, args=("auto",))

    base = st.session_state["edit_base"]
    master = st.session_state["master"]
    st.caption("중분류·소분류를 직접 고치고(대분류·상세분류는 입력값이라 고칠 수 없습니다), 확인한 행은 '승인'에 체크하세요. 승인한 행만 기준표에 반영됩니다. "
               "새 값을 직접 입력할 수 있고, 새 값이면 '새 분류 확인'도 체크해야 반영됩니다.")
    edited = st.data_editor(
        base, key="editor", hide_index=True, width="stretch",
        disabled=["특약코드", "특약명", "담보코드", "담보명", "대분류", "상세분류", "신뢰도", "유사도", "상태",
                  "동일한 기존 특약코드", "기존 분류 비교"],
        column_config={
            "승인": st.column_config.CheckboxColumn("승인"),
            "새 분류 확인": st.column_config.CheckboxColumn(
                "새 분류 확인", help="기준표에 없는 중분류·소분류를 일부러 추가하는 경우에만 체크"),
            "중분류": st.column_config.TextColumn("중분류 (새 값 입력 가능)"),
            "소분류": st.column_config.TextColumn("소분류 (새 값 입력 가능)"),
        },
    )
    st.session_state["edited"] = edited
    st.write(f"승인 **{int(edited['승인'].sum())}건** / 전체 {len(edited)}건")

    # 새 분류인데 확인 체크가 빠진 승인 행을 알려준다
    models = st.session_state["models"]
    missing = [r["특약코드"] for _, r in edited[edited["승인"] & ~edited["새 분류 확인"]].iterrows()
               if is_new_class(r, models["_hierarchy"], models["_minors"])]
    if missing:
        st.warning(f"기준표에 없는 새 분류입니다. 맞으면 '새 분류 확인'도 체크하세요: {', '.join(missing)}")
    diff = edited[(edited["기존 분류 비교"] == "기존과 다름")]
    if len(diff):
        st.error(f"동일 특약명인데 예측이 기준표의 기존 분류와 다릅니다. 꼭 확인하세요: {', '.join(diff['특약코드'])}")

# ---------- ④ 반영·다운로드 ----------
elif page == "④ 반영·다운로드":
    st.header("④ 반영·다운로드")
    edited = st.session_state.get("edited")
    if edited is None:
        st.warning("먼저 ③ 화면에서 분류를 확인하고 승인해 주세요.")
        st.stop()

    def to_excel(df, sheet):
        buf = BytesIO()
        df.to_excel(buf, index=False, sheet_name=sheet)
        return buf.getvalue()

    if st.session_state.get("apply_msg"):
        st.success(st.session_state["apply_msg"])

    models, master = st.session_state["models"], st.session_state["master"]
    ok, skipped = pick_approved(edited, models["_hierarchy"], set(master["특약코드"]), models["_minors"])
    done = skipped[skipped["건너뛴 사유"].str.startswith("이미")] if len(skipped) else skipped
    skipped = skipped.drop(done.index) if len(skipped) else skipped
    st.write(f"승인 {int(edited['승인'].sum())}건 중 **반영 가능 {len(ok)}건**, 건너뜀 {len(skipped)}건"
             + (f", 이미 반영됨 {len(done)}건" if len(done) else ""))
    if len(skipped):
        st.warning("아래 행은 반영되지 않습니다.")
        st.dataframe(skipped[["특약코드", "특약명"] + LEVELS + ["건너뛴 사유"]],
                     hide_index=True, width="stretch")

    if st.button("승인분 기준표 반영 + 재학습", type="primary", disabled=len(ok) == 0):
        added = ok[MASTER_COLS]
        new_master = pd.concat([master, added], ignore_index=True)
        with st.spinner("재학습 중입니다..."):
            models, checked = train(new_master)
        st.session_state.update(master=new_master, models=models, checked=checked)
        summary = {"반영 건수": len(added), "건너뜀 건수": len(skipped),
                   "건너뜀 사유": skipped["건너뛴 사유"].tolist() if len(skipped) else [],
                   "기준표 건수": len(new_master), "학습 건수": int((checked["상태"] == "정상").sum()),
                   "반영한 특약코드": added["특약코드"].tolist()}
        name = history.save_version(new_master, edited, summary)
        st.session_state["apply_msg"] = (
            f"{len(added)}건을 기준표에 반영하고 재학습했습니다. (기준표 {len(new_master)}건, "
            f"학습 {summary['학습 건수']}건) 이력에 '{name}' 버전으로 저장했습니다.")
        st.rerun()

    st.subheader("엑셀 다운로드 (현재 상태)")
    c1, c2 = st.columns(2)
    c1.download_button("분류 결과 (승인 여부 포함)", to_excel(edited, "분류결과"),
                       file_name="classification_result.xlsx")
    c2.download_button("갱신된 기준표", to_excel(st.session_state["master"], "기준표"),
                       file_name="coverage_master_updated.xlsx")

    # ---- 반영 이력: 반영할 때마다 버전이 하나씩 쌓이고, 날짜별로 골라 내려받을 수 있다 ----
    st.subheader("반영 이력")
    st.caption("반영할 때마다 history 폴더에 저장됩니다. 클라우드에서는 앱이 재시작되면 사라질 수 있으니, "
               "이력은 내 컴퓨터나 회사 서버에서 실행할 때 사용하세요.")
    versions = history.list_versions()
    if not versions:
        st.info("아직 반영 이력이 없습니다.")
    current_date = None
    for v in versions:
        if v["date"] != current_date:
            current_date = v["date"]
            st.markdown(f"##### {current_date}")
        sm = v["summary"]
        cols = st.columns([1.2, 3, 1.6, 1.6, 1])
        cols[0].write(v["time"])
        cols[1].write(f"반영 {sm.get('반영 건수', '?')}건 · 건너뜀 {sm.get('건너뜀 건수', '?')}건 · "
                      f"기준표 {sm.get('기준표 건수', '?')}건")
        cols[2].download_button("기준표 받기", history.read_file(v["name"], "master"),
                                file_name=f"coverage_master_{v['name']}.xlsx", key=f"m_{v['name']}")
        cols[3].download_button("결과 받기", history.read_file(v["name"], "result"),
                                file_name=f"classification_{v['name']}.xlsx", key=f"r_{v['name']}")
        if cols[4].button("삭제", key=f"d_{v['name']}"):
            st.session_state["delete_pending"] = v["name"]
        if st.session_state.get("delete_pending") == v["name"]:
            st.warning(f"{v['date']} {v['time']} 버전을 삭제하면 되돌릴 수 없습니다. 정말 삭제할까요?")
            y, n, _ = st.columns([1, 1, 6])
            if y.button("삭제 확정", key=f"y_{v['name']}"):
                history.delete_version(v["name"])
                st.session_state["delete_pending"] = None
                st.rerun()
            if n.button("취소", key=f"n_{v['name']}"):
                st.session_state["delete_pending"] = None
                st.rerun()

# ---------- ⑤ 손해율 집계 (선택) ----------
else:
    st.header("⑤ 손해율 집계 (선택)")
    if st.session_state["master"] is None:
        st.warning("먼저 ① 화면에서 기준표를 올리고 학습해 주세요.")
        st.stop()

    file = st.file_uploader("담보별 실적 엑셀 (coverage_perf.xlsx)", type="xlsx")
    if file:
        perf = pd.read_excel(file, dtype={"기준월": str, "담보코드": str})
        # 기준표 분류를 바탕으로, ③에서 수정한 신규 특약 분류가 있으면 그것을 우선한다
        table = st.session_state["master"][["담보코드", "대분류"]]
        edited = st.session_state.get("edited")
        if edited is not None:
            table = pd.concat([table, edited[["담보코드", "대분류"]]], ignore_index=True)

        months = sorted(perf["기준월"].unique())
        picked = st.multiselect("기준월", months, default=months)
        perf = perf[perf["기준월"].isin(picked)]

        out = loss_ratio_by_major(perf, table)
        total = out["보험금"].sum() / out["위험보험료"].sum() * 100 if out["위험보험료"].sum() else 0
        st.metric("전체 손해율", f"{total:.1f}%")
        st.dataframe(out.style.format({"위험보험료": "{:,.0f}", "보험금": "{:,.0f}", "손해율(%)": "{:.1f}"}),
                     hide_index=True, width="stretch")
        st.bar_chart(out.set_index("대분류")["손해율(%)"])
        st.caption("손해율 = 대분류별 보험금 합계 ÷ 위험보험료 합계 × 100 (평균의 평균이 아닙니다)")
