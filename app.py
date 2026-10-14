"""신규 담보(특약) 자동 분류 도구 (Streamlit 화면). 실행: streamlit run app.py"""
import pandas as pd
import streamlit as st

from html import escape
from io import BytesIO

import history
from model import read_master, train, classify, accuracy
from rules import (LEVELS, INPUT_COLS, MASTER_COLS, missing_columns, check_master, check_same_name, pick_approved, is_new_class, loss_ratio_by_major, explain_row)

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
        master = read_master(latest)
        if missing_columns(master, MASTER_COLS):
            # 예전 구조(특약명 열 없음)로 저장된 이력은 불러오지 않는다
            st.session_state["old_history"] = latest.parent.name
        else:
            with st.spinner("최근 반영 버전을 불러오는 중입니다..."):
                models, checked = train(master)
            st.session_state.update(master=master, models=models, checked=checked,
                                    loaded_from=latest.parent.name)

# 다른 화면으로 갔다 와도 슬라이더 값이 남도록 매번 다시 저장해 둔다 (처음에는 기본값)
for key, default in [("threshold", 0.7), ("min_sim", 0.8), ("approve_auto", True)]:
    st.session_state[key] = st.session_state.get(key, default)

PAGES = ["① 기준표 업로드·학습", "② 분류·수정·승인", "③ 반영·다운로드", "④ 손해율 집계(선택)"]
st.session_state.setdefault("page", PAGES[0])


def go_to(name):
    """이전/다음 버튼을 누르면 사이드바 메뉴와 같은 값(page)을 바꿔서 화면을 이동한다."""
    st.session_state["page"] = name


def nav_buttons():
    """화면 맨 아래의 이전/다음 버튼."""
    i = PAGES.index(st.session_state["page"])
    st.divider()
    # 다음 버튼은 오른쪽 맨 끝에 붙인다
    st.markdown("<style>.st-key-nav_next {width: 100%; display: flex; justify-content: flex-end;} .st-key-nav_next .stButton {width: fit-content;}</style>", unsafe_allow_html=True)
    left, right = st.columns(2)
    if i > 0:
        left.button(f"← 이전: {PAGES[i - 1]}", on_click=go_to, args=(PAGES[i - 1],), key="nav_prev")
    if i < len(PAGES) - 1:
        right.button(f"다음: {PAGES[i + 1]} →", on_click=go_to, args=(PAGES[i + 1],), type="primary", key="nav_next")


def commit_edits():
    """수정 표에서 고친 내용을 edit_base에 합쳐 저장한다 (다른 화면에 갔다 와도 고친 내용이 남도록)."""
    base, state = st.session_state.get("edit_base"), st.session_state.get("editor")
    if base is not None and state:
        base = base.copy()
        for row, changes in state.get("edited_rows", {}).items():
            for col, value in changes.items():
                base.at[int(row), col] = value
        st.session_state["edit_base"] = base
    st.session_state.pop("editor", None)


def finish():
    """안내 문구만 보이고 화면을 멈출 때도 이전/다음 버튼은 보이게 한다."""
    nav_buttons()
    st.stop()


page = st.sidebar.radio("메뉴", PAGES, key="page")


STATUS_ORDER = ["검토 필요", "계층 불일치", "동일 특약명(코드 상이)", "자동 분류"]


def sort_by_status(df):
    """상태별로 묶어서 정렬한다. 같은 상태 안에서는 원래 순서를 유지한다."""
    rank = df["상태"].map(lambda x: STATUS_ORDER.index(x) if x in STATUS_ORDER else len(STATUS_ORDER))
    return df.assign(_순서=rank).sort_values("_순서", kind="stable").drop(columns="_순서").reset_index(drop=True)


# 칸 색 (배경색과 글자색을 함께 지정해 밝은/어두운 화면 모두에서 읽히게 한다)
KIND_STYLE = {
    "low": "background-color: #fde4e4; color: #a4161a; font-weight: 700",    # 신뢰도 낮음
    "sim": "background-color: #fff0cf; color: #8a5300; font-weight: 700",    # 유사도 낮음
    "hier": "background-color: #ece5fb; color: #5a2fa8; font-weight: 700",   # 계층 불일치
    "dup": "background-color: #dfeeff; color: #0b57b5; font-weight: 700",    # 동일 특약명
    "diff": "background-color: #ffd08a; color: #6b3a00; font-weight: 700",   # 기존 분류와 다름
    "miss": "background-color: #fde4e4; color: #a4161a; font-style: italic", # 입력 비어 있음
    "new": "background-color: #d9f2ef; color: #0b6b61; font-weight: 700",    # 신규 보장 가능성
    "model": "background-color: #eef0fc; color: #3a45a8",                    # 모델 추천값
    "ref": "background-color: #eeeeee; color: #555555",                      # 기존 기준표(참고)
}
STATUS_TEXT = {
    "자동 분류": "color: #17693a; font-weight: 700", "검토 필요": "color: #a4161a; font-weight: 700",
    "계층 불일치": "color: #5a2fa8; font-weight: 700", "동일 특약명(코드 상이)": "color: #0b57b5; font-weight: 700",
}


# 위쪽 전체 결과 표는 HTML로 그린다 (추천 열을 굵은 테두리로 감싸고 이유를 줄바꿈해서 보여주기 위해)
REASON_COLOR = {"new": "#0b6b61", "sim": "#8a5300", "low": "#a4161a", "hier": "#5a2fa8",
                "dup": "#0b57b5", "diff": "#8a5300", "miss": "#a4161a"}
RESULT_COLS = [("상태", 128, ""), ("특약코드", 62, "입력"), ("특약명", 142, "입력"), ("대분류", 60, "입력"),
               ("중분류", 78, "추천"), ("소분류", 78, "추천"), ("상세분류", 92, "입력"), ("신뢰도", 52, "계산"),
               ("유사도", 52, "계산"), ("이유", 300, ""), ("담보코드", 75, ""), ("담보명", 120, "")]
RESULT_CSS = """
<style>
.rt-wrap{max-height:640px;overflow:auto;border:1px solid rgba(128,128,128,.35);border-radius:8px}
table.rt{border-collapse:separate;border-spacing:0;font-size:13px;table-layout:fixed}
table.rt th{position:sticky;top:0;z-index:1;background:#f0f2f6;color:#555;font-weight:500;text-align:left;
  padding:6px 8px;border-bottom:1px solid rgba(128,128,128,.35);white-space:nowrap}
table.rt td{padding:6px 8px;border-bottom:1px solid rgba(128,128,128,.25);vertical-align:middle;overflow-wrap:anywhere}
table.rt td.num{text-align:right}
table.rt .tag{display:block;width:max-content;margin-bottom:2px;padding:0 6px;border-radius:4px;font-size:10.5px;background:#e3e6ec;color:#475467}
table.rt .tag.rec{background:#eef0fc;color:#3a45a8;border:1px solid #9aa4e6}
table.rt .tag.calc{background:transparent;color:#3a45a8;border:1px dashed #9aa4e6}
table.rt .pl{display:inline-block;padding:1px 9px;border-radius:999px;border:1px solid currentColor;font-size:12px;line-height:1.35;text-align:center}
table.rt ul{margin:0;padding:0;list-style:none;display:flex;flex-direction:column;gap:3px}
table.rt li{font-size:12px;line-height:1.4;padding-left:8px;border-left:3px solid rgba(128,128,128,.4)}
table.rt .bl{border-left:3px solid #000} table.rt .br{border-right:3px solid #000}
table.rt th.bt{border-top:3px solid #000} table.rt td.bb{border-bottom:3px solid #000}
</style>
"""


def html_result_table(result, threshold, min_sim, hierarchy):
    """전체 분류 결과 표(HTML). 이유가 있는 칸만 색칠하고, 추천 열(중분류·소분류)을 굵은 테두리로 감싼다."""
    names = [c for c, _, _ in RESULT_COLS]
    cg = "".join(f'<col style="width:{w}px">' for _, w, _ in RESULT_COLS)
    head = ""
    for col, _, tag in RESULT_COLS:
        edge = " bl bt" if col == "중분류" else " br bt" if col == "소분류" else ""
        label = {"대분류": "대분류", "중분류": "중분류", "소분류": "소분류"}.get(col, col)
        tag_html = f'<span class="tag {"rec" if tag == "추천" else "calc" if tag == "계산" else ""}">{tag}</span>' if tag else ""
        head += f'<th class="{edge.strip()}">{tag_html}{label}</th>'
    body = []
    n_rows = len(result)
    for n, (_, r) in enumerate(result.iterrows()):
        why = explain_row(r, threshold, min_sim, hierarchy)
        kinds = {}
        for kind, cols, _ in why:
            for col in cols:
                kinds[col] = kind
        values = {c: ("" if pd.isna(r.get(c)) else str(r.get(c))) for c in names if c not in ("상태", "이유", "신뢰도", "유사도")}
        for col in ["대분류", "상세분류"]:
            if values[col].strip() == "":
                values[col] = "(비어 있음)"
        values["신뢰도"], values["유사도"] = f'{r["신뢰도"]:.2f}', f'{r["유사도"]:.2f}'
        last = (n == n_rows - 1) and r["상태"] != "동일 특약명(코드 상이)"
        tds = f'<td><span class="pl" style="{STATUS_TEXT.get(r["상태"], "")}">{escape(r["상태"])}</span></td>'
        for col in names[1:]:
            if col == "이유":
                items = "".join(f'<li style="border-left-color:{REASON_COLOR.get(k, "#999")}">{escape(t)}</li>' for k, _, t in why)
                tds += f'<td>{"<ul>" + items + "</ul>" if items else "—"}</td>'
                continue
            kind = kinds.get(col) or ("model" if col in ("중분류", "소분류") else "")
            cls = ("num " if col in ("신뢰도", "유사도") else "") + ("bl " if col == "중분류" else "br " if col == "소분류" else "") \
                + ("bb" if last and col in ("중분류", "소분류") else "")
            tds += f'<td class="{cls.strip()}" style="{KIND_STYLE.get(kind, "")}">{escape(values[col])}</td>'
        body.append(f"<tr>{tds}</tr>")
        if r["상태"] == "동일 특약명(코드 상이)":      # 기준표의 기존 행을 바로 아래에 회색으로 끼워 넣는다
            old = {"특약코드": r["기존_특약코드"], "특약명": r["특약명"], "담보코드": r["담보코드"], "담보명": r["담보명"],
                   **{lv: r["기존_" + lv] for lv in LEVELS}}
            is_last = n == n_rows - 1
            tds = f'<td><span class="pl" style="{KIND_STYLE["ref"]}">기존 기준표(참고)</span></td>'
            for col in names[1:]:
                cls = ("bl " if col == "중분류" else "br " if col == "소분류" else "") + ("bb" if is_last and col in ("중분류", "소분류") else "")
                tds += f'<td class="{cls.strip()}" style="{KIND_STYLE["ref"]}">{escape(str(old.get(col, "")))}</td>'
            body.append(f"<tr>{tds}</tr>")
    width = sum(w for _, w, _ in RESULT_COLS)
    return (RESULT_CSS + f'<div class="rt-wrap"><table class="rt" style="width:{width}px"><colgroup>{cg}</colgroup>'
            f'<thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


# ---------- ① 기준표 업로드·학습 ----------
if page == "① 기준표 업로드·학습":
    st.header("① 기준표 업로드·학습")
    if st.session_state.get("loaded_from"):
        st.info(f"최근 반영 버전({st.session_state['loaded_from']})을 자동으로 불러와 학습했습니다. "
                "새 기준표로 바꾸려면 아래에서 업로드하세요.")
    if st.session_state.get("old_history"):
        st.warning(f"이력 폴더의 최근 버전({st.session_state['old_history']})은 예전 열 구성이라 자동으로 불러오지 않았습니다. "
                   "아래에서 새 기준표를 업로드하세요.")
    file = st.file_uploader("기준표 엑셀 (coverage_master.xlsx, 첫 번째 시트 사용)", type="xlsx")
    if file and st.session_state.get("pending_id") != file.file_id:
        master = read_master(file)
        lack = missing_columns(master, MASTER_COLS)
        if lack:
            st.error(f"기준표에 없는 열이 있습니다: {', '.join(lack)}  (필요한 열: {', '.join(MASTER_COLS)})")
            finish()
        # 화면을 옮겨도 남도록 업로드한 기준표를 보관한다
        st.session_state.update(pending_master=master, pending_name=file.name, pending_id=file.file_id)
        st.session_state.pop("train_msg", None)

    master = st.session_state.get("pending_master")
    if master is not None:
        st.caption(f"업로드한 기준표: {st.session_state['pending_name']} (화면을 옮겨도 유지됩니다)")
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
            st.session_state["train_msg"] = f"학습 완료! ({(checked['상태'] == '정상').sum()}건으로 학습했습니다)"
    if st.session_state.get("train_msg"):
        st.success(st.session_state["train_msg"])
    elif st.session_state["models"] is not None:
        st.info("이미 학습된 모델이 있습니다. 다시 학습하려면 기준표를 업로드하세요.")

# ---------- ② 신규 특약 분류 ----------
elif page == "② 분류·수정·승인":
    st.header("② 분류·수정·승인")
    if st.session_state["models"] is None:
        st.warning("먼저 ① 화면에서 기준표를 올리고 학습해 주세요.")
        finish()

    file = st.file_uploader(f"신규 특약 엑셀 ({', '.join(INPUT_COLS)})", type="xlsx")
    if file and st.session_state["new_file_id"] != file.file_id:
        new_df = pd.read_excel(file, dtype=str)
        lack = missing_columns(new_df, INPUT_COLS)
        if lack:
            st.error(f"신규 특약 파일에 없는 열이 있습니다: {', '.join(lack)}")
            finish()
        st.session_state["new_df"] = new_df
        st.session_state["new_file_id"] = file.file_id
        st.session_state["new_file_name"] = file.name
        for key in ["edit_base", "edit_sig", "edit_orig", "editor"]:   # 새 파일이면 수정 내용 초기화
            st.session_state.pop(key, None)
    if st.session_state["new_df"] is not None:
        st.caption(f"업로드한 신규 특약: {st.session_state.get('new_file_name', '')} (화면을 옮겨도 유지됩니다)")
    c1, c2 = st.columns(2)
    threshold = c1.slider("신뢰도 기준 (이 값 이상이면 자동 분류)", 0.0, 1.0, step=0.05, key="threshold")
    min_sim = c2.slider("기준표와의 유사도 기준 (이 값 미만이면 검토 필요)", 0.0, 1.0, step=0.05, key="min_sim",
                        help="기준표에 없던 표현이 많은 특약은 신뢰도가 높아도 사람이 확인하도록 합니다.")

    commit_edits()   # 지난번에 표에서 고친 내용을 먼저 저장해 둔다

    new_df = st.session_state["new_df"]
    if new_df is not None:
        result = classify(st.session_state["models"], new_df, threshold, min_sim)
        result = sort_by_status(check_same_name(result, st.session_state["checked"])).reset_index(drop=True)
        st.session_state["result"] = result

        counts = result["상태"].value_counts()
        st.write("　".join(f"**{k}** {v}건" for k, v in counts.items()))

        with st.expander("상태 안내 · 색 안내", expanded=True):
            st.markdown(
                "| 상태 | 붙는 조건 | 할 일 |\n|---|---|---|\n"
                "| **자동 분류** | 신뢰도와 유사도가 모두 기준 이상 | 대체로 그대로 승인 |\n"
                "| **검토 필요** | 신뢰도나 유사도가 기준 미만, 또는 입력의 대분류·상세분류가 비어 있음 | 추천값을 확인하고 필요하면 수정 |\n"
                "| **계층 불일치** | 추천한 중분류가 입력된 대분류의 하위가 아님 | 중분류나 대분류를 고침 |\n"
                "| **동일 특약명(코드 상이)** | 기준표에 같은 특약명이 있는데 특약코드가 다름 | 중복 등록인지 확인 (기존 분류가 이유에 표시됨) |\n")
            st.markdown(
                "- **입력**: 파일에 있던 값(특약코드·특약명·대분류·상세분류). **추천**: 중분류·소분류는 입력 파일에 없는 값이고, "
                "기준표로 학습한 모델이 계산한 추천값입니다 (연한 남색 바탕).\n"
                "- **신뢰도**는 모델이 자기 답을 얼마나 확신하는지, **유사도**는 특약명의 글자 조합이 기준표에 얼마나 있었는지입니다. "
                "낯선 표현이면 신뢰도가 높아도 검토 필요로 둡니다.\n"
                "- **신규 보장 가능성**은 입력 파일의 상세분류가 기준표에 한 번도 없을 때 붙습니다. 이유 열에는 원인에서 결과 순서로 "
                "(신규 보장 → 처음 보는 글자 조각 → 신뢰도와 2위 후보) 적습니다.\n"
                "- 이유가 여러 개여도 상태는 하나만 보입니다 (우선순위: 동일 특약명 > 입력 비어 있음 > 계층 불일치 > 검토 필요·자동 분류). "
                "이유는 해당 칸의 색과 '이유' 열에 모두 나옵니다.")
            st.markdown(
                "색: " + " ".join(f'<span style="{KIND_STYLE[k]}; padding:1px 8px; border-radius:4px">{t}</span>' for k, t in [
                    ("new", "신규 보장 가능성"), ("low", "신뢰도 낮음"), ("sim", "유사도 낮음"), ("hier", "계층 불일치"), ("dup", "동일 특약명"),
                    ("diff", "기존과 다름"), ("miss", "입력 비어 있음"), ("model", "모델 추천값")]),
                unsafe_allow_html=True)

        # ---------- 위: 전체 분류 결과 (읽기 전용) ----------
        st.subheader("전체 분류 결과 (추천값 포함, 읽기 전용)")
        hierarchy = st.session_state["models"]["_hierarchy"]
        st.html(html_result_table(result, threshold, min_sim, hierarchy))

        # ---------- 아래: 자동 분류가 아닌 행만 고쳐서 승인 ----------
        st.subheader("확인이 필요한 행 수정·승인")
        auto_mask = result["상태"] == "자동 분류"
        review = result[~auto_mask].reset_index(drop=True)
        editor_cols = (["승인", "새 분류 확인", "상태", "특약코드", "특약명"] + LEVELS
                       + ["신뢰도", "유사도", "이유", "담보코드", "담보명", "동일한 기존 특약코드", "기존 분류 비교"])

        # 자동 분류가 아닌 행이 바뀌었거나 기준 슬라이더가 바뀌면 수정 표를 새로 만든다
        sig = (tuple(review["특약코드"]), threshold, min_sim)
        if st.session_state.get("edit_sig") != sig:
            base = review.rename(columns={"기존_특약코드": "동일한 기존 특약코드"}).copy()
            base["이유"] = [" · ".join(t for _, _, t in explain_row(r, threshold, min_sim, hierarchy)) or "—"
                           for _, r in review.iterrows()]
            base["승인"], base["새 분류 확인"] = False, False
            base = base[editor_cols]
            st.session_state.update(edit_base=base, edit_orig=base.copy(), edit_sig=sig)
            st.session_state.pop("editor", None)

        def set_approval(kind):
            """승인 칸을 전체 선택 / 전체 해제한다. 고친 분류 내용은 그대로 둔다."""
            commit_edits()
            current = st.session_state["edit_base"].copy()
            current["승인"] = kind == "all"
            st.session_state["edit_base"] = current

        n_auto = int(auto_mask.sum())
        b1, b2, b3, _ = st.columns([1, 1, 2.6, 4])
        b1.button("전체 선택", on_click=set_approval, args=("all",))
        b2.button("전체 해제", on_click=set_approval, args=("none",))
        b3.checkbox(f"자동 분류 {n_auto}건도 함께 승인", key="approve_auto",
                    help="끄면 자동 분류 행은 승인되지 않아 기준표에 반영되지 않습니다.")

        if len(review) == 0:
            st.success("확인이 필요한 행이 없습니다. 모두 자동 분류입니다.")
            edited_review = st.session_state["edit_base"]
        else:
            st.caption("연필 모양이 붙은 열(승인·새 분류 확인·중분류·소분류)만 고칠 수 있습니다. "
                       "승인한 행만 기준표에 반영되고, 기준표에 없는 새 값이면 '새 분류 확인'도 체크해야 합니다. "
                       "수정 가능한 칸은 표 기능상 색이 칠해지지 않아, 읽기 전용 칸과 '이유' 열로 알립니다.")
            # 읽기 전용 칸만 색칠한다 (수정 가능한 칸은 색이 적용되지 않는다)
            kinds = pd.DataFrame("", index=range(len(review)), columns=editor_cols)
            for n, (_, r) in enumerate(review.iterrows()):
                for kind, cols, _ in explain_row(r, threshold, min_sim, hierarchy):
                    for col in cols:
                        col = "대분류" if col == "중분류" else col      # 계층 불일치는 대분류 칸으로 알린다
                        col = "이유" if kind == "diff" else col         # 기존과 다름은 이유 칸으로 알린다
                        if col in kinds.columns:
                            kinds.at[n, col] = kind

            def paint_edit(_):
                css = kinds.map(lambda k: KIND_STYLE.get(k, ""))
                css["상태"] = st.session_state["edit_base"]["상태"].map(lambda x: STATUS_TEXT.get(x, ""))
                return css

            base = st.session_state["edit_base"]
            edited_review = st.data_editor(
                base.style.apply(paint_edit, axis=None), key="editor", hide_index=True, width="stretch",
                height=min(36 * (len(base) + 1) + 3, 520),
                disabled=["상태", "특약코드", "특약명", "대분류", "상세분류", "신뢰도", "유사도", "이유", "담보코드", "담보명",
                          "동일한 기존 특약코드", "기존 분류 비교"],
                column_config={
                    "승인": st.column_config.CheckboxColumn("승인", width=75),
                    "새 분류 확인": st.column_config.CheckboxColumn(
                        "새 분류 확인", width=115, help="기준표에 없는 중분류·소분류를 일부러 추가하는 경우에만 체크"),
                    "상태": st.column_config.Column("상태", width=150),
                    "특약코드": st.column_config.Column("특약코드", width=75),
                    "특약명": st.column_config.Column("특약명", width=220),
                    "대분류": st.column_config.TextColumn("대분류 (입력)", width=100),
                    "중분류": st.column_config.TextColumn("중분류 (추천)", width=115),
                    "소분류": st.column_config.TextColumn("소분류 (추천)", width=115),
                    "상세분류": st.column_config.TextColumn("상세분류 (입력)", width=115),
                    "신뢰도": st.column_config.NumberColumn("신뢰도", format="%.2f", width=65),
                    "유사도": st.column_config.NumberColumn("유사도", format="%.2f", width=65),
                    "이유": st.column_config.Column("이유", width=360),
                    "담보코드": st.column_config.Column("담보코드", width=80),
                    "담보명": st.column_config.Column("담보명", width=140),
                })

            # 고친 내용 목록 (원래 추천값과 비교)
            orig = st.session_state["edit_orig"]
            changes = [f"{orig.at[i, '특약코드']} {col}: {orig.at[i, col]} → {edited_review.at[i, col]}"
                       for i in range(len(orig)) for col in ["중분류", "소분류"]
                       if orig.at[i, col] != edited_review.at[i, col]]
            if changes:
                st.info("**고친 내용** (위 표의 추천값과 비교)  \n" + "  \n".join(changes))

        # ---- 위·아래 표를 합쳐 반영 화면(③)에 넘길 최종 표를 만든다 ----
        final_cols = (["상태", "승인", "새 분류 확인", "특약코드", "특약명", "담보코드", "담보명"] + LEVELS
                      + ["신뢰도", "유사도", "동일한 기존 특약코드", "기존 분류 비교"])
        final = result.rename(columns={"기존_특약코드": "동일한 기존 특약코드"}).copy()
        final["승인"] = bool(st.session_state["approve_auto"]) & auto_mask
        final["새 분류 확인"] = False
        idx = final.index[~auto_mask]
        for col in ["승인", "새 분류 확인", "중분류", "소분류"]:
            final.loc[idx, col] = edited_review[col].tolist()
        final["승인"] = final["승인"].astype(bool)
        final["새 분류 확인"] = final["새 분류 확인"].astype(bool)
        final = final[final_cols]
        st.session_state["edited"] = final
        st.write(f"승인 **{int(final['승인'].sum())}건** "
                 f"(이 표 {int(final.loc[idx, '승인'].sum())}건 + 자동 분류 {int(final.loc[auto_mask, '승인'].sum())}건) "
                 f"/ 전체 {len(final)}건")

        # 새 분류인데 확인 체크가 빠진 승인 행, 동일 특약명인데 예측이 기존과 다른 행을 알려준다
        models = st.session_state["models"]
        missing = [r["특약코드"] for _, r in final[final["승인"] & ~final["새 분류 확인"]].iterrows()
                   if is_new_class(r, models["_hierarchy"], models["_minors"])]
        if missing:
            st.warning(f"기준표에 없는 새 분류입니다. 맞으면 '새 분류 확인'도 체크하세요: {', '.join(missing)}")
        diff = final[final["기존 분류 비교"] == "기존과 다름"]
        if len(diff):
            st.error(f"동일 특약명인데 예측이 기준표의 기존 분류와 다릅니다. 꼭 확인하세요: {', '.join(diff['특약코드'])}")

        with st.expander("정확도 확인 (정답지가 있을 때만)"):
            key_file = st.file_uploader("정답지 엑셀 (answer_key.xlsx)", type="xlsx", key="answer")
            if key_file:
                st.session_state["answer_df"] = pd.read_excel(key_file, dtype=str)
            if st.session_state.get("answer_df") is not None:
                per_level, by_type = accuracy(result, st.session_state["answer_df"])
                st.write("**단계별 정확도**")
                st.table(pd.Series(per_level, name="정확도").map("{:.1%}".format))
                st.write("**유형별 정확도 (중분류·소분류 모두 일치 기준)**")
                st.table(pd.Series(by_type, name="정확도").map("{:.1%}".format))

# ---------- ③ 반영·다운로드 ----------
elif page == "③ 반영·다운로드":
    st.header("③ 반영·다운로드")
    edited = st.session_state.get("edited")
    if edited is None:
        st.warning("먼저 ② 화면에서 분류를 확인하고 승인해 주세요.")
        finish()

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

# ---------- ④ 손해율 집계 (선택) ----------
else:
    st.header("④ 손해율 집계 (선택)")
    if st.session_state["master"] is None:
        st.warning("먼저 ① 화면에서 기준표를 올리고 학습해 주세요.")
        finish()

    file = st.file_uploader("담보별 실적 엑셀 (coverage_perf.xlsx)", type="xlsx")
    if file:
        st.session_state["perf_df"] = pd.read_excel(file, dtype={"기준월": str, "담보코드": str})
    if st.session_state.get("perf_df") is not None:
        perf = st.session_state["perf_df"]
        # 기준표 분류를 바탕으로, ②에서 수정한 신규 특약 분류가 있으면 그것을 우선한다
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

nav_buttons()
