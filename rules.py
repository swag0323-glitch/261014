"""검사 규칙과 전처리 (누락, 계층 오류 등)."""
import re
import pandas as pd

LEVELS = ["대분류", "중분류", "소분류", "상세분류"]
PREDICT_LEVELS = ["중분류", "소분류"]   # 모델이 예측하는 단계 (대분류·상세분류는 입력에 이미 있다)
INPUT_COLS = ["담보코드", "담보명", "특약코드", "특약명", "대분류", "상세분류"]   # 신규 특약 파일의 열
MASTER_COLS = ["담보코드", "담보명", "특약코드", "특약명", "대분류", "중분류", "소분류", "상세분류"]   # 기준표의 열


def missing_columns(df, cols):
    """엑셀에 꼭 있어야 하는 열 중 빠진 것의 목록 (화면에서 안내용)."""
    return [c for c in cols if c not in df.columns]


def clean_name(name):
    """특약명 전처리: 괄호 문구, 맨 앞 R, 공백, Ⅱ/Ⅲ, 플러스, 특약, N종 제거 + 진단금→진단비."""
    s = str(name)
    s = re.sub(r"\([^)]*\)|（[^）]*）", "", s)   # 괄호 안 문구 제거
    s = re.sub(r"\d+종", "", s)                  # N종 제거
    s = re.sub(r"^R", "", s)                     # 판매 형태 표기 'R(무)'의 앞머리 R 제거
    for word in ["Ⅱ", "Ⅲ", "플러스", "특약"]:
        s = s.replace(word, "")
    s = re.sub(r"\s+", "", s)                    # 공백 제거
    return s.replace("진단금", "진단비")


def bigrams(text):
    """글자 두 개씩 묶은 조각 (유사도 계산용)."""
    return {text[i:i + 2] for i in range(len(text) - 1)}


def name_similarity(name, known):
    """신규 특약명의 글자 조각 중 기준표에도 있던 비율 (0~1). 낮을수록 기준표에 없던 새 표현이다."""
    pieces = bigrams(clean_name(name))
    return round(len(pieces & known) / len(pieces), 3) if pieces else 0.0


def unknown_pieces(name, known):
    """신규 특약명의 글자 조각(두 글자씩) 중 기준표에 없던 것을 순서대로 돌려준다."""
    text = clean_name(name)
    pieces = [text[i:i + 2] for i in range(len(text) - 1)]
    return [p for p in dict.fromkeys(pieces) if p not in known], len(set(pieces))


def is_new_class(row, hierarchy, minors):
    """중분류나 소분류가 기준표에 없던 값이면 '새 분류'."""
    return row["중분류"] not in hierarchy or row["소분류"] not in minors


def build_hierarchy(master):
    """기준표에서 '중분류 → 대분류' 관계를 만든다.
    같은 중분류가 가장 많이 속한 대분류를 정상 부모로 본다 (소수 오입력은 오류로 간주)."""
    ok = master.dropna(subset=["대분류", "중분류"])
    ok = ok[(ok["대분류"].str.strip() != "") & (ok["중분류"].str.strip() != "")]
    counts = ok.groupby(["중분류", "대분류"]).size().reset_index(name="n")
    best = counts.sort_values("n", ascending=False).drop_duplicates("중분류")
    return dict(zip(best["중분류"], best["대분류"]))


def is_child(hierarchy, major, middle):
    """중분류가 해당 대분류의 하위인지 확인."""
    return hierarchy.get(middle) == major


def check_master(master):
    """기준표 검사: 각 행에 상태 컬럼을 붙여 돌려준다 (정상 / 누락 / 계층 오류)."""
    df = master.copy()
    for col in LEVELS:
        df[col] = df[col].fillna("").astype(str).str.strip()
    hierarchy = build_hierarchy(df)

    def status(row):
        if any(row[c] == "" for c in LEVELS):
            return "누락"
        if not is_child(hierarchy, row["대분류"], row["중분류"]):
            return "계층 오류"
        return "정상"

    df["상태"] = df.apply(status, axis=1)
    return df


def check_same_name(result, checked_master):
    """신규 특약 중 특약명이 기준표에 이미 있는데 특약코드가 다르면 '동일 특약명(코드 상이)'으로 표시.
    기준표의 기존 분류를 '기존_대분류' 등 컬럼으로 함께 붙이고, 중·소분류 예측이 기존과 같은지 비교한다."""
    out = result.copy()
    # 정상 행을 먼저 두어, 같은 이름이 여러 개면 정상 행의 분류를 보여준다
    base = checked_master.assign(_ord=(checked_master["상태"] != "정상"))
    base = base.sort_values("_ord").drop_duplicates("특약명")
    base = base.set_index(base["특약명"].str.strip())

    for level in LEVELS:
        out["기존_" + level] = ""
    out["기존_특약코드"] = ""
    out["기존 분류 비교"] = ""
    for i, row in out.iterrows():
        name = str(row["특약명"]).strip()
        if name in base.index and base.loc[name, "특약코드"] != row["특약코드"]:
            out.at[i, "상태"] = "동일 특약명(코드 상이)"
            out.at[i, "기존_특약코드"] = base.loc[name, "특약코드"]
            for level in LEVELS:
                out.at[i, "기존_" + level] = base.loc[name, level]
            same = all(row[lv] == base.loc[name, lv] for lv in PREDICT_LEVELS)
            out.at[i, "기존 분류 비교"] = "같음" if same else "기존과 다름"
    return out


def explain_row(r, threshold, min_sim, hierarchy):
    """신규 특약 한 행의 '이유'를 돌려준다: [(종류, [칸 이름들], 설명 글자), ...]
    종류: miss(입력 비어 있음) / hier(계층 불일치) / dup(동일 특약명) / diff(기존과 다름) / low(신뢰도) / sim(유사도)"""
    out = []
    for col in ["대분류", "상세분류"]:
        if str(r[col]).strip() == "":
            out.append(("miss", [col], f"{col}가 비어 있음 (입력 확인)"))
    if str(r["대분류"]).strip() and not is_child(hierarchy, r["대분류"], r["중분류"]):
        out.append(("hier", ["대분류", "중분류"], f'{r["중분류"]}은(는) {r["대분류"]}의 하위가 아님'))
    if r["상태"] == "동일 특약명(코드 상이)":
        out.append(("dup", ["특약코드", "특약명"],
                    f'기준표 {r["기존_특약코드"]}와 특약명 동일 ({r["기존_중분류"]}/{r["기존_소분류"]}/{r["기존_상세분류"]})'))
        if r["기존 분류 비교"] == "기존과 다름":
            cols = [lv for lv in PREDICT_LEVELS if r[lv] != r["기존_" + lv]]
            out.append(("diff", cols, "예측이 기존 분류와 다름"))
    # 신규 보장 가능성 → 처음 보는 표현(유사도) → 신뢰도 순서로, 원인에서 결과 쪽으로 적는다
    if r.get("상세분류_신규"):
        out.append(("new", ["상세분류"], f"신규 보장 가능성: 기준표에 없는 상세분류 '{r['상세분류']}'"))
    if r["유사도"] < min_sim:
        n, lack = int(r.get("조각수", 0)), [t for t in str(r.get("낯선조각", "")).split(",") if t]
        text = f'유사도 {r["유사도"]:.2f}: 글자 조각 {n}개 중 {n - len(lack)}개만 기준표에 있음'
        if lack:
            text += f' (처음 보는 조각: {", ".join(lack)})'
        out.append(("sim", ["유사도"], text))
    if r["신뢰도"] < threshold:
        text = f'신뢰도 {r["신뢰도"]:.2f}'
        if r.get("신뢰도_단계"):
            step = r["신뢰도_단계"]
            text += f": {step} '{r[step]}' 확률 {r['신뢰도']:.0%}"
            if r.get("신뢰도_차선"):
                text += f", 2위 '{r['신뢰도_차선']}' {r['신뢰도_차선확률']:.0%}"
        out.append(("low", ["신뢰도"], text))
    return out


def pick_approved(edited, hierarchy, master_codes, minors):
    """승인된 행을 기준표에 넣을 행과 건너뛸 행(사유 포함)으로 나눈다.
    기준표에 없던 새 분류(중분류·소분류)는 '새 분류 확인'에 체크한 행만 반영한다."""
    ok, skipped = [], []
    for _, r in edited[edited["승인"]].iterrows():
        new_class = is_new_class(r, hierarchy, minors)
        if any(str(r[c]).strip() == "" for c in LEVELS):
            reason = "누락 (4단계 중 빈 칸이 있음)"
        elif new_class and not r.get("새 분류 확인", False):
            reason = "새 분류 (②에서 '새 분류 확인'을 체크해야 반영)"
        elif not new_class and not is_child(hierarchy, r["대분류"], r["중분류"]):
            reason = "계층 오류 (중분류가 대분류의 하위가 아님)"
        elif r["상태"] == "동일 특약명(코드 상이)":
            reason = "동일 특약명 (기준표에 이미 있음)"
        elif r["특약코드"] in master_codes:
            reason = "이미 기준표에 있는 특약코드"
        else:
            ok.append(r)
            continue
        skipped.append({**r.to_dict(), "건너뛴 사유": reason})
    return pd.DataFrame(ok), pd.DataFrame(skipped)


def loss_ratio_by_major(perf, class_table):
    """대분류별 손해율 집계. 손해율(%) = 보험금 합계 ÷ 위험보험료 합계 × 100 (평균의 평균이 아님).
    class_table: 담보코드, 대분류 컬럼이 있는 표. 코드가 중복되면 뒤쪽 행을 쓴다."""
    lookup = class_table.drop_duplicates("담보코드", keep="last")[["담보코드", "대분류"]]
    df = perf.merge(lookup, on="담보코드", how="left")
    df["대분류"] = df["대분류"].fillna("(분류 없음)")
    for col in ["위험보험료", "보험금"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    out = df.groupby("대분류", as_index=False)[["위험보험료", "보험금"]].sum()
    out["손해율(%)"] = (out["보험금"] / out["위험보험료"].where(out["위험보험료"] > 0) * 100).round(1)
    return out.sort_values("위험보험료", ascending=False).reset_index(drop=True)
