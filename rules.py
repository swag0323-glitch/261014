"""검사 규칙과 전처리 (누락, 계층 오류 등)."""
import re
import pandas as pd

LEVELS = ["대분류", "중분류", "소분류", "상세분류"]


def clean_name(name):
    """담보명 전처리: 괄호 문구, 공백, Ⅱ/Ⅲ, 플러스, 특약, N종 제거 + 진단금→진단비."""
    s = str(name)
    s = re.sub(r"\([^)]*\)|（[^）]*）", "", s)   # 괄호 안 문구 제거
    s = re.sub(r"\d+종", "", s)                  # N종 제거
    for word in ["Ⅱ", "Ⅲ", "플러스", "특약"]:
        s = s.replace(word, "")
    s = re.sub(r"\s+", "", s)                    # 공백 제거
    return s.replace("진단금", "진단비")


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
    """신규 담보 중 담보명이 기준표에 이미 있는데 코드가 다르면 '동일 담보명(코드 상이)'으로 표시.
    기준표의 기존 분류를 '기존_대분류' 등 컬럼으로 함께 붙인다."""
    out = result.copy()
    # 정상 행을 먼저 두어, 같은 이름이 여러 개면 정상 행의 분류를 보여준다
    base = checked_master.assign(_ord=(checked_master["상태"] != "정상"))
    base = base.sort_values("_ord").drop_duplicates("담보명")
    base = base.set_index(base["담보명"].str.strip())

    for level in LEVELS:
        out["기존_" + level] = ""
    out["기존_담보코드"] = ""
    for i, row in out.iterrows():
        name = str(row["담보명"]).strip()
        if name in base.index and base.loc[name, "담보코드"] != row["담보코드"]:
            out.at[i, "상태"] = "동일 담보명(코드 상이)"
            out.at[i, "기존_담보코드"] = base.loc[name, "담보코드"]
            for level in LEVELS:
                out.at[i, "기존_" + level] = base.loc[name, level]
    return out


def pick_approved(edited, hierarchy, master_codes):
    """승인된 행을 기준표에 넣을 행과 건너뛸 행(사유 포함)으로 나눈다."""
    ok, skipped = [], []
    for _, r in edited[edited["승인"]].iterrows():
        if any(str(r[c]).strip() == "" for c in LEVELS):
            reason = "누락 (4단계 중 빈 칸이 있음)"
        elif not is_child(hierarchy, r["대분류"], r["중분류"]):
            reason = "계층 오류 (중분류가 대분류의 하위가 아님)"
        elif r["상태"] == "동일 담보명(코드 상이)":
            reason = "동일 담보명 (기준표에 이미 있음)"
        elif r["담보코드"] in master_codes:
            reason = "이미 기준표에 있는 담보코드"
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
