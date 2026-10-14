"""학습과 분류 (로컬 scikit-learn만 사용). 특약 한 줄 = 한 건이고, 중분류·소분류를 예측한다."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from rules import (LEVELS, PREDICT_LEVELS, INPUT_COLS, MASTER_COLS, clean_name,
                   check_master, build_hierarchy, is_child, bigrams, name_similarity, unknown_pieces)


def read_master(path):
    """기준표 엑셀의 첫 번째 시트만 읽는다 (모두 문자열)."""
    return pd.read_excel(path, sheet_name=0, dtype=str)


def make_text(df):
    """모델에 넣을 글자: 대분류 + 상세분류 + 전처리한 특약명. (대분류·상세분류는 이미 정해진 값이라 힌트로 쓴다)"""
    return (df["대분류"].fillna("") + " " + df["상세분류"].fillna("") + " "
            + df["특약명"].map(clean_name))


def train(master):
    """검사를 통과한 행으로 중분류·소분류 분류기를 각각 학습한다.
    반환: (모델 딕셔너리, 검사 결과 표)"""
    checked = check_master(master)
    ok = checked[checked["상태"] == "정상"]
    x = make_text(ok)
    models = {}
    for level in PREDICT_LEVELS:
        pipe = make_pipeline(
            TfidfVectorizer(analyzer="char", ngram_range=(1, 3)),
            LogisticRegression(max_iter=1000, C=10),
        )
        pipe.fit(x, ok[level])
        models[level] = pipe
    models["_hierarchy"] = build_hierarchy(ok)
    models["_details"] = set(ok["상세분류"])                                    # 기준표의 상세분류 목록
    models["_minors"] = set(ok["소분류"])                                       # 기준표의 소분류 목록
    models["_known"] = set().union(*[bigrams(clean_name(n)) for n in ok["특약명"]])   # 기준표 특약명의 글자 조각
    return models, checked


def classify(models, new_df, threshold=0.7, min_sim=0.8):
    """신규 특약의 중분류·소분류를 예측한다. 대분류·상세분류는 입력값을 그대로 쓴다.
    신뢰도 = 중분류·소분류 예측 확률의 최댓값 중 낮은 쪽 (두 단계 모두 확실해야 높게 나온다).
    신뢰도가 높아도 기준표에 없던 표현이 많으면(유사도가 낮으면) 검토 필요로 본다."""
    result = new_df[INPUT_COLS].copy()
    for col in ["대분류", "상세분류"]:
        result[col] = result[col].fillna("").astype(str).str.strip()
    x = make_text(result)
    probs, top = [], {}
    for level in PREDICT_LEVELS:
        proba = models[level].predict_proba(x)
        order = np.argsort(-proba, axis=1)
        classes = models[level].classes_
        rows = np.arange(len(x))
        result[level] = classes[order[:, 0]]
        probs.append(proba[rows, order[:, 0]])
        second = order[:, 1] if proba.shape[1] > 1 else order[:, 0]
        top[level] = (classes[second], proba[rows, second])
    result["신뢰도"] = pd.DataFrame(probs).min().round(3).values
    result["유사도"] = result["특약명"].map(lambda n: name_similarity(n, models["_known"]))
    # 신뢰도가 낮은 이유를 설명하기 위한 값: 어느 단계가 더 낮았는지, 2위 후보는 무엇이었는지
    low_step = np.argmin(np.array(probs), axis=0)
    result["신뢰도_단계"] = [PREDICT_LEVELS[i] for i in low_step]
    result["신뢰도_차선"] = [top[PREDICT_LEVELS[i]][0][n] for n, i in enumerate(low_step)]
    result["신뢰도_차선확률"] = [top[PREDICT_LEVELS[i]][1][n] for n, i in enumerate(low_step)]
    pieces = result["특약명"].map(lambda n: unknown_pieces(n, models["_known"]))
    result["낯선조각"] = pieces.map(lambda t: ",".join(t[0]))
    result["조각수"] = pieces.map(lambda t: t[1])
    result["상세분류_신규"] = (result["상세분류"] != "") & ~result["상세분류"].isin(models["_details"])
    auto = (result["신뢰도"] >= threshold) & (result["유사도"] >= min_sim)
    result["상태"] = auto.map({True: "자동 분류", False: "검토 필요"})
    # 예측한 중분류가 입력된 대분류의 하위가 아니면 계층 불일치
    bad = ~result.apply(lambda r: is_child(models["_hierarchy"], r["대분류"], r["중분류"]), axis=1)
    result.loc[bad, "상태"] = "계층 불일치"
    # 대분류·상세분류가 비어 있는 입력은 사람이 먼저 채워야 한다
    result.loc[(result["대분류"] == "") | (result["상세분류"] == ""), "상태"] = "검토 필요"
    extra = ["신뢰도_단계", "신뢰도_차선", "신뢰도_차선확률", "낯선조각", "조각수", "상세분류_신규"]
    return result[MASTER_COLS + ["신뢰도", "유사도", "상태"] + extra]


def accuracy(result, answer):
    """정답지와 비교해 단계별 / 유형별 정확도를 계산한다. (특약코드로 맞춘다)
    정답지에 있는 단계(정답_대분류 ~ 정답_상세분류)만 계산한다. 대분류·상세분류는 입력값을 그대로 쓰는 단계다.
    반환: (단계별 정확도, 유형별 정확도, '신규 보장' 유형을 뺀 단계별 정확도)"""
    levels = [lv for lv in LEVELS if "정답_" + lv in answer.columns]
    answer = answer[["특약코드"] + ["정답_" + lv for lv in levels] + ["유형"]]
    merged = result.merge(answer, on="특약코드")

    def per(df):
        return {lv: (df[lv] == df["정답_" + lv]).mean() for lv in levels} if len(df) else {}

    merged["모두일치"] = (merged["중분류"] == merged["정답_중분류"]) & (merged["소분류"] == merged["정답_소분류"])
    by_type = merged.groupby("유형")["모두일치"].mean().to_dict()
    return per(merged), by_type, per(merged[merged["유형"] != "신규 보장"])


def low_confidence_split(result, answer, threshold=0.7):
    """'신규 보장' 유형 중 신뢰도가 기준 미만(=검토 필요로 분리된)인 건수. 반환: (신규 보장 건수, 분리된 건수)"""
    merged = result.merge(answer[["특약코드", "유형"]], on="특약코드")
    new = merged[merged["유형"] == "신규 보장"]
    return len(new), int((new["신뢰도"] < threshold).sum())
