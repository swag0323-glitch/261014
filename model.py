"""학습과 분류 (로컬 scikit-learn만 사용)."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from rules import LEVELS, clean_name, check_master, build_hierarchy, is_child


def read_master(path):
    """기준표 엑셀의 첫 번째 시트만 읽는다 (모두 문자열)."""
    return pd.read_excel(path, sheet_name=0, dtype=str)


def train(master):
    """검사를 통과한 행으로 4단계 분류기를 각각 학습한다.
    반환: (모델 딕셔너리, 검사 결과 표)"""
    checked = check_master(master)
    ok = checked[checked["상태"] == "정상"]
    x = ok["담보명"].map(clean_name)
    models = {}
    for level in LEVELS:
        pipe = make_pipeline(
            TfidfVectorizer(analyzer="char", ngram_range=(1, 3)),
            LogisticRegression(max_iter=1000, C=10),
        )
        pipe.fit(x, ok[level])
        models[level] = pipe
    models["_hierarchy"] = build_hierarchy(ok)
    return models, checked


def classify(models, new_df, threshold=0.7):
    """신규 담보를 분류한다. 신뢰도 = 상세분류 예측 확률의 최댓값."""
    result = new_df[["담보코드", "담보명"]].copy()
    x = result["담보명"].map(clean_name)
    for level in LEVELS:
        result[level] = models[level].predict(x)
    result["신뢰도"] = models["상세분류"].predict_proba(x).max(axis=1).round(3)
    result["상태"] = result["신뢰도"].map(lambda p: "자동 분류" if p >= threshold else "검토 필요")
    # 예측한 중분류가 예측한 대분류의 하위가 아니면 계층 불일치
    bad = ~result.apply(lambda r: is_child(models["_hierarchy"], r["대분류"], r["중분류"]), axis=1)
    result.loc[bad, "상태"] = "계층 불일치"
    return result


def accuracy(result, answer):
    """정답지와 비교해 4단계별 / 유형별 정확도를 계산한다."""
    merged = result.merge(answer, on="담보코드", suffixes=("", "_정답지"))
    per_level = {lv: (merged[lv] == merged["정답_" + lv]).mean() for lv in LEVELS}
    merged["상세일치"] = merged["상세분류"] == merged["정답_상세분류"]
    by_type = merged.groupby("유형")["상세일치"].mean().to_dict()
    return per_level, by_type
