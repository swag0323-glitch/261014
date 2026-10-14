"""2단계 확인용: 기준표 검사(누락/계층 오류)와 신규 특약 검사(동일 특약명, 계층 불일치)."""
import sys
import pandas as pd
from model import read_master, train, classify
from rules import check_same_name

# 사용법: python run_step1.py [엑셀 폴더]  (폴더를 안 쓰면 현재 폴더의 파일을 사용)
FOLDER = sys.argv[1] + "/" if len(sys.argv) > 1 else ""

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

master = read_master(FOLDER + "coverage_master.xlsx")
models, checked = train(master)

print("[기준표 검사] 학습에서 제외되는 행")
print(checked[checked["상태"] != "정상"].to_string(index=False), "\n")

new = pd.read_excel(FOLDER + "new_coverage.xlsx", dtype=str)
result = check_same_name(classify(models, new), checked)
print("[신규 특약 상태별 건수]")
print(result["상태"].value_counts().to_string(), "\n")

print("[동일 특약명(코드 상이)] 예측 vs 기준표 기존 분류")
same = result[result["상태"] == "동일 특약명(코드 상이)"]
print(same[["특약코드", "특약명", "중분류", "소분류",
            "기존_특약코드", "기존_중분류", "기존_소분류"]].to_string(index=False))
