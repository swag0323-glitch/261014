"""1단계 확인용: 화면 없이 터미널에서 학습 → 분류 → 정확도 출력."""
import sys
import pandas as pd
from model import read_master, train, classify, accuracy

# 사용법: python run_step1.py [엑셀 폴더]  (폴더를 안 쓰면 현재 폴더의 파일을 사용)
FOLDER = sys.argv[1] + "/" if len(sys.argv) > 1 else ""

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 100)

master = read_master(FOLDER + "coverage_master.xlsx")
models, checked = train(master)
print("기준표 검사 결과:\n", checked["상태"].value_counts().to_string(), "\n")

new = pd.read_excel(FOLDER + "new_coverage.xlsx", dtype=str)
result = classify(models, new)
print(result.to_string(index=False), "\n")

answer = pd.read_excel(FOLDER + "answer_key.xlsx", dtype=str)
per_level, by_type = accuracy(result, answer)
print("단계별 정확도:")
for lv, v in per_level.items():
    print(f"  {lv}: {v:.1%}")
print("유형별 정확도(중분류·소분류 모두 일치 기준):")
for t, v in by_type.items():
    print(f"  {t}: {v:.1%}")
