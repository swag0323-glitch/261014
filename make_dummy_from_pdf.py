"""상품요약서 PDF의 (선택특약) 이름으로 더미 엑셀 4종을 만든다.
특약명만 PDF에서 가져오고, 나머지 열(담보·분류·실적 등)은 아래 규칙으로 임의로 만든 가상 값이다.
사용법: python make_dummy_from_pdf.py 상품요약서.pdf 8 16 sample_data
        (먼저 한 번만: pip install pypdf)  ※ 특약명은 공개 자료라 결과 폴더를 깃에 올려도 된다.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def read_names(pdf_path, first_page, last_page):
    """PDF 해당 페이지에서 '+ 특약명' 줄에서 선택특약 이름을 순서대로, 중복 없이 뽑는다."""
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    text = "\n".join(reader.pages[i].extract_text() for i in range(first_page - 1, last_page))
    skip = ["인수특약", "지정대리청구서비스", "전환특약", "선지급"]   # 제도성특약은 제외
    names = []
    for line in text.split("\n"):
        m = re.match(r"^\s*\+\s*(.+?)\s*(\(선택특약\))?\s*$", line)   # "+ 특약명" 줄만 사용
        if m and not any(w in m.group(1).replace(" ", "") for w in skip):
            names.append(m.group(1).strip())
    return list(dict.fromkeys(names))


def core_of(name):
    """판매 형태 표기(R, (무), (갱신형), (간편심사형))를 뺀 핵심 이름."""
    return re.sub(r"\(간편심사형\)|\(갱신형\)|\(무\)|^R", "", name).replace(" ", "")


def classify(core):
    """핵심 이름 → (대분류, 중분류, 소분류, 상세분류). 연구용 가상 분류 규칙이다."""
    c = core
    has = lambda *words: all(w in c for w in words)

    if has("납입면제"):
        detail = next(d for k, d in [("일반암", "일반암"), ("뇌졸중", "뇌졸중"), ("급성심근경색", "급성심근경색"),
                                     ("뇌출혈", "뇌출혈"), ("50%장해", "50%장해")] if k in c)
        return "기타", "납입면제", "납입면제", detail
    if has("정기특약"):
        return "사망", "정기사망", "사망", "정기사망"
    if has("재해사망"):
        return "사망", "재해사망", "사망", "재해사망"
    if has("재해장해"):
        return "상해", "재해장해", "장해", next(d for d in ["3~100%", "50%이상", "80%이상"] if d in c)
    if has("재해골절"):
        return "상해", "골절", "치료", "골절"
    if has("깁스"):
        return "상해", "골절", "치료", "깁스"
    if has("재해수술"):
        return "상해", "재해수술", "수술", "재해수술"
    if has("6대질병장해"):
        return "질병", "질병장해", "장해", "6대질병50%이상"
    if has("질병장해"):
        return "질병", "질병장해", "장해", next(d for d in ["3~100%", "50%이상", "80%이상"] if d in c)

    # 암
    if "검사지원" in c or "병리검사" in c:
        for k, d in [("암자기공명", "MRI"), ("양전자", "PET"), ("갑상선바늘", "갑상선생검"),
                     ("유방바늘", "유방생검"), ("전립선바늘", "전립선생검")]:
            if k in c:
                return "질병", "암", "검사", d
        if has("혈관조영술"):
            return "질병", "순환계", "검사", "혈관조영술"
        return "질병", "순환계", "검사", "뇌심장질환검사"
    if has("다빈치"):
        return "질병", "암", "수술", "로봇수술"
    if has("산정특례"):
        mid, d = ("암", "암") if c.startswith("암") else ("뇌혈관", "뇌혈관질환") if c.startswith("뇌") else ("심장", "심장질환")
        return "질병", mid, "산정특례", d
    for k, d in [("고액암진단", "고액암"), ("소액암진단", "소액암"), ("일반암진단", "일반암"),
                 ("남녀특정암진단", "남녀특정암"), ("림프절전이암진단", "림프절전이암"),
                 ("특정전이암진단", "특정전이암"), ("3대주요암진단", "주요암")]:
        if has(k):
            return "질병", "암", "진단", d
    if c == "암수술특약":
        return "질병", "암", "수술", "암"
    if has("요양병원암입원"):
        return "질병", "암", "입원", "요양병원"
    if has("암직접치료") and has("입원"):
        return "질병", "암", "입원", "암직접치료"
    if has("암직접치료") and has("통원"):
        d = "상급종합병원" if "상급종합병원" in c else "종합병원" if "종합병원" in c else "암직접치료"
        return "질병", "암", "통원", d
    for k, d in [("세기조절", "세기조절"), ("양성자", "양성자"), ("중입자", "중입자"), ("항암방사선", "일반방사선")]:
        if has(k) and has("방사선"):
            return "질병", "암", "방사선치료", d
    for k, d in [("표적항암", "표적항암"), ("특정면역항암", "면역항암"), ("특정항암호르몬", "항암호르몬"),
                 ("항암약물치료", "항암약물"), ("갑상선암수술후호르몬", "갑상선암호르몬")]:
        if has(k):
            return "질병", "암", "약물치료", d
    if has("암주요치료(항암호르몬약물)"):
        return "질병", "암", "주요치료", "항암호르몬"
    if has("암주요치료"):
        d = "비급여암주요치료" if "비급여" in c else "상급종합병원암주요치료" if "상급종합병원" in c else "암주요치료"
        return "질병", "암", "주요치료", d

    # 뇌·심장·순환계
    if has("특정순환계질환주요치료"):
        return "질병", "순환계", "주요치료", "상급종합병원순환계" if "상급종합병원" in c else "특정순환계질환"
    if has("대동맥인조혈관"):
        return "질병", "순환계", "수술", "대동맥인조혈관치환"
    for k, mid, sub, d in [
        ("뇌졸중진단", "뇌혈관", "진단", "뇌졸중"), ("뇌출혈진단", "뇌혈관", "진단", "뇌출혈"),
        ("뇌전증진단", "뇌혈관", "진단", "뇌전증"), ("급성뇌경색증혈전", "뇌혈관", "치료", "급성뇌경색증"),
        ("뇌혈관질환진단", "뇌혈관", "진단", "뇌혈관질환"), ("뇌혈관질환수술", "뇌혈관", "수술", "뇌혈관질환"),
        ("뇌혈관질환입원", "뇌혈관", "입원", "뇌혈관질환"), ("뇌혈관질환", "뇌혈관", "통원", "뇌혈관질환"),
        ("급성심근경색증진단", "심장", "진단", "급성심근경색"), ("급성심근경색증혈전", "심장", "치료", "급성심근경색"),
        ("허혈성심장질환진단", "심장", "진단", "허혈성심장질환"), ("허혈성심장질환수술", "심장", "수술", "허혈성심장질환"),
        ("허혈성심장질환입원", "심장", "입원", "허혈성심장질환"), ("허혈성심장질환", "심장", "통원", "허혈성심장질환"),
        ("부정맥진단", "심장", "진단", "부정맥"), ("심장판막수술", "심장", "수술", "심장판막"),
        ("관상동맥", "심장", "수술", "관상동맥우회")]:
        if k in c:
            return "질병", mid, sub, d

    # 일반 질병·입원·간병
    if has("질병수술"):
        d = "상급종합병원질병수술" if "상급종합병원" in c else "종합병원질병수술" if "종합병원" in c else "질병수술"
        return "질병", "일반질병", "수술", d
    for k, sub, d in [("1~5종수술", "수술", "1~5종"), ("3대기관양성신생물", "수술", "양성신생물및용종"),
                      ("5대장기이식", "수술", "장기이식"), ("조혈모세포이식", "수술", "조혈모세포이식"),
                      ("간경변증진단", "진단", "간경변증"), ("갑상선기능항진증", "치료", "갑상선기능항진증"),
                      ("통풍진단", "진단", "통풍"), ("응급실내원", "응급실", "응급실내원")]:
        if k in c:
            return "질병", "일반질병", sub, d
    if has("첫날부터"):
        d = ("중환자실입원" if "중환자실" in c else "상급종합병원입원" if "상급종합병원" in c
             else "종합병원입원" if "종합병원" in c else "첫날입원")
        return "질병", "입원", "입원일당", d
    if has("간병"):
        sub = "간호간병통합서비스" if "통합서비스" in c else "간병인사용"
        return "질병", "간병", sub, "요양병원" if "(요양병원)" in c else "요양병원제외"
    raise ValueError(f"분류 규칙이 없는 특약명입니다: {core}")


def main(pdf, first, last, out_dir):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    names = read_names(pdf, int(first), int(last))
    info = {n: classify(core_of(n)) for n in names}   # 분류 못 하는 이름이 있으면 여기서 멈춘다
    rng = np.random.default_rng(11)

    # 신규 특약으로 뗄 이름: 신규 보장 = 핵심 이름 통째로, 표기 변형 = 같은 핵심 이름의 다른 판매 형태
    new_cores = ["통풍진단특약Ⅱ", "간경변증진단특약", "5대장기이식수술특약", "뇌전증진단특약",
                 "대동맥인조혈관치환수술특약"]
    by_core = {}
    for n in names:
        by_core.setdefault(core_of(n), []).append(n)
    exclude = {n for c in new_cores for n in by_core.get(c, [])}          # 신규 보장 종류는 기준표에서 모두 뺀다
    new_names = {by_core[c][0]: "신규 보장" for c in new_cores if c in by_core}   # 신규 특약에는 종류별 1건만 넣는다
    variant_cores = [c for c, v in by_core.items() if len(v) >= 3 and c not in new_cores]
    pick = variant_cores[::max(1, len(variant_cores) // 20)][:20]
    for c in pick:
        new_names[by_core[c][-1]] = "표기 변형"
    master_names = [n for n in names if n not in new_names and n not in exclude]
    same_names = master_names[3::max(1, len(master_names) // 5)][:5]    # 동일 특약명(코드 상이) 시험용

    # 담보는 (대분류, 중분류) 단위로 묶는다
    cov = {}
    def cov_of(major, middle):
        key = (major, middle)
        if key not in cov:
            cov[key] = (f"CV{len(cov) + 1:04d}", f"{major} {middle}보장")
        return cov[key]

    rows = []
    for i, n in enumerate(master_names, 1):
        major, middle, minor, detail = info[n]
        c, cn = cov_of(major, middle)
        rows.append([c, cn, f"TK{i:04d}", n, major, middle, minor, detail])
    cols = ["담보코드", "담보명", "특약코드", "특약명", "대분류", "중분류", "소분류", "상세분류"]
    master = pd.DataFrame(rows, columns=cols)

    # 의도한 오류 5건 (누락 3, 계층 오류 2)
    errors = [
        ("(무)암직접치료입원특약B(간편심사형)", ("질병", "암", "", "암직접치료"), "소분류 누락"),
        ("(무)뇌졸중진단특약B(간편심사형)", ("질병", "뇌혈관", "진단", ""), "상세분류 누락"),
        ("(무)재해골절치료특약B(간편심사형)", ("상해", "", "치료", "골절"), "중분류 누락"),
        ("(무)허혈성심장질환진단특약B(간편심사형)", ("상해", "심장", "진단", "허혈성심장질환"), "계층 오류(심장은 질병 하위)"),
        ("(무)질병수술특약B(간편심사형)", ("상해", "일반질병", "수술", "질병수술"), "계층 오류(일반질병은 질병 하위)"),
    ]
    guide = []
    for k, (n, (major, middle, minor, detail), why) in enumerate(errors, 1):
        c, cn = cov_of(major, middle or "미지정")
        code = f"TK{8000 + k}"
        master.loc[len(master)] = [c, cn, code, n, major, middle, minor, detail]
        guide.append([code, n, why])
    guide = pd.DataFrame(guide, columns=["특약코드", "특약명", "의도한 오류"])

    # 신규 특약 + 정답지
    new_rows, key_rows = [], []
    mcols = ["담보코드", "담보명", "특약코드", "특약명", "대분류", "상세분류"]
    items = [(n, kind) for n, kind in new_names.items()]
    items += [(n, "동일 특약명(코드 상이)") for n in same_names]
    for k, (n, kind) in enumerate(items, 1):
        major, middle, minor, detail = info[n]
        c, cn = cov_of(major, middle)
        code = f"TK{9000 + k}"
        new_rows.append([c, cn, code, n, major, detail])
        key_rows.append([c, cn, code, n, major, middle, minor, detail, kind])
    newdf = pd.DataFrame(new_rows, columns=mcols)
    key = pd.DataFrame(key_rows, columns=["담보코드", "담보명", "특약코드", "특약명", "정답_대분류", "정답_중분류", "정답_소분류", "정답_상세분류", "유형"])

    # 담보별 실적
    base = {"질병": .62, "상해": .55, "사망": .48, "기타": .30}
    perf = []
    for (major, _), (c, _) in cov.items():
        for mon in range(1, 7):
            prem = int(rng.integers(1_500_000, 3_500_000))
            perf.append([f"2026{mon:02d}", c, int(rng.integers(5, 150)), prem,
                         int(prem * np.clip(base[major] + rng.normal(0, .12), .1, 1.2))])
    perf = pd.DataFrame(perf, columns=["기준월", "담보코드", "신계약건수", "위험보험료", "보험금"])

    # 특약 단위 실적: 정상 분류 행(오류 행 제외)과 신규 특약 모두
    ok_rows = master[~master["특약코드"].isin([f"TK{8000 + k}" for k in range(1, 6)])]
    codes = list(ok_rows["특약코드"]) + list(newdf["특약코드"])
    major_of = dict(zip(ok_rows["특약코드"], ok_rows["대분류"])) | dict(zip(newdf["특약코드"], newdf["대분류"]))
    perf_tk = []
    for code in codes:
        for mon in range(1, 7):
            prem = int(rng.integers(120_000, 520_000))
            perf_tk.append([f"2026{mon:02d}", code, int(rng.integers(1, 40)), prem,
                            int(prem * np.clip(base[major_of[code]] + rng.normal(0, .22), .05, 1.6))])
    perf_tk = pd.DataFrame(perf_tk, columns=["기준월", "특약코드", "신계약건수", "위험보험료", "보험금"])
    perf_tk.to_excel(out / "coverage_perf_special.xlsx", index=False, sheet_name="실적(특약)")

    with pd.ExcelWriter(out / "coverage_master.xlsx") as w:
        master.to_excel(w, sheet_name="기준표", index=False)
        guide.to_excel(w, sheet_name="더미안내", index=False)
    newdf.to_excel(out / "new_coverage.xlsx", index=False, sheet_name="신규특약")
    key.to_excel(out / "answer_key.xlsx", index=False, sheet_name="정답지")
    perf.to_excel(out / "coverage_perf.xlsx", index=False, sheet_name="실적")
    print(f"특약명 {len(names)}개 → 기준표 {len(master)}건, 신규 {len(newdf)}건, 담보 {len(cov)}개, 실적 {len(perf)}건")
    print(key["유형"].value_counts().to_string())


if __name__ == "__main__":
    main(*sys.argv[1:5])
