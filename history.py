"""반영 이력 저장 (history 폴더에 반영할 때마다 버전 폴더를 하나씩 만든다)."""
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

HISTORY_DIR = Path(__file__).parent / "history"
NAME_FORMAT = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}(_\d+)?$")
FILES = {"master": "갱신된_기준표.xlsx", "result": "분류결과.xlsx", "summary": "요약.json"}


def save_version(master, edited, summary):
    """기준표, 분류·승인 결과, 요약을 새 버전 폴더에 저장하고 버전 이름을 돌려준다."""
    base = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    name, n = base, 1
    while (HISTORY_DIR / name).exists():      # 같은 초에 또 저장하면 번호를 붙인다
        n += 1
        name = f"{base}_{n}"
    folder = HISTORY_DIR / name
    folder.mkdir(parents=True)
    master.to_excel(folder / FILES["master"], index=False, sheet_name="기준표")
    edited.to_excel(folder / FILES["result"], index=False, sheet_name="분류결과")
    (folder / FILES["summary"]).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return name


def list_versions():
    """저장된 버전을 최신순으로 돌려준다."""
    if not HISTORY_DIR.exists():
        return []
    out = []
    for folder in sorted(HISTORY_DIR.iterdir(), reverse=True):
        if folder.is_dir() and NAME_FORMAT.match(folder.name):
            path = folder / FILES["summary"]
            summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            out.append({"name": folder.name, "date": folder.name[:10],
                        "time": f"{folder.name[11:13]}:{folder.name[13:15]}:{folder.name[15:17]}",
                        "summary": summary})
    return out


def read_file(name, kind):
    """버전의 파일 내용을 읽는다. kind는 'master' 또는 'result'."""
    if not NAME_FORMAT.match(name):
        raise ValueError("잘못된 버전 이름입니다.")
    return (HISTORY_DIR / name / FILES[kind]).read_bytes()


def delete_version(name):
    """버전 폴더를 삭제한다. 이력 폴더 밖은 절대 지우지 않는다."""
    if not NAME_FORMAT.match(name):
        raise ValueError("잘못된 버전 이름입니다.")
    folder = HISTORY_DIR / name
    if folder.is_dir():
        shutil.rmtree(folder)


def latest_master_path():
    """가장 최근 버전의 기준표 경로 (없으면 None)."""
    versions = list_versions()
    return (HISTORY_DIR / versions[0]["name"] / FILES["master"]) if versions else None
