from datetime import datetime

def days_since(date_str: str | None) -> int:
    """
    마지막 착용일로부터 경과 일수
    """
    if not date_str:
        return 999  # 한 번도 안 입은 옷

    try:
        last = datetime.strptime(date_str, "%Y-%m-%d")
        return (datetime.now() - last).days
    except Exception:
        return 999
