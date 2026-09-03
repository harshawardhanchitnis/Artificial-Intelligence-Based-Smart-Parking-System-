from __future__ import annotations

import json

from app.core.config import get_settings
from app.db.session import engine
from app.services.reliability_service import collect_readiness


def main() -> int:
    settings = get_settings()
    report = collect_readiness(settings.parking_data_root, settings.model_root, engine)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
