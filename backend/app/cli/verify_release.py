from __future__ import annotations

import json
import tomllib

from app.core.config import PROJECT_ROOT, get_settings
from app.db.session import engine
from app.ml.model_store import ModelNotReadyError, load_model
from app.release import APPLICATION_VERSION, validate_release_state
from app.services.reliability_service import collect_readiness


def main() -> int:
    settings = get_settings()
    readiness = collect_readiness(settings.parking_data_root, settings.model_root, engine)
    version_file = PROJECT_ROOT / "VERSION"

    try:
        artifact = load_model(settings.model_root)
        backend_package = tomllib.loads(
            (PROJECT_ROOT / "backend" / "pyproject.toml").read_text(encoding="utf-8")
        )
        frontend_package = json.loads(
            (PROJECT_ROOT / "frontend" / "package.json").read_text(encoding="utf-8")
        )
        report = validate_release_state(
            version_text=version_file.read_text(encoding="utf-8"),
            backend_version=str(backend_package["project"]["version"]),
            frontend_version=str(frontend_package["version"]),
            readiness=readiness,
            model_name=str(artifact.metadata.get("model_name", "")),
            threshold=artifact.threshold,
            metadata=artifact.metadata,
        )
    except (OSError, KeyError, TypeError, ValueError, ModelNotReadyError) as exc:
        report = {
            "ready": False,
            "version": APPLICATION_VERSION,
            "checks": [],
            "summary": {"passed": 0, "total": 7},
            "error": str(exc),
        }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
