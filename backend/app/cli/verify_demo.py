import json

from app.api.routes.demo import readiness


def main() -> None:
    report = readiness()
    print(json.dumps(report, indent=2))
    if not report["ready"]:
        raise SystemExit("Offline presentation preflight failed")


if __name__ == "__main__":
    main()
