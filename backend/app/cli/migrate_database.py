import json

from app.db.session import initialize_database


def main() -> None:
    print(json.dumps(initialize_database(), indent=2))


if __name__ == "__main__":
    main()
