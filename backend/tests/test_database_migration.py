from sqlalchemy import create_engine, inspect

from app.db.migrations import ANALYSIS_COLUMNS, migrate_sqlite


def test_legacy_analysis_table_is_upgraded_without_data_loss(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{(tmp_path / 'legacy.db').as_posix()}")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            """CREATE TABLE analysis_records (
            id INTEGER PRIMARY KEY,
            dataset VARCHAR(40) NOT NULL,
            scenario_id VARCHAR(160) NOT NULL,
            total_spaces INTEGER NOT NULL,
            occupied_spaces INTEGER NOT NULL,
            vacant_spaces INTEGER NOT NULL,
            processing_time_ms FLOAT NOT NULL,
            result_image_path VARCHAR(500),
            created_at DATETIME NOT NULL
            )"""
        )
        connection.exec_driver_sql(
            """INSERT INTO analysis_records VALUES
            (1, 'PKLot', 'legacy-1', 10, 6, 4, 12.5, NULL, '2026-01-01 10:00:00')"""
        )

    first = migrate_sqlite(engine)
    second = migrate_sqlite(engine)
    column_names = {column["name"] for column in inspect(engine).get_columns("analysis_records")}
    with engine.connect() as connection:
        preserved = connection.exec_driver_sql(
            "SELECT dataset, scenario_id FROM analysis_records WHERE id = 1"
        ).one()

    assert set(first["applied_columns"]) == {name for name, _ in ANALYSIS_COLUMNS}
    assert second["applied_columns"] == []
    assert {name for name, _ in ANALYSIS_COLUMNS}.issubset(column_names)
    assert preserved == ("PKLot", "legacy-1")
