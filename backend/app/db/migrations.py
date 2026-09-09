from __future__ import annotations

from sqlalchemy import Engine

SCHEMA_VERSION = 9
ANALYSIS_COLUMNS = (
    ("model_name", "VARCHAR(100)"),
    ("average_confidence", "FLOAT"),
    ("ground_truth_agreement", "FLOAT"),
    ("prediction_json", "TEXT"),
    ("source_type", "VARCHAR(20) NOT NULL DEFAULT 'scenario'"),
    ("media_asset_id", "INTEGER"),
    ("job_id", "INTEGER"),
    ("layout_id", "INTEGER"),
    ("localization_confidence", "FLOAT"),
    ("result_status", "VARCHAR(30) NOT NULL DEFAULT 'success'"),
)
VERIFIED_LAYOUT_COLUMNS = (
    ("structure_json", "TEXT"),
    ("lifecycle_state", "VARCHAR(20) NOT NULL DEFAULT 'active'"),
    ("frames_observed", "INTEGER NOT NULL DEFAULT 0"),
    ("consensus", "FLOAT NOT NULL DEFAULT 0"),
    ("revalidated_at", "DATETIME"),
)
LAYOUT_COLUMNS = (
    ("verified_layout_id", "INTEGER"),
    ("verification_state", "VARCHAR(30) NOT NULL DEFAULT 'auto_detected'"),
    ("camera_fingerprint", "VARCHAR(64)"),
    ("preview_image_path", "VARCHAR(255)"),
)


def migrate_sqlite(engine: Engine) -> dict[str, object]:
    """Apply idempotent, additive migrations without replacing existing rows."""
    if engine.dialect.name != "sqlite":
        return {"schema_version": None, "applied_columns": [], "database": engine.dialect.name}

    applied: list[str] = []
    with engine.begin() as connection:
        columns = {
            str(row[1]) for row in connection.exec_driver_sql("PRAGMA table_info(analysis_records)")
        }
        if columns:
            for column_name, column_type in ANALYSIS_COLUMNS:
                if column_name not in columns:
                    # Both identifiers come from the static allow-list above.
                    connection.exec_driver_sql(
                        f"ALTER TABLE analysis_records ADD COLUMN {column_name} {column_type}"
                    )
                    applied.append(column_name)
        layout_columns = {
            str(row[1]) for row in connection.exec_driver_sql("PRAGMA table_info(detected_layouts)")
        }
        if layout_columns:
            for column_name, column_type in LAYOUT_COLUMNS:
                if column_name not in layout_columns:
                    # Both identifiers come from the static allow-list above.
                    connection.exec_driver_sql(
                        f"ALTER TABLE detected_layouts ADD COLUMN {column_name} {column_type}"
                    )
                    applied.append(f"detected_layouts.{column_name}")
        verified_columns = {
            str(row[1])
            for row in connection.exec_driver_sql("PRAGMA table_info(verified_layouts)")
        }
        if verified_columns:
            for column_name, column_type in VERIFIED_LAYOUT_COLUMNS:
                if column_name not in verified_columns:
                    # Both identifiers come from the static allow-list above.
                    connection.exec_driver_sql(
                        f"ALTER TABLE verified_layouts ADD COLUMN {column_name} {column_type}"
                    )
                    applied.append(f"verified_layouts.{column_name}")
        if columns:
            connection.exec_driver_sql(f"PRAGMA user_version = {SCHEMA_VERSION}")
    return {
        "schema_version": SCHEMA_VERSION,
        "applied_columns": applied,
        "database": "sqlite",
    }
