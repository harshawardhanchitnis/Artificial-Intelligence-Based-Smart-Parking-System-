"""Ground-truth label parsing shared by every dataset preparation path.

A leaf module by design: it imports nothing from the application so that both
``preparation`` and ``integrity`` can depend on it without a cycle.
"""

from __future__ import annotations


def parse_occupancy_attribute(value: str | None) -> bool | None:
    """Interpret a PKLot ``occupied`` attribute.

    ``None`` means *unlabelled*, which is not the same as *vacant*.  Several
    PUCPR dates in the upstream archive ship ``<space>`` elements with no
    ``occupied`` attribute at all; reading a missing attribute as "0" turns
    those spaces into confident vacant labels even though most of them hold a
    car, so the distinction is preserved here and each caller decides whether to
    drop or quarantine the affected source.
    """
    if value is None:
        return None
    text = value.strip()
    if text == "1":
        return True
    if text == "0":
        return False
    return None
