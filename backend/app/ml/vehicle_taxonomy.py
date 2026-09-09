"""The three vehicle classes the product exposes, and how sources map into them.

The product taxonomy is deliberately small: a parking system needs to know that
a bay is filled by something car-sized, something two-wheeled, or something
lorry-sized.  Finer distinctions -- hatchback against sedan, scooter against
motorcycle -- are noise a detector cannot make reliably from a CCTV mast and
that no parking decision depends on.

Detectors are trained on wider taxonomies, so every source label is folded into
one of the three product classes here, at one place, and anything that is not a
supported vehicle stays unmapped rather than becoming a fourth class.

The taxonomy is a closed set, and membership is decided by what a label *means*,
never by how large the object happens to look.  Two labels are deliberately
excluded:

``bus``      A coach is not a lorry, and folding it into ``TRUCK`` would put an
             unsupported vehicle in front of the user under a supported name.
             It was measured to be actively harmful as well as wrong: on a
             steep top-down PKLot view the detector labelled 31 ordinary cars
             ``bus``, which surfaced as 31 trucks in a car park that had none.
``bicycle``  A pedal cycle is not a motorcycle, and a parking product has no
             use for it.

Excluding a label does not make the object disappear from the system's
reasoning.  Occupancy is decided by the crop classifier and the fusion policy,
not by the vehicle taxonomy, so a bay filled by a coach, a skip or anything else
outside these three classes can still be reported ``OCCUPIED`` or ``UNCERTAIN``.
What it cannot do is acquire a vehicle class it does not have.
"""

from __future__ import annotations

from typing import Final

CAR: Final = "CAR"
TWO_WHEELER: Final = "TWO_WHEELER"
TRUCK: Final = "TRUCK"

#: The complete set of user-facing vehicle classes.  Nothing else may reach the
#: interface, the analytics or the stored records.
PRODUCT_CLASSES: Final[tuple[str, ...]] = (CAR, TWO_WHEELER, TRUCK)

#: Display names, so the interface never has to invent its own wording.
DISPLAY_NAMES: Final[dict[str, str]] = {
    CAR: "Car",
    TWO_WHEELER: "Two-wheeler",
    TRUCK: "Truck",
}

#: COCO identifier -> product class.  Every other COCO class is unsupported,
#: including ``bus`` and ``bicycle``.
#:
#: ``car`` absorbs SUVs and vans because COCO has no separate label for them and
#: the product does not want one.
COCO_CLASS_MAP: Final[dict[str, str]] = {
    "car": CAR,
    "motorcycle": TWO_WHEELER,
    "truck": TRUCK,
}

#: Source labels seen in vehicle datasets other than COCO, folded the same way.
#: Kept explicit so a future detector can be adopted without re-deriving the
#: policy, and so the mapping can be asserted in tests.
EXTENDED_CLASS_MAP: Final[dict[str, str]] = {
    **COCO_CLASS_MAP,
    "suv": CAR,
    "van": CAR,
    "minivan": CAR,
    "hatchback": CAR,
    "sedan": CAR,
    "pickup": TRUCK,
    "lorry": TRUCK,
    "scooter": TWO_WHEELER,
    "motorbike": TWO_WHEELER,
    "moped": TWO_WHEELER,
}

#: Labels that name a real vehicle the product deliberately does not expose.
#: Held separately from "everything else in COCO" so the exclusion is a stated
#: policy that tests can assert, rather than an accident of omission.
UNSUPPORTED_VEHICLE_LABELS: Final[frozenset[str]] = frozenset(
    {"bus", "bicycle", "train", "boat", "airplane"}
)


def product_class(source_label: str) -> str | None:
    """Fold a detector's own label into a product class.

    Returns ``None`` for anything outside the three supported classes, which is
    how a person, a shopping trolley or a bollard stays out of the taxonomy
    instead of becoming a fourth vehicle type.
    """
    return EXTENDED_CLASS_MAP.get(source_label.strip().lower())


def supported_source_labels() -> frozenset[str]:
    """Every source label that maps to a product class."""
    return frozenset(EXTENDED_CLASS_MAP)


def display_name(product_label: str) -> str:
    """Interface wording for a product class."""
    return DISPLAY_NAMES.get(product_label, product_label)


def empty_counts() -> dict[str, int]:
    """A zeroed count per product class, so callers never invent key names."""
    return dict.fromkeys(PRODUCT_CLASSES, 0)
