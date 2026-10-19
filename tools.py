from langchain_core.tools import tool


@tool
def estimate_furniture_delivery(
    distance_km: float,
    item_count: int,
    floor: int,
    lift_available: bool,
    assembly_required: bool,
) -> dict:
    """Estimate furniture delivery, carrying, and optional assembly charges.

    Standard delivery is available up to 30 km. Base delivery costs INR 500.
    Distances above 10 km add INR 30 per extra kilometre. Without a lift,
    carrying costs INR 100 per item for each floor above ground floor.
    Optional assembly costs INR 200 per item. This tool does not schedule
    delivery or assembly.
    """
    if isinstance(distance_km, bool) or not isinstance(distance_km, (int, float)) or distance_km < 0:
        raise ValueError("Distance must be a non-negative number.")

    if isinstance(item_count, bool) or not isinstance(item_count, int) or item_count <= 0:
        raise ValueError("Item count must be a positive integer.")

    if isinstance(floor, bool) or not isinstance(floor, int) or floor < 0:
        raise ValueError("Floor must be a non-negative integer.")

    if not isinstance(lift_available, bool):
        raise ValueError("Lift availability must be True or False.")

    if not isinstance(assembly_required, bool):
        raise ValueError("Assembly selection must be True or False.")

    if distance_km > 30:
        return {
            "status": "MANUAL_QUOTE_REQUIRED",
            "base_charge": None,
            "distance_charge": None,
            "carrying_charge": None,
            "assembly_charge": None,
            "total": None,
        }

    base_charge = 500.0
    distance_charge = max(0.0, float(distance_km) - 10.0) * 30.0
    carrying_charge = 0.0 if lift_available else floor * item_count * 100.0
    assembly_charge = item_count * 200.0 if assembly_required else 0.0
    total = base_charge + distance_charge + carrying_charge + assembly_charge

    return {
        "status": "ESTIMATE_AVAILABLE",
        "base_charge": round(base_charge, 2),
        "distance_charge": round(distance_charge, 2),
        "carrying_charge": round(carrying_charge, 2),
        "assembly_charge": round(assembly_charge, 2),
        "total": round(total, 2),
    }
