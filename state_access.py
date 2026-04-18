import logging
from typing import Any, Dict, Iterable, List, Optional


logger = logging.getLogger(__name__)

PLATFORM_UNIT_PREFIXES = (
    "Guide_Ship_Surface",
    "Truck_Ground",
    "Recon_UAV_FixWing",
    "Cruiser_Surface",
    "Destroyer_Surface",
    "Flagship_Surface",
    "Shipboard_Aircraft_FixWing",
    "Merchant_Ship_Surface",
)

MISSILE_PREFIXES = (
    "HighCostAttackMissile",
    "LowCostAttackMissile",
    "ShipToGround_CruiseMissile",
    "Short_Range_InterceptMissile",
    "Long_Range_InterceptMissile",
    "AIM_AirMissile",
    "AIM",
    "JDAM",
    "AttackMissile",
    "CruiseMissile",
    "InterceptMissile",
)

SCORE_KEY_PRIORITY = {
    "RED": ("redScore", "redDestroyScore", "redCost"),
    "BLUE": ("blueScore", "blueDestroyScore", "blueCost"),
}


def safe_simtime(state) -> Optional[int]:
    try:
        return int(state.simtime())
    except Exception:
        return None


def raw_state_dict(state) -> Dict[str, Any]:
    raw = getattr(state, "raw_state_dict", None)
    return raw if isinstance(raw, dict) else {}


def extract_score_dict(state) -> Dict[str, Any]:
    raw = raw_state_dict(state)
    for key in ("score", "Score"):
        value = raw.get(key)
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, (int, float)):
            return {"total": float(value)}

    try:
        if hasattr(state, "get_score") and callable(getattr(state, "get_score")):
            return {"total": float(state.get_score())}
    except Exception:
        pass

    return {}


def get_side_score(state, faction_name: str) -> float:
    score = extract_score_dict(state)
    side = str(faction_name or "").upper()
    for key in SCORE_KEY_PRIORITY.get(side, ()):
        value = score.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except Exception:
            continue

    total = score.get("total")
    if isinstance(total, (int, float)):
        return float(total)
    try:
        return float(total)
    except Exception:
        return 0.0


def infer_unit_type(identifier: str) -> Optional[str]:
    text = str(identifier or "")
    for prefix in PLATFORM_UNIT_PREFIXES + MISSILE_PREFIXES:
        if prefix in text:
            return prefix
    return None


def is_platform_unit(identifier: str) -> bool:
    return infer_unit_type(identifier) in PLATFORM_UNIT_PREFIXES


def is_missile_unit(identifier: str) -> bool:
    return infer_unit_type(identifier) in MISSILE_PREFIXES


def iter_units(state) -> List[Any]:
    try:
        units = state.find_units()
        if isinstance(units, list):
            return units
    except Exception:
        pass

    units = getattr(state, "unit_states", None)
    return list(units or [])


def iter_platform_units(state) -> List[Any]:
    return [unit for unit in iter_units(state) if is_platform_unit(getattr(unit, "id", ""))]


def iter_missile_units(state) -> List[Any]:
    return [unit for unit in iter_units(state) if is_missile_unit(getattr(unit, "id", ""))]


def friendly_unit_count(state) -> int:
    return len(iter_platform_units(state))


def get_vehicle_state(unit) -> Optional[Any]:
    return getattr(unit, "vehicle_state", None) or getattr(unit, "vechicle_state", None)


def get_unit_type_name(unit) -> Optional[str]:
    try:
        unit_type = unit.unit_type_name()
        if unit_type:
            return str(unit_type)
    except Exception:
        pass
    return infer_unit_type(getattr(unit, "id", ""))


def get_unit_position(state=None, unit_id: Optional[str] = None, unit=None):
    if state is not None and unit_id:
        try:
            pos = state.get_unit_position(unit_id)
            if callable(pos):
                pos = pos()
            if pos is not None:
                return pos
        except Exception:
            pass

    if unit is not None:
        try:
            pos = unit.position()
            if callable(pos):
                pos = pos()
            return pos
        except Exception:
            pass

    return None


def serialize_position(pos) -> Optional[Dict[str, float]]:
    if pos is None:
        return None
    lat = getattr(pos, "lat", None)
    lon = getattr(pos, "lon", None)
    alt = getattr(pos, "alt", None)
    if lat is None or lon is None:
        return None
    try:
        payload = {"lat": float(lat), "lon": float(lon)}
        if alt is not None:
            payload["alt"] = float(alt)
        return payload
    except Exception:
        return None


def get_target_position(state, target_id: str):
    if not target_id:
        return None
    try:
        pos = state.get_target_position(target_id)
        if callable(pos):
            pos = pos()
        return pos
    except Exception:
        return None


def get_unit_hp(state=None, unit_id: Optional[str] = None, unit=None) -> Optional[float]:
    unit_state = None
    if state is not None and unit_id:
        try:
            unit_state = state.get_unit_state(unit_id)
        except Exception:
            unit_state = None

    if unit_state is not None:
        for attr in ("hp", "HP"):
            value = getattr(unit_state, attr, None)
            if isinstance(value, (int, float)):
                return float(value)
        vehicle_state = get_vehicle_state(unit_state)
        if vehicle_state is not None:
            for attr in ("hp", "HP"):
                value = getattr(vehicle_state, attr, None)
                if isinstance(value, (int, float)):
                    return float(value)

    if unit is not None:
        for attr in ("hp", "HP"):
            value = getattr(unit, attr, None)
            if isinstance(value, (int, float)):
                return float(value)
        vehicle_state = get_vehicle_state(unit)
        if vehicle_state is not None:
            for attr in ("hp", "HP"):
                value = getattr(vehicle_state, attr, None)
                if isinstance(value, (int, float)):
                    return float(value)

    return None


def get_unit_velocity(unit) -> Optional[float]:
    vehicle_state = get_vehicle_state(unit)
    value = getattr(vehicle_state, "vel", None) if vehicle_state is not None else None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def is_unit_active(unit) -> Optional[bool]:
    vehicle_state = get_vehicle_state(unit)
    value = getattr(vehicle_state, "is_active", None) if vehicle_state is not None else None
    if isinstance(value, bool):
        return value
    return None


def is_unit_hidden(unit) -> Optional[bool]:
    vehicle_state = get_vehicle_state(unit)
    value = getattr(vehicle_state, "is_hide_on", None) if vehicle_state is not None else None
    if isinstance(value, bool):
        return value
    return None


def get_weapon_inventory(unit) -> Dict[str, int]:
    inventory: Dict[str, int] = {}
    for weapon in getattr(unit, "weapon_states", []) or []:
        name = getattr(weapon, "name", "")
        if hasattr(name, "name"):
            name = name.name
        name = str(name or "")
        try:
            count = int(getattr(weapon, "num", 0) or 0)
        except Exception:
            count = 0
        inventory[name] = inventory.get(name, 0) + count
    return inventory


def get_detected_target_ids(state, attackable_only: bool = False) -> List[str]:
    try:
        model = "filter" if attackable_only else "normal"
        targets = state.find_all_detect_targets(model=model) or []
    except TypeError:
        try:
            targets = state.find_all_detect_targets() or []
        except Exception:
            targets = []
    except Exception:
        targets = []
    seen = []
    seen_set = set()
    for target_id in targets:
        text = str(target_id or "")
        if text and text not in seen_set:
            seen.append(text)
            seen_set.add(text)
    return seen


def get_detector_states(state, target_contains: Optional[str] = None) -> List[Any]:
    try:
        return list(state.find_detector_states(target_id_contain_str=target_contains) or [])
    except Exception:
        return []


def incoming_threat_count(state) -> int:
    threats = get_detector_states(state, "AttackMissile")
    if not threats:
        threats = get_detector_states(state, "CruiseMissile")
    return len(threats)


def distance_between_positions(position_a, position_b) -> Optional[float]:
    if position_a is None or position_b is None:
        return None
    try:
        if hasattr(position_a, "distance") and callable(getattr(position_a, "distance")):
            return float(position_a.distance(position_b))
    except Exception:
        pass
    try:
        if hasattr(position_a, "distance_to") and callable(getattr(position_a, "distance_to")):
            return float(position_a.distance_to(position_b))
    except Exception:
        pass
    return None


def distance_unit_to_target(state, unit_id: str, target_id: str) -> Optional[float]:
    try:
        distance = state.distance(unit_id, target_id)
        if distance is not None:
            return float(distance)
    except Exception:
        pass
    unit_pos = get_unit_position(state=state, unit_id=unit_id)
    target_pos = get_target_position(state, target_id)
    return distance_between_positions(unit_pos, target_pos)


def recent_target_track(history, target_id: str, n: int = 3) -> List[Any]:
    if history is None or not hasattr(history, "last_n_records"):
        return []
    try:
        return list(history.last_n_records(target_id, n) or [])
    except Exception:
        return []


def predict_target_position(history, target_id: str, delta_t: float):
    if history is None or not hasattr(history, "predict_target_position"):
        return None
    try:
        return history.predict_target_position(target_id, delta_t)
    except Exception:
        return None
