import math
import os
from collections import Counter
from typing import Dict, Iterable, List, Optional, Set, Tuple


RED_SHIP_TARGET_PREFIXES = (
    "Flagship_Surface",
    "Cruiser_Surface",
    "Destroyer_Surface",
)
RED_TRUCK_ONLY_TYPES = {"MoveToEngage", "FocusFire", "ShootAndScoot"}
RED_GUIDE_TYPES = {"GuideAttack"}
RED_TRUCK_PREFIX = "Truck_Ground"
RED_GUIDE_PREFIXES = ("Guide_Ship_Surface",)
RED_MISSILE_PREFIXES = ("HighCostAttackMissile", "LowCostAttackMissile")
RED_FIRE_TYPES = {"FocusFire", "ShootAndScoot"}
RED_SCOUT_TYPES = {"ScoutArea"}
RED_ALLOWED_ALLOCATION_TYPES = {"FocusFire", "ShootAndScoot", "MoveToEngage", "GuideAttack", "ScoutArea"}
RED_SCOUT_PREFIX = "Recon_UAV_FixWing"
RED_GUIDE_SHIP_PREFIX = "Guide_Ship_Surface"


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except Exception:
        return default


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        return float(raw)
    except Exception:
        return default


RED_MAX_ATTACK_GROUP_SIZE = _read_int_env("RED_MAX_ATTACK_GROUP_SIZE", 4)
RED_FLAGSHIP_MAX_ATTACK_GROUP_SIZE = _read_int_env("RED_FLAGSHIP_MAX_ATTACK_GROUP_SIZE", 6)
RED_MOVE_THEN_FIRE_COMMIT_RATIO = _read_float_env("RED_MOVE_THEN_FIRE_COMMIT_RATIO", 0.34)


def _as_list(value) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value if isinstance(v, str) and v]
    return []


def _has_prefix(value: str, prefixes: Iterable[str]) -> bool:
    text = str(value or "")
    return any(text.startswith(prefix) for prefix in prefixes)


def _is_red_ship_target(target_id: str) -> bool:
    return _has_prefix(target_id, RED_SHIP_TARGET_PREFIXES)


def _unit_ids_from_action(action: dict) -> List[str]:
    if not isinstance(action, dict):
        return []
    return _as_list(action.get("UnitIds"))


def _target_id_from_action(action: dict) -> str:
    if not isinstance(action, dict):
        return ""
    return str(action.get("Target_Id", "")).strip()


def _collect_authorized_units(allocation_plan: dict) -> Set[str]:
    allowed: Set[str] = set()
    if not isinstance(allocation_plan, dict):
        return allowed

    for key in ("main_attack", "support_guidance", "scout_tasks"):
        for item in allocation_plan.get(key, []) or []:
            if not isinstance(item, dict):
                continue
            allowed.update(_as_list(item.get("unit_ids")))

    allowed.update(_as_list(allocation_plan.get("reserve_units")))
    allowed.update(_as_list(allocation_plan.get("withheld_units")))
    return allowed


def _collect_authorized_targets(allocation_plan: dict) -> Set[str]:
    allowed: Set[str] = set()
    if not isinstance(allocation_plan, dict):
        return allowed

    for key in ("main_attack", "support_guidance"):
        for item in allocation_plan.get(key, []) or []:
            if not isinstance(item, dict):
                continue
            target_id = str(item.get("target_id", "")).strip()
            if target_id:
                allowed.add(target_id)
    return allowed


def _available_truck_ids(unit_roster: List[dict]) -> Set[str]:
    trucks: Set[str] = set()
    for unit in unit_roster or []:
        if not isinstance(unit, dict):
            continue
        if str(unit.get("role")) != "fire":
            continue
        if not unit.get("alive", True):
            continue
        unit_id = str(unit.get("unit_id", "")).strip()
        if unit_id:
            trucks.add(unit_id)
    return trucks


def _available_guide_ids(unit_roster: List[dict]) -> Set[str]:
    guides: Set[str] = set()
    for unit in unit_roster or []:
        if not isinstance(unit, dict):
            continue
        if str(unit.get("role")) != "guide":
            continue
        if not unit.get("alive", True) or not unit.get("available", True):
            continue
        unit_id = str(unit.get("unit_id", "")).strip()
        if unit_id:
            guides.add(unit_id)
    return guides


def _target_table_map(target_table: List[dict]) -> Dict[str, dict]:
    table = {}
    for target in target_table or []:
        if not isinstance(target, dict):
            continue
        target_id = str(target.get("target_id", "")).strip()
        if target_id:
            table[target_id] = target
    return table


def _is_flagship_target(target_id: str) -> bool:
    return str(target_id or "").startswith("Flagship_Surface")


def _max_attack_group_size(target_id: str, attack_window: str) -> int:
    if attack_window == "fire_now" and _is_flagship_target(target_id):
        return max(1, RED_FLAGSHIP_MAX_ATTACK_GROUP_SIZE)
    return max(1, RED_MAX_ATTACK_GROUP_SIZE)


def _allowed_move_then_fire_commit(truck_count: int) -> int:
    if truck_count <= 0:
        return 0
    return max(1, int(math.ceil(truck_count * RED_MOVE_THEN_FIRE_COMMIT_RATIO)))


def _locked_task_entries(task_board: Optional[dict]) -> Dict[str, dict]:
    locked: Dict[str, dict] = {}
    if not isinstance(task_board, dict):
        return locked
    for unit_id, task in task_board.items():
        if not isinstance(task, dict):
            continue
        if task.get("reassignable") is False:
            locked[str(unit_id)] = task
    return locked


def _bucket_contract_issues(
    allocation_plan: dict,
    task_board: Optional[dict] = None,
    target_table: Optional[List[dict]] = None,
) -> List[str]:
    if not isinstance(allocation_plan, dict):
        return []

    issues: List[str] = []
    main_targets: Set[str] = set()
    locked_tasks = _locked_task_entries(task_board)
    target_map = _target_table_map(target_table or [])
    bucket_specs = {
        "main_attack": {
            "types": {"FocusFire", "ShootAndScoot", "MoveToEngage"},
            "prefixes": (RED_TRUCK_PREFIX,),
        },
        "support_guidance": {
            "types": {"GuideAttack"},
            "prefixes": (RED_GUIDE_SHIP_PREFIX,),
        },
        "scout_tasks": {
            "types": {"ScoutArea"},
            "prefixes": (RED_SCOUT_PREFIX,),
        },
    }

    for bucket_name, spec in bucket_specs.items():
        for item in allocation_plan.get(bucket_name, []) or []:
            if not isinstance(item, dict):
                continue
            task_type = str(item.get("task_type", "")).strip()
            unit_ids = _as_list(item.get("unit_ids"))
            target_id = str(item.get("target_id", "")).strip()
            attack_window = str(item.get("attack_window", "")).strip() or str(
                target_map.get(target_id, {}).get("attack_window", "")
            ).strip()

            if task_type not in spec["types"]:
                issues.append("invalid_role_bucket_assignment")
            for unit_id in unit_ids:
                if not _has_prefix(unit_id, spec["prefixes"]):
                    issues.append("invalid_role_bucket_assignment")
                locked = locked_tasks.get(unit_id)
                if locked and (
                    str(locked.get("task_type", "")).strip() != task_type
                    or str(locked.get("target_id", "")).strip() != target_id
                ):
                    issues.append("locked_task_reassigned")

            if bucket_name == "main_attack":
                if target_id in main_targets:
                    issues.append("unauthorized_target_rewrite")
                elif target_id:
                    main_targets.add(target_id)
                if unit_ids and len(unit_ids) > _max_attack_group_size(target_id, attack_window):
                    issues.append("over_sized_attack_group")

    deduped: List[str] = []
    for issue in issues:
        if issue not in deduped:
            deduped.append(issue)
    return deduped


def _required_reserve(truck_count: int) -> int:
    if truck_count <= 0:
        return 0
    return max(1, truck_count // 3)


def validate_red_action_semantics(action: dict) -> List[str]:
    if not isinstance(action, dict):
        return []

    action_type = str(action.get("Type", "")).strip()
    unit_ids = _as_list(action.get("UnitIds"))
    target_id = str(action.get("Target_Id", "")).strip()
    reasons: List[str] = []

    if action_type in RED_TRUCK_ONLY_TYPES:
        for unit_id in unit_ids:
            if _has_prefix(unit_id, RED_MISSILE_PREFIXES):
                reasons.append("missile_used_as_unit")
                continue
            if not unit_id.startswith(RED_TRUCK_PREFIX):
                reasons.append("invalid_unit_type")
        if target_id and not _is_red_ship_target(target_id):
            reasons.append("invalid_target_type")

    elif action_type in RED_GUIDE_TYPES:
        for unit_id in unit_ids:
            if _has_prefix(unit_id, RED_MISSILE_PREFIXES):
                reasons.append("missile_used_as_unit")
                continue
            if not _has_prefix(unit_id, RED_GUIDE_PREFIXES):
                reasons.append("invalid_unit_type")
        if target_id and not _is_red_ship_target(target_id):
            reasons.append("invalid_target_type")

    elif action_type in RED_SCOUT_TYPES:
        for unit_id in unit_ids:
            if _has_prefix(unit_id, RED_MISSILE_PREFIXES):
                reasons.append("missile_used_as_unit")
                continue
            if not unit_id.startswith(RED_SCOUT_PREFIX):
                reasons.append("invalid_unit_type")

    deduped: List[str] = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    return deduped


def validate_red_plan_semantics(actions_json: List[dict], validation_context: Optional[dict] = None) -> List[str]:
    validation_context = validation_context or {}
    reasons: List[str] = []
    seen_units: Set[str] = set()
    duplicate_found = False
    allocation_plan = validation_context.get("allocation_plan") or {}
    task_board = validation_context.get("task_board") or {}
    target_table = validation_context.get("target_table") or []
    target_map = _target_table_map(target_table)

    reasons.extend(_bucket_contract_issues(allocation_plan, task_board=task_board, target_table=target_table))

    for action in actions_json or []:
        for unit_id in _unit_ids_from_action(action):
            if unit_id in seen_units:
                duplicate_found = True
            seen_units.add(unit_id)
    if duplicate_found:
        reasons.append("duplicate_unit_assignment")

    authorized_units = _collect_authorized_units(allocation_plan)
    authorized_targets = _collect_authorized_targets(allocation_plan)
    if authorized_units:
        for action in actions_json or []:
            for unit_id in _unit_ids_from_action(action):
                if unit_id not in authorized_units:
                    reasons.append("unauthorized_unit_in_operator_output")
                    break
            target_id = _target_id_from_action(action)
            if target_id and authorized_targets and target_id not in authorized_targets:
                reasons.append("unauthorized_target_rewrite")

    unit_roster = validation_context.get("unit_roster") or []
    truck_ids = _available_truck_ids(unit_roster)
    committed_trucks: Set[str] = set()
    fire_actions = 0
    guide_actions = 0
    guided_targets: Set[str] = set()
    locked_tasks = _locked_task_entries(task_board)
    for action in actions_json or []:
        if not isinstance(action, dict):
            continue
        action_type = str(action.get("Type", "")).strip()
        if action_type in RED_TRUCK_ONLY_TYPES:
            committed_trucks.update(uid for uid in _unit_ids_from_action(action) if uid in truck_ids)
        if action_type in RED_FIRE_TYPES:
            fire_actions += 1
            target_id = _target_id_from_action(action)
            attack_window = str(target_map.get(target_id, {}).get("attack_window", "")).strip()
            if attack_window == "observe_only":
                reasons.append("invalid_attack_window_policy")
            if len(_unit_ids_from_action(action)) > _max_attack_group_size(target_id, attack_window):
                reasons.append("over_sized_attack_group")
        elif action_type in RED_GUIDE_TYPES:
            guide_actions += 1
            target_id = _target_id_from_action(action)
            if target_id:
                guided_targets.add(target_id)

        for unit_id in _unit_ids_from_action(action):
            locked = locked_tasks.get(unit_id)
            if locked and (
                str(locked.get("task_type", "")).strip() != action_type
                or str(locked.get("target_id", "")).strip() != _target_id_from_action(action)
            ):
                reasons.append("locked_task_reassigned")

    reserve_units = set(_as_list((allocation_plan or {}).get("reserve_units")))
    if truck_ids:
        reserve_min = _required_reserve(len(truck_ids))
        max_commit = max(0, len(truck_ids) - reserve_min)
        if len(committed_trucks) > max_commit:
            reasons.append("over_committed_trucks")
        if len(reserve_units & truck_ids) < reserve_min:
            reasons.append("missing_reserve_force")
    for action in actions_json or []:
        if not isinstance(action, dict):
            continue
        if str(action.get("Type", "")).strip() in RED_TRUCK_ONLY_TYPES:
            if reserve_units.intersection(_unit_ids_from_action(action)):
                reasons.append("reserve_unit_used_in_fire_action")

    has_fire_now_target = any(
        isinstance(target, dict) and str(target.get("attack_window")) == "fire_now"
        for target in target_table
    )
    if has_fire_now_target and fire_actions == 0:
        reasons.append("missing_fire_action_with_valid_window")
    guide_ids = _available_guide_ids(unit_roster)
    if guide_ids:
        fire_now_ship_targets = {
            str(target.get("target_id"))
            for target in target_table
            if isinstance(target, dict)
            and str(target.get("attack_window")) == "fire_now"
            and _is_red_ship_target(str(target.get("target_id", "")))
        }
        attacked_ship_targets = {
            _target_id_from_action(action)
            for action in actions_json or []
            if isinstance(action, dict)
            and str(action.get("Type", "")).strip() in RED_FIRE_TYPES
            and _target_id_from_action(action) in fire_now_ship_targets
        }
        if attacked_ship_targets and not (guided_targets & attacked_ship_targets):
            reasons.append("missing_guidance_support_for_ship_attack")

    move_then_fire_targets = {
        str(target.get("target_id"))
        for target in target_table
        if isinstance(target, dict) and str(target.get("attack_window")) == "move_then_fire"
    }
    if move_then_fire_targets:
        move_then_fire_committed = {
            unit_id
            for action in actions_json or []
            if isinstance(action, dict)
            and str(action.get("Type", "")).strip() == "MoveToEngage"
            and _target_id_from_action(action) in move_then_fire_targets
            for unit_id in _unit_ids_from_action(action)
            if unit_id in truck_ids
        }
        if len(move_then_fire_committed) > _allowed_move_then_fire_commit(len(truck_ids)):
            reasons.append("over_committed_trucks")

    deduped: List[str] = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    return deduped


def filter_red_actions_semantically(
    actions_json: List[dict], validation_context: Optional[Dict] = None
) -> Tuple[List[dict], Counter]:
    filtered: List[dict] = []
    reasons: Counter = Counter()

    for action in actions_json or []:
        issues = validate_red_action_semantics(action)
        if issues:
            reasons.update(issues)
            continue
        filtered.append(action)

    plan_issues = validate_red_plan_semantics(filtered, validation_context=validation_context)
    if plan_issues:
        reasons.update(plan_issues)
        filtered = []

    if actions_json and not filtered and reasons:
        reasons["empty_after_semantic_filter"] += 1

    return filtered, reasons
