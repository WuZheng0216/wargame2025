import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from runtime_paths import project_root


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _build_entry(
    *,
    typ: str,
    observation: str,
    lesson: str,
    tags: list[str],
    phase: str,
    target_type: str,
    symptom: str,
    trigger: str,
    recommended_action: str,
    score_pattern: str,
    cost_risk: str,
) -> dict:
    created_at = _now_utc()
    normalized_lesson = _normalize(lesson)
    lesson_hash = hashlib.sha1(normalized_lesson.encode("utf-8")).hexdigest()
    lesson_id = lesson_hash[:16]
    source_batch_id = "minimal_red_baseline_20260405"
    return {
        "lesson_id": lesson_id,
        "type": typ,
        "observation": observation,
        "lesson": lesson,
        "tags": tags,
        "phase": phase,
        "target_type": target_type,
        "symptom": symptom,
        "trigger": trigger,
        "recommended_action": recommended_action,
        "score_pattern": score_pattern,
        "cost_risk": cost_risk,
        "battle_meta": {
            "event_log_path": "",
            "final_score": "",
            "side": "red",
            "written_at": created_at,
            "source_batch_id": source_batch_id,
            "source_run_id": "",
            "source_written_at": created_at,
            "source_mode": "minimal_curated_baseline",
        },
        "source": "manual_minimal_baseline",
        "created_at": created_at,
        "normalized_lesson": normalized_lesson,
        "lesson_hash": lesson_hash,
        "source_batch_id": source_batch_id,
        "source_run_id": "",
        "source_written_at": created_at,
        "source_mode": "minimal_curated_baseline",
    }


def _minimal_lessons() -> list[dict]:
    return [
        _build_entry(
            typ="failure",
            observation="开局全压主攻、不给预备队，容易被语义规则打回并让节奏变慢。",
            lesson="开局至少保留三分之一火力车作为预备队，第一轮先满足预备队和基础侦察，再展开主攻。",
            tags=["开局", "预备队", "语义规则"],
            phase="opening",
            target_type="surface_group",
            symptom="semantic_replan_due_to_missing_reserve",
            trigger="开局存在可打目标且分配方案想一次性投入全部火力车",
            recommended_action="reserve_force_first_then_attack",
            score_pattern="减少开局空转并更快进入有效提交",
            cost_risk="过度前压会导致开局空转和成本失控",
        ),
        _build_entry(
            typ="failure",
            observation="UAV 经常被当成一个大团给同一条 ScoutArea，缺少侧向展开和分工。",
            lesson="侦察无人机至少分成前出、左翼、右翼三组，不要把所有 UAV 长时间塞进同一个侦察任务。",
            tags=["UAV", "侦察", "编队搜索"],
            phase="opening",
            target_type="recon",
            symptom="uav_blob_search",
            trigger="scout_tasks 只包含单一目标或单一区域且 UAV 数量大于等于 4",
            recommended_action="split_uav_lateral_search",
            score_pattern="提高航迹刷新覆盖并减少 stale track",
            cost_risk="侦察成团会降低目标更新质量并拖慢火力转化",
        ),
        _build_entry(
            typ="success",
            observation="运动中的蓝方舰艇在有引导、航迹较新时更容易被后续打击转换成有效得分。",
            lesson="对运动舰艇优先建立 GuideAttack，再等待短暂稳定时间后组织火力，不要在引导刚建立的同一拍急着齐射。",
            tags=["引导", "运动目标", "命中率"],
            phase="midgame",
            target_type="moving_surface",
            symptom="guided_salvo_too_early",
            trigger="主攻目标为运动舰艇且 guide 刚刚建立",
            recommended_action="guide_then_wait_then_fire",
            score_pattern="更容易把接近命中的打击转成正收益",
            cost_risk="引导未稳定就发射会放大低成本弹浪费",
        ),
        _build_entry(
            typ="failure",
            observation="低成本弹在远距离、旧航迹、无引导时容易高成本低收益。",
            lesson="低成本弹不要作为远距离运动舰的默认主攻武器；当航迹陈旧或无引导时，应先补侦察、补引导或机动靠近。",
            tags=["低成本弹", "航迹新鲜度", "成本控制"],
            phase="midgame",
            target_type="moving_surface",
            symptom="low_cost_long_range_waste",
            trigger="target track stale 或没有 guide 且 low-cost 飞行时间过长",
            recommended_action="refresh_and_guide_before_low_cost",
            score_pattern="减少低收益发射并改善成本收益比",
            cost_risk="长距离低成本齐射会快速抬高 redCost",
        ),
        _build_entry(
            typ="success",
            observation="当 fire_now 窗口明确存在时，至少有一条真正的火力动作更容易通过语义校验并形成节奏。",
            lesson="如果 target_table 中存在 fire_now 目标，allocation_plan 里必须包含至少一条对应的 main_attack，不要只给侦察和引导。",
            tags=["火力窗口", "语义规则", "节奏"],
            phase="midgame",
            target_type="surface_group",
            symptom="fire_window_but_no_fire_action",
            trigger="存在 fire_now 目标但计划只包含 scout 或 guidance",
            recommended_action="always_commit_one_fire_action_for_fire_now",
            score_pattern="减少 semantic_replan 并维持主攻节奏",
            cost_risk="只侦察不打会浪费窗口并拖慢收益积累",
        ),
        _build_entry(
            typ="failure",
            observation="已经锁定在 task_board 上的单位若频繁被改派，容易触发 semantic_replan。",
            lesson="在任务未到期前，优先沿用已锁定的 guide/scout/fire 任务，只对可重分配单位做调整。",
            tags=["task_board", "重规划", "稳定性"],
            phase="midgame",
            target_type="task_management",
            symptom="locked_task_reassigned",
            trigger="已有非空 task_board 且新计划尝试更改相同单位的任务或目标",
            recommended_action="respect_locked_tasks",
            score_pattern="减少中段空转并提升动作提交稳定性",
            cost_risk="频繁重派会消耗窗口并降低节奏稳定性",
        ),
        _build_entry(
            typ="success",
            observation="高价值目标达成后，继续盲目扩火容易把 destroy score 优势变成 cost 负担。",
            lesson="主目标得手后先复核 score 与剩余窗口，再决定是否扩火；若新增收益不明确，优先保持侦察和机动而不是继续扫射。",
            tags=["高价值目标", "收手", "成本收益"],
            phase="endgame",
            target_type="high_value_surface",
            symptom="overfire_after_primary_success",
            trigger="destroy score 已明显领先但 allocator 仍持续扩大火力",
            recommended_action="reassess_after_primary_kill",
            score_pattern="防止领先后被成本反噬",
            cost_risk="后续盲目扩火会抬高 redCost 并削弱净得分",
        ),
        _build_entry(
            typ="success",
            observation="侦察刷新、引导和火力分层配合时，主攻动作更容易转成稳定正收益。",
            lesson="把侦察刷新、引导保障、火力投射当成三层链路：先刷新航迹，再稳定引导，最后提交火力，不要把三者挤在一个时刻完成。",
            tags=["侦察", "引导", "火力链路"],
            phase="midgame",
            target_type="surface_group",
            symptom="sensor_guide_fire_coupling",
            trigger="同一轮同时想完成 scout、guide、fire 且目标仍在运动",
            recommended_action="stagger_scout_guide_fire",
            score_pattern="提高火力链路稳定性和命中兑现率",
            cost_risk="三层链路同时拥挤会让火力动作质量下降",
        ),
    ]


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    root = Path(project_root())
    test_dir = root / "test"
    active_path = test_dir / "red_lessons_structured.jsonl"
    minimal_path = test_dir / "red_lessons_minimal.jsonl"
    archive_dir = test_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    entries = _minimal_lessons()
    _write_jsonl(minimal_path, entries)

    backup_path = None
    if active_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = archive_dir / f"red_lessons_pre_minimal_{timestamp}.jsonl"
        shutil.copyfile(active_path, backup_path)

    shutil.copyfile(minimal_path, active_path)

    print(f"minimal_path={minimal_path}")
    print(f"active_path={active_path}")
    if backup_path:
        print(f"backup_path={backup_path}")
    print(f"entry_count={len(entries)}")


if __name__ == "__main__":
    main()
