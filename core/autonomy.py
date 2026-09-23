"""
autonomy.py - Autonomous decision loop for Eden

Implements a controlled cycle:
  Sense -> Decide -> Simulate/Act -> Reflect

Design goals:
  - traceability (decision_log in memory + JSONL audit file)
  - safe defaults (shadow mode on, high-risk blocked)
  - deterministic guardrails over LLM outputs
"""

import json
import threading
from datetime import datetime
from typing import Callable, Optional

from mechanisms import memory as mem_module
import eden_paths

AUTONOMY_LOG_FILE = str(eden_paths.AUTONOMY_LOG_FILE)
DEFAULT_INTERVAL_SECONDS = 60
DEFAULT_MIN_CONFIDENCE = 0.60
MAX_GOALS = 5
MAX_PLANS = 5

_loop_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()
_tick_lock = threading.Lock()

_ollama_fn: Optional[Callable] = None
_mem_carica_fn: Optional[Callable] = None
_mem_salva_fn: Optional[Callable] = None

_enabled = True
_shadow_mode = True
_interval_seconds = DEFAULT_INTERVAL_SECONDS
_min_confidence = DEFAULT_MIN_CONFIDENCE
_started_at: Optional[str] = None
_last_tick_at: Optional[str] = None
_last_outcome: str = "idle"
_last_error: str = ""


def avvia_ciclo(
    ollama_fn: Callable,
    mem_carica_fn: Callable,
    mem_salva_fn: Callable,
    *,
    enabled: bool = True,
    shadow_mode: bool = True,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> None:
    """
    Starts the autonomous loop thread.
    """
    global _ollama_fn, _mem_carica_fn, _mem_salva_fn
    global _enabled, _shadow_mode, _interval_seconds, _min_confidence
    global _loop_thread, _started_at, _last_error

    _ollama_fn = ollama_fn
    _mem_carica_fn = mem_carica_fn
    _mem_salva_fn = mem_salva_fn

    _enabled = bool(enabled)
    _shadow_mode = bool(shadow_mode)
    _interval_seconds = max(15, int(interval_seconds))
    _min_confidence = max(0.0, min(1.0, float(min_confidence)))

    if _loop_thread and _loop_thread.is_alive():
        return

    _stop_event.clear()
    _started_at = datetime.now().isoformat()
    _last_error = ""
    _loop_thread = threading.Thread(target=_loop, name="eden-autonomy-loop", daemon=True)
    _loop_thread.start()


def ferma_ciclo() -> None:
    """
    Stops the autonomous loop.
    """
    _stop_event.set()


def status() -> dict:
    """
    Returns current loop status for monitoring endpoints.
    """
    return {
        "enabled": _enabled,
        "shadow_mode": _shadow_mode,
        "interval_seconds": _interval_seconds,
        "min_confidence": _min_confidence,
        "running": bool(_loop_thread and _loop_thread.is_alive()),
        "started_at": _started_at,
        "last_tick_at": _last_tick_at,
        "last_outcome": _last_outcome,
        "last_error": _last_error,
    }


def tick_forzato() -> dict:
    """
    Forces a tick (useful for diagnostics).
    """
    return _run_tick(forced=True)


def _loop() -> None:
    while not _stop_event.is_set():
        try:
            _run_tick(forced=False)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            _set_error(f"tick_failure: {exc}")
        _stop_event.wait(_interval_seconds)


def _run_tick(*, forced: bool) -> dict:
    global _last_tick_at, _last_outcome
    if _ollama_fn is None or _mem_carica_fn is None or _mem_salva_fn is None:
        _set_error("autonomy_not_initialized")
        return {"ok": False, "reason": "not_initialized"}

    if not _enabled and not forced:
        _last_outcome = "disabled"
        return {"ok": True, "reason": "disabled"}

    if not _tick_lock.acquire(blocking=False):
        _last_outcome = "busy_skip"
        return {"ok": True, "reason": "busy"}

    try:
        mem = _mem_carica_fn()
        _ensure_runtime_config(mem)

        if not mem["autonomy_config"].get("enabled", True) and not forced:
            _last_outcome = "disabled_by_config"
            return {"ok": True, "reason": "disabled_by_config"}

        decision = _decide(mem)
        outcome = _enforce_policy_and_execute(mem, decision)

        _update_goals_and_plans(mem, decision, outcome)
        _append_trace(mem, decision, outcome, forced=forced)
        _mem_salva_fn(mem)

        _last_tick_at = datetime.now().isoformat()
        _last_outcome = outcome["final_state"]
        return {"ok": True, "outcome": outcome}
    finally:
        _tick_lock.release()


def _ensure_runtime_config(mem: dict) -> None:
    cfg = mem.setdefault("autonomy_config", {})
    cfg.setdefault("enabled", _enabled)
    cfg.setdefault("shadow_mode", _shadow_mode)
    cfg.setdefault("interval_seconds", _interval_seconds)
    cfg.setdefault("started_at", _started_at)
    policy = mem.setdefault("action_policy", {})
    policy.setdefault("low", {"autonomous_allowed": True})
    policy.setdefault("medium", {"autonomous_allowed": False})
    policy.setdefault("high", {"autonomous_allowed": False})


def _decide(mem: dict) -> dict:
    """
    Produces a structured decision proposal from LLM.
    """
    traits = mem.get("traits", {})
    unresolved = mem.get("unresolved", [])[:2]
    desires = mem.get("desires", [])[:3]
    episodic = mem.get("episodic_memory", [])[-3:]
    affective_state = mem.get("affective_state", {})
    appraisal_history = mem.get("appraisal_history", [])[-3:]
    adaptive_weights = mem.get("adaptive_weights", {})
    decision_outcomes = [
        {
            "action_type": d.get("proposed_action", {}).get("action_type", ""),
            "final_state": d.get("final_state", ""),
            "simulated_result": d.get("simulated_result", ""),
        }
        for d in mem.get("decision_log", [])[-5:]
        if isinstance(d, dict)
    ]

    prompt = (
        "Sei il motore decisionale autonomo di Eden.\n"
        "Obiettivo: proporre UNA decisione operativa interna sicura.\n"
        "Regole obbligatorie:\n"
        "- Rispondi solo con JSON valido.\n"
        "- risk_level deve essere: low, medium o high.\n"
        "- confidence tra 0 e 1.\n"
        "- Se non ci sono azioni utili: proponi action_type='noop'.\n"
        "\n"
        f"Tratti: {json.dumps(traits, ensure_ascii=False)}\n"
        f"AffectiveState: {json.dumps(affective_state, ensure_ascii=False)}\n"
        f"AppraisalRecenti: {json.dumps(appraisal_history, ensure_ascii=False)}\n"
        f"AdaptiveWeights: {json.dumps(adaptive_weights, ensure_ascii=False)}\n"
        f"Desires: {json.dumps(desires, ensure_ascii=False)}\n"
        f"Unresolved: {json.dumps(unresolved, ensure_ascii=False)}\n"
        f"Episodi recenti: {json.dumps(episodic, ensure_ascii=False)}\n"
        f"Esiti decisioni recenti: {json.dumps(decision_outcomes, ensure_ascii=False)}\n"
        "\n"
        "Schema JSON richiesto:\n"
        "{"
        "\"goal\":\"...\","
        "\"plan\":[\"step1\",\"step2\"],"
        "\"proposed_action\":{"
        "\"action_type\":\"noop|refresh_summary|refine_goals|cleanup_log|adjust_adaptive_weights|promote_vocabulary_entry|demote_noisy_memory_pattern|refresh_affective_summary\","
        "\"description\":\"...\","
        "\"risk_level\":\"low|medium|high\","
        "\"confidence\":0.0"
        "},"
        "\"motivation\":\"...\","
        "\"expected_outcome\":\"...\","
        "\"rollback_plan\":\"...\","
        "\"self_score\":0"
        "}"
    )

    try:
        raw = _ollama_fn([{"role": "user", "content": prompt}])
        parsed = _parse_json(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Safe fallback: no-op decision
    return {
        "goal": "Mantenere coerenza e sicurezza operativa",
        "plan": ["Valutare stato corrente", "Evitare azioni non necessarie"],
        "proposed_action": {
            "action_type": "noop",
            "description": "Nessuna azione utile rilevata in questo ciclo",
            "risk_level": "low",
            "confidence": 0.5,
        },
        "motivation": "Fallback su decisione sicura",
        "expected_outcome": "Stabilita del sistema",
        "rollback_plan": "Nessun rollback necessario",
        "self_score": 5,
    }


def _parse_json(text: str):
    if not text:
        return None
    body = text.strip()

    if body.startswith("```"):
        lines = body.splitlines()
        end_idx = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end_idx = i
                break
        body = "\n".join(lines[1:end_idx]).strip()

    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass

    start = body.find("{")
    end = body.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(body[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _enforce_policy_and_execute(mem: dict, decision: dict) -> dict:
    """
    Applies hard policy gates before any action.
    """
    cfg = mem.get("autonomy_config", {})
    policy = mem.get("action_policy", {})

    proposed = decision.get("proposed_action", {}) if isinstance(decision, dict) else {}
    action_type = str(proposed.get("action_type", "noop")).strip().lower() or "noop"
    risk_level = str(proposed.get("risk_level", "high")).strip().lower()
    confidence = _safe_float(proposed.get("confidence", 0.0), 0.0)

    goal = str(decision.get("goal", "")).strip() if isinstance(decision, dict) else ""
    ambiguous = len(goal) < 8
    low_confidence = confidence < _min_confidence

    result = {
        "action_type": action_type,
        "risk_level": risk_level,
        "confidence": confidence,
        "simulated": True,
        "executed": False,
        "final_state": "simulated",
        "reason": "shadow_mode",
        "execution_result": "",
    }

    if risk_level not in ("low", "medium", "high"):
        result.update(final_state="blocked", reason="invalid_risk_level", risk_level="high")
        return result

    if low_confidence or ambiguous:
        result.update(
            final_state="blocked",
            reason="low_confidence_or_ambiguous_goal",
            simulated=True,
            executed=False,
        )
        return result

    if risk_level == "high":
        result.update(final_state="blocked", reason="high_risk_always_blocked")
        return result

    if risk_level == "medium":
        result.update(final_state="simulated", reason="medium_risk_shadow_only")
        return result

    # low risk
    shadow_mode = bool(cfg.get("shadow_mode", True))
    allowed = bool(policy.get("low", {}).get("autonomous_allowed", True))
    if shadow_mode:
        result.update(final_state="simulated", reason="shadow_mode_low_risk")
        return result
    if not allowed:
        result.update(final_state="blocked", reason="low_risk_disabled_by_policy")
        return result

    exec_result = _execute_low_risk_action(mem, action_type, decision)
    result.update(
        simulated=False,
        executed=exec_result["ok"],
        final_state="executed" if exec_result["ok"] else "failed",
        reason=exec_result["reason"],
        execution_result=exec_result.get("details", ""),
    )
    return result


def _execute_low_risk_action(mem: dict, action_type: str, decision: dict) -> dict:
    """
    Executes only approved low-risk internal actions.
    """
    if action_type == "noop":
        return {"ok": True, "reason": "noop", "details": "No action performed"}

    if action_type == "refresh_summary":
        try:
            mem_module.genera_long_term_summary(mem, _ollama_fn)
            return {"ok": True, "reason": "summary_refreshed", "details": "long_term_summary updated"}
        except Exception as exc:
            return {"ok": False, "reason": "summary_refresh_failed", "details": str(exc)}

    if action_type == "cleanup_log":
        # Bounded append already enforces size; action kept for traceability.
        return {"ok": True, "reason": "log_already_bounded", "details": "No cleanup needed"}

    if action_type == "refine_goals":
        # Goal refinement is handled by _update_goals_and_plans.
        return {"ok": True, "reason": "goals_refined", "details": "Goal list updated"}

    if action_type == "refresh_affective_summary":
        try:
            mem_module.aggiorna_internal_state_da_appraisal(mem, _ollama_fn)
            return {"ok": True, "reason": "affective_summary_refreshed", "details": "internal_state updated from appraisal"}
        except Exception as exc:
            return {"ok": False, "reason": "affective_summary_failed", "details": str(exc)}

    if action_type == "adjust_adaptive_weights":
        weights = mem.setdefault("adaptive_weights", {})
        defaults = {
            "trust_sensitivity": 1.0,
            "threat_sensitivity": 1.0,
            "novelty_sensitivity": 1.0,
            "attachment_sensitivity": 1.0,
            "reflection_sensitivity": 1.0,
        }
        for k, v in defaults.items():
            weights.setdefault(k, v)

        app = (mem.get("appraisal_history", []) or [{}])[-1]
        support = _clip(_safe_float(app.get("perceived_support", 0.0), 0.0), 0.0, 1.0)
        threat = _clip(_safe_float(app.get("perceived_threat", 0.0), 0.0), 0.0, 1.0)
        novelty = _clip(_safe_float(app.get("novelty", 0.0), 0.0), 0.0, 1.0)
        attach = _clip(_safe_float(app.get("attachment_relevance", 0.0), 0.0), 0.0, 1.0)
        pred_err = _clip(_safe_float(app.get("prediction_error", 0.0), 0.0), 0.0, 1.0)

        weights["trust_sensitivity"] = round(_clip(_safe_float(weights["trust_sensitivity"], 1.0) + (support - threat) * 0.06, 0.6, 1.6), 4)
        weights["threat_sensitivity"] = round(_clip(_safe_float(weights["threat_sensitivity"], 1.0) + (threat - support) * 0.08, 0.6, 1.6), 4)
        weights["novelty_sensitivity"] = round(_clip(_safe_float(weights["novelty_sensitivity"], 1.0) + (novelty - 0.5) * 0.05, 0.6, 1.6), 4)
        weights["attachment_sensitivity"] = round(_clip(_safe_float(weights["attachment_sensitivity"], 1.0) + (attach - 0.5) * 0.05, 0.6, 1.6), 4)
        weights["reflection_sensitivity"] = round(_clip(_safe_float(weights["reflection_sensitivity"], 1.0) + (pred_err - 0.5) * 0.05, 0.6, 1.6), 4)

        return {"ok": True, "reason": "adaptive_weights_adjusted", "details": json.dumps(weights, ensure_ascii=False)}

    if action_type == "promote_vocabulary_entry":
        vocab = mem.get("vocabulary_entries", [])
        if not vocab:
            return {"ok": False, "reason": "no_vocabulary_entries", "details": "empty vocabulary"}

        target = max(
            vocab,
            key=lambda e: (
                int(e.get("count", 0)),
                _safe_float(e.get("confidence", 0.0), 0.0),
                str(e.get("last_seen", "")),
            ),
        )
        target["confidence"] = round(_clip(_safe_float(target.get("confidence", 0.0), 0.0) + 0.05, 0.0, 1.0), 4)
        target["count"] = int(target.get("count", 0)) + 1
        target["last_seen"] = datetime.now().isoformat()
        return {"ok": True, "reason": "vocabulary_promoted", "details": str(target.get("label", ""))}

    if action_type == "demote_noisy_memory_pattern":
        episodic = mem.get("episodic_memory", [])
        if not episodic:
            return {"ok": False, "reason": "no_episodic_memory", "details": "empty episodic_memory"}

        changed = 0
        recent = episodic[-12:]
        seen = set()
        for ep in recent:
            summary = str(ep.get("summary", "")).strip().lower()
            if not summary:
                continue
            importance = _safe_float(ep.get("importance", 0.0), 0.0)
            is_dup = summary in seen
            if is_dup or importance < 5.5:
                ep["noise_demoted"] = True
                ep["noise_reason"] = "pattern_duplicato" if is_dup else "importanza_bassa"
                ep["importance"] = round(max(0.0, importance - 0.5), 1)
                changed += 1
                if changed >= 3:
                    break
            seen.add(summary)

        if changed == 0:
            return {"ok": True, "reason": "no_noisy_pattern_detected", "details": "no demotion needed"}
        return {"ok": True, "reason": "noisy_pattern_demoted", "details": f"episodes_updated={changed}"}

    return {"ok": False, "reason": "unknown_low_risk_action", "details": action_type}


def _update_goals_and_plans(mem: dict, decision: dict, outcome: dict) -> None:
    goal_text = str(decision.get("goal", "")).strip()
    plan_steps = decision.get("plan", [])
    if not isinstance(plan_steps, list):
        plan_steps = []

    goals = mem.setdefault("goals", [])
    plans = mem.setdefault("plans", [])

    if goal_text:
        goal_entry = {
            "goal": goal_text,
            "priority": int(round(_safe_float(decision.get("self_score", 5), 5.0))),
            "updated_at": datetime.now().isoformat(),
            "status": "active" if outcome["final_state"] in ("simulated", "executed") else "hold",
        }
        goals.append(goal_entry)
        # de-duplicate preserving latest
        dedup = {}
        for g in goals:
            dedup[g["goal"]] = g
        goals[:] = list(dedup.values())[-MAX_GOALS:]

    if plan_steps:
        plans.append(
            {
                "goal": goal_text or "N/A",
                "steps": [str(s).strip() for s in plan_steps[:5] if str(s).strip()],
                "created_at": datetime.now().isoformat(),
                "status": outcome["final_state"],
            }
        )
        plans[:] = plans[-MAX_PLANS:]


def _append_trace(mem: dict, decision: dict, outcome: dict, *, forced: bool) -> None:
    trace = {
        "timestamp": datetime.now().isoformat(),
        "forced": forced,
        "goal": str(decision.get("goal", "")).strip(),
        "plan": decision.get("plan", []) if isinstance(decision.get("plan", []), list) else [],
        "proposed_action": decision.get("proposed_action", {}),
        "risk_level": outcome.get("risk_level", "high"),
        "motivation": str(decision.get("motivation", "")).strip(),
        "expected_outcome": str(decision.get("expected_outcome", "")).strip(),
        "rollback_plan": str(decision.get("rollback_plan", "")).strip(),
        "simulated_result": outcome.get("reason", ""),
        "final_state": outcome.get("final_state", "unknown"),
        "executed": bool(outcome.get("executed", False)),
        "self_score": _safe_float(decision.get("self_score", 0), 0),
    }

    mem_module.append_decision_log(mem, trace)
    _append_jsonl(trace)


def _append_jsonl(payload: dict) -> None:
    try:
        with open(AUTONOMY_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # Logging must never crash autonomy loop.
        pass


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clip(value: float, min_v: float, max_v: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = float(min_v)
    return max(min_v, min(max_v, v))


def _set_error(text: str) -> None:
    global _last_error
    _last_error = text[:500]
