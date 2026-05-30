# ----------------------------------------------------------------------------------------- #
# skeleton.py
# Workflow:
# Stage 1 - try small verified direct Isar proofs
# Stage 2 - if direct proof fails, use safe theorem-shape-aware outlines
# Stage 3 - LLM candidates are forced into outline form
# Stage 4 - safety scoring chooses among outlines
# Stage 5 - Fill/Prove handles remaining holes
# ----------------------------------------------------------------------------------------- #

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict, Iterable
import json
import os
import re
from functools import lru_cache
from planner.prompts import SKELETON_PROMPT
import requests

# Pull defaults from your existing prover/config.py
from prover.config import (
    MODEL as DEFAULT_MODEL,
    OLLAMA_HOST,
    TIMEOUT_S as OLLAMA_TIMEOUT_S,
    OLLAMA_NUM_PREDICT,
    TEMP as OLLAMA_TEMP,
    TOP_P as OLLAMA_TOP_P,
)

# Isabelle helpers (for quick sketch check)
from prover.isabelle_api import build_theory, run_theory, last_print_state_block, finished_ok
from prover.utils import parse_subgoals

# Reuse local-context miner from repair (defs/facts list)
from planner.repair import _facts_from_state as _facts_from_state

# One HTTP session (keep-alive)
_SESSION = requests.Session()

@dataclass(slots=True)
class Skeleton:
    text: str
    holes: List[Tuple[int, int]]  # (start_idx, end_idx) spans where 'sorry' occurs

SORRY_RE = re.compile(r"\bsorry\b")
PROOF_RE = re.compile(r"(?m)^\s*proof(?:\b|\s|\()", re.UNICODE)
QED_RE   = re.compile(r"(?m)^\s*qed\b", re.UNICODE)
BY_INLINE_RE = re.compile(r"\s+by\s+.*$")
# New regex helpers for sanitization
CASE_START_RE   = re.compile(r"(?m)^\s*case\b")
NEXT_OR_QED_RE  = re.compile(r"(?m)^\s*(next|qed)\b")
SHOW_THESIS_RE = re.compile(r"(?m)^\s*(?:then\s+)?show\s+\?thesis\b")
# Match the meta-variable immediately after 'show', capturing the 'show ' prefix so we can preserve any
# leading tokens like 'then', 'from ...', 'with ...', 'finally', etc. We only rewrite the '?thesis|?case' token.
SHOW_META_AT_SHOW = re.compile(r"(?m)(?P<prefix>\bshow\s+)\?(?P<meta>thesis|case)\b")
BARE_PROOF_RE  = re.compile(r"(?m)^\s*proof\s*$")
# General proof-mode detectors (for ?case/?thesis normalization)
_PROOF_OPEN_RE   = re.compile(r"(?m)^\s*proof(?:\s*\(([^)]+)\))?\s*$")
_QED_LINE_RE     = re.compile(r"(?m)^\s*qed\b")
_MODE_CASES_RE   = re.compile(r"^\s*cases\b")
_MODE_CASES_RULE = re.compile(r"^\s*cases\s+rule:")
_MODE_INDUCT_RE  = re.compile(r"^\s*(?:induction|induct|coinduction|coinduct)\b")
_HAS_TACTIC_NEXT = re.compile(r"(?m)^\s*(?:by\b|apply\b|proof\b|sorry\b|done\b)")
_HAVE_OR_SHOW    = re.compile(r"(?m)^\s*(have|show)\b")
_INLINE_BY       = re.compile(r"\s+by\s+.+$")
_CONTINUATION_HEAD = re.compile(r"(?m)^\s*(?:using|from|with|then|ultimately|finally|also|moreover)\b")
_STMT_OR_BOUNDARY = re.compile(r"(?m)^\s*(?:have|show|assume|case|next|qed)\b")

# =============================================================================
# Provider shims: Ollama (default), Hugging Face ("hf:"), Gemini ("gemini:")
# =============================================================================

def _ollama_generate_simple(
    prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    num_predict: Optional[int] = None,
    timeout_s: Optional[int] = None,
) -> str:
    url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
    payload = {
        "model": model or DEFAULT_MODEL,
        "prompt": prompt,
        "options": {
            "temperature": OLLAMA_TEMP if temperature is None else temperature,
            "top_p": OLLAMA_TOP_P if top_p is None else top_p,
            "num_predict": OLLAMA_NUM_PREDICT if num_predict is None else num_predict,
        },
        "stream": False,
    }
    resp = _SESSION.post(url, json=payload, timeout=timeout_s or OLLAMA_TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("response") or "").strip()

def _hf_generate_simple(
    prompt: str,
    model_id: str,
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    timeout_s: Optional[int] = None,
) -> str:
    token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACE_API_TOKEN is not set")
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}"}
    payload: Dict[str, Any] = {
        "inputs": prompt,
        "parameters": {
            "temperature": OLLAMA_TEMP if temperature is None else temperature,
            "top_p": OLLAMA_TOP_P if top_p is None else top_p,
            "max_new_tokens": OLLAMA_NUM_PREDICT if max_new_tokens is None else max_new_tokens,
            "return_full_text": False,
        },
        "options": {"wait_for_model": True},
    }
    resp = _SESSION.post(url, headers=headers, json=payload, timeout=timeout_s or OLLAMA_TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and data:
        return (data[0].get("generated_text") or "").strip()
    if isinstance(data, dict):
        if "generated_text" in data:
            return (data["generated_text"] or "").strip()
        choices = data.get("choices") or []
        if choices:
            t = choices[0].get("text") or choices[0].get("generated_text") or ""
            return str(t).strip()
    return str(data).strip()

@lru_cache(maxsize=1)
def _gemini_list_models_cached(api_key: str) -> List[str]:
    if not api_key:
        return []
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        resp = _SESSION.get(url, timeout=OLLAMA_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
        out = []
        for m in data.get("models", []):
            name = m.get("name", "")
            short = name.split("/")[-1] if name else ""
            if short:
                out.append(short)
        return out
    except Exception:
        return []

def _gemini_resolve_model_id(model_id: str, *, timeout_s: Optional[int] = None) -> str:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return model_id
    models = _gemini_list_models_cached(api_key)
    if model_id in models:
        return model_id
    cands = [m for m in models if m.startswith(model_id)]
    if cands:
        stable = [m for m in cands if ("preview" not in m and "exp" not in m)]
        return (stable or cands)[0]
    return model_id

def _gemini_cli_available() -> bool:
    from shutil import which
    return which("gemini") is not None

def _gemini_cli_generate_simple(prompt: str, model_id: str, *, timeout_s: Optional[int] = None) -> str:
    import subprocess
    cmd = ["gemini", "-m", model_id]
    proc = subprocess.run(
        cmd, input=prompt, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=timeout_s or OLLAMA_TIMEOUT_S, env=os.environ.copy()
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gemini CLI failed ({proc.returncode}): {(proc.stderr or proc.stdout).strip()}")
    return proc.stdout.strip()

def _gemini_rest_generate_simple(prompt: str, model_id: str, *, timeout_s: Optional[int] = None) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set (needed for Gemini REST)")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": OLLAMA_NUM_PREDICT}}
    resp = _SESSION.post(url, json=body, timeout=timeout_s or OLLAMA_TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()
    try:
        cands = data.get("candidates") or []
        if cands:
            parts = ((cands[0].get("content") or {}).get("parts")) or []
            if parts:
                return (parts[0].get("text") or "").strip()
    except Exception:
        pass
    return str(data).strip()

def _gemini_generate_simple(prompt: str, model_id: str, *, timeout_s: Optional[int] = None) -> str:
    resolved = _gemini_resolve_model_id(model_id, timeout_s=timeout_s)
    if _gemini_cli_available():
        try:
            return _gemini_cli_generate_simple(prompt, resolved, timeout_s=timeout_s)
        except Exception:
            pass
    try:
        return _gemini_rest_generate_simple(prompt, resolved, timeout_s=timeout_s)
    except Exception:
        fallback = "gemini-2.5-pro"
        if _gemini_cli_available():
            try:
                return _gemini_cli_generate_simple(prompt, fallback, timeout_s=timeout_s)
            except Exception:
                pass
        return _gemini_rest_generate_simple(prompt, fallback, timeout_s=timeout_s)

def _generate_simple(
    prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    num_predict: Optional[int] = None,
    timeout_s: Optional[int] = None,
) -> str:
    if model:
        if model.startswith("hf:"):
            return _hf_generate_simple(
                prompt, model_id=model[len("hf:"):],
                temperature=temperature, top_p=top_p,
                max_new_tokens=num_predict, timeout_s=timeout_s
            )
        if model.startswith("gemini:"):
            return _gemini_generate_simple(
                prompt, model_id=model[len("gemini:"):],
                timeout_s=timeout_s
            )
        if model.startswith("ollama:"):
            model = model[len("ollama:"):]
    return _ollama_generate_simple(
        prompt, model=model, temperature=temperature, top_p=top_p,
        num_predict=num_predict, timeout_s=timeout_s
    )

# -----------------------------------------------------------------------------
# Utilities: sorry spans, sanitize, state block, facts, scoring
# -----------------------------------------------------------------------------

def find_sorry_spans(isar: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in SORRY_RE.finditer(isar)]

def _ensure_lemma_header(text: str, goal: str) -> str:
    body = text.lstrip()
    if not body.startswith("lemma"):
        return f'lemma "{goal}"\n{body}'
    return text

def _normalize_calculation_ellipsis(text: str) -> str:
    # Replace Unicode ellipsis and spaced PDF (. . .) with Isar token "..."
    text = text.replace("…", "...")
    text = re.sub(r"\.\s*\.\s*\.", "...", text)
    return text

def _crop_to_first_proof_block(text: str) -> str:
    """
    Keep only the first lemma..(proof..qed)* block; be nesting-aware so the cropped
    text ends at the *matching* 'qed' for the first 'proof', not at an inner one.
    """
    # Find the first lemma line
    m_lemma = re.search(r'(?m)^\s*lemma\s+"[^"]*"', text)
    if not m_lemma:
        return text
    tail = text[m_lemma.start():]

    # Find the first 'proof' after that lemma
    m_proof = PROOF_RE.search(tail)
    if not m_proof:
        # No proof — still return from lemma onward (sanitizers may add a skeleton later)
        return tail

    # Walk forward and balance nested 'proof'/'qed'
    depth = 1
    end_idx = None
    pos = m_proof.end()

    # We want earliest upcoming PROOF or QED at each step
    while True:
        m_next_proof = PROOF_RE.search(tail, pos)
        m_next_qed   = QED_RE.search(tail, pos)

        if not m_next_proof and not m_next_qed:
            # No closing 'qed' — return as-is; later passes may append a final 'qed'
            break

        # Choose whichever comes first
        choose_qed = False
        if m_next_qed and m_next_proof:
            choose_qed = (m_next_qed.start() <= m_next_proof.start())
        elif m_next_qed and not m_next_proof:
            choose_qed = True
        else:
            choose_qed = False

        if choose_qed:
            depth -= 1
            pos = m_next_qed.end()
            if depth == 0:
                end_idx = pos
                break
        else:
            depth += 1
            pos = m_next_proof.end()

    if end_idx is None:
        # Could not find the matching outer 'qed'; return tail as-is (other sanitizers add one)
        return tail

    return tail[:end_idx] + "\n"

def _drop_redundant_sorry(text: str) -> str:
    """Remove a `sorry` that directly follows a finisher.

    Models frequently emit BOTH a real finisher and a trailing `sorry` for the
    same obligation, e.g.

        have f1: "..."
          using a1 by simp
          sorry            <-- illegal: the `have` is already closed by `by simp`

    The skeleton prompt's examples only ever model `sorry` at every leaf, which
    nudges the model toward appending `sorry` even when it also supplied a `by`.
    `_ensure_have_show_bodies` only *adds* missing bodies; it never removes this
    redundant `sorry`, so the malformed pair survives and aborts the proof before
    the fill/repair stages can run. This pass deletes the dangling `sorry` when the
    nearest preceding non-blank line already closed the goal.

    A line "closes the goal" if it is `done`, a bare `.`/`..`, a standalone
    `by <method>`, or ends in an inline ` by <method>` (e.g. `using a1 by simp`).
    We intentionally do NOT treat `proof`/`next`/`qed`/`case` as closers, so we
    never strip a `sorry` that is the legitimate body of a freshly opened goal.

    Additionally collapses *duplicate* sorries: a standalone `sorry` whose nearest
    preceding non-blank line already ends the obligation with `sorry` (either a
    standalone `sorry` or an inline `... sorry`, e.g. `have f4: "..." sorry`). The
    model sometimes emits both an inline and a trailing sorry for one `have`, which
    is malformed; we keep the first and drop the redundant follow-on.
    """
    lines = text.splitlines()
    out: List[str] = []
    # Index, in `out`, of the most recent non-blank line (for back-reference).
    last_nonblank = -1
    closer_by = re.compile(r"(?m)^\s*by\b")
    closer_done = re.compile(r"(?m)^\s*(?:done|\.\.?)\s*$")
    inline_by = re.compile(r"\s+by\s+\S")
    ends_in_sorry = re.compile(r"\bsorry\s*$")
    for L in lines:
        if SORRY_RE.search(L) and L.strip() == "sorry" and last_nonblank >= 0:
            prev = out[last_nonblank]
            # (a) redundant after a real finisher
            if closer_by.match(prev) or closer_done.match(prev) or inline_by.search(prev):
                continue
            # (b) duplicate sorry: the obligation is already terminated by a sorry
            #     on the previous non-blank line (standalone or inline).
            if ends_in_sorry.search(prev):
                continue
        out.append(L)
        if L.strip() != "":
            last_nonblank = len(out) - 1
    return "\n".join(out)

def _normalize_show_kinds(text: str) -> str:
    """
    Flip ONLY the '?thesis'/'?case' meta after 'show', preserving any prefix (e.g., 'then', 'from', 'with', 'using', 'finally').
    Policy (line-local, nesting-aware):
      • Inside branches of (co)induction → 'show ?case'
      • Inside branches of 'proof (cases …)' (incl. 'cases rule: …') → 'show ?thesis'
      • Outside any explicit case branch → default to 'show ?thesis'
    """
    lines = text.splitlines()
    # Track current proof mode for the *innermost* open proof
    stack: list[str] = []  # values: "induct" | "cases" | "plain"
    in_case_branch = False
    for i, L in enumerate(lines):
        m_open = _PROOF_OPEN_RE.match(L)
        if m_open:
            mode = (m_open.group(1) or "").strip()
            if not mode:
                stack.append("plain")
            elif _MODE_INDUCT_RE.match(mode):
                stack.append("induct")
            elif _MODE_CASES_RULE.match(mode) or _MODE_CASES_RE.match(mode):
                stack.append("cases")
            else:
                stack.append("plain")
            continue
        if _QED_LINE_RE.match(L):
            if stack:
                stack.pop()
            continue
        if CASE_START_RE.match(L):
            in_case_branch = True
            continue
        if NEXT_OR_QED_RE.match(L):
            in_case_branch = False
            continue

        def _repl(m: "re.Match[str]") -> str:
            current = stack[-1] if stack else "plain"
            want = "case" if (in_case_branch and current == "induct") else "thesis"
            # If already correct, return unchanged
            if m.group("meta") == want:
                return m.group(0)
            return f'{m.group("prefix")}?{want}'

        # Rewrite at most once per line; this keeps everything except the meta token intact.
        lines[i] = SHOW_META_AT_SHOW.sub(_repl, L, count=1)
    return "\n".join(lines)

def _ensure_have_show_bodies(text: str) -> str:
    """
    Ensure every 'have …' / 'show …' has a body, but *preserve* local continuations:
    scan forward across lines starting with using/from/with/then/also/moreover/finally,
    and only insert 'sorry' at the *end* of that block if no tactic/proof body occurs
    before the next statement/boundary (have/show/assume/case/next/qed).
    """
    lines = text.splitlines()
    i, n = 0, len(lines)
    out: List[str] = []
    while i < n:
        L = lines[i]
        out.append(L)
        if _HAVE_OR_SHOW.match(L) and not _INLINE_BY.search(L):
            j = i + 1
            # skip blank lines
            while j < n and lines[j].strip() == "":
                out.append(lines[j]); j += 1
            # consume a local continuation block
            saw_body = False
            k = j
            while k < n:
                Nk = lines[k]
                if _HAS_TACTIC_NEXT.match(Nk):
                    saw_body = True
                    break
                if _STMT_OR_BOUNDARY.match(Nk):
                    break
                if _CONTINUATION_HEAD.match(Nk) or Nk.strip() == "":
                    out.append(Nk); k += 1
                    continue
                # unknown line: stop the block here
                break
            if not saw_body:
                indent = L[:len(L) - len(L.lstrip(" "))]
                out.append(f"{indent}  sorry")
            i = k
            continue
        i += 1
    return "\n".join(out)

def _maybe_proof_dash(text: str) -> str:
    """
    If there is a bare 'proof' at top-level and calculational cues present, prefer 'proof -'.
    """
    if not BARE_PROOF_RE.search(text):
        return text
    if re.search(r"(?m)^\s*(have|also|moreover|ultimately|finally|hence|thus)\b", text):
        return BARE_PROOF_RE.sub("proof -", text, count=1)
    return text

def _sanitize_outline(text: str, goal: str, *, force_outline: bool) -> str:
    text = _ensure_lemma_header(text, goal)
    # Normalize ellipsis first (avoid Unicode / spaced form)
    text = _normalize_calculation_ellipsis(text)

    # Keep content from *this* lemma onwards
    goal_header = f'lemma "{goal}"'
    idx = text.find(goal_header)
    if idx >= 0:
        text = text[idx:]
    else:
        first_lemma = text.find("lemma ")
        if first_lemma >= 0:
            text = text[first_lemma:]

    # Ensure proof/qed skeleton exists
    if not PROOF_RE.search(text):
        text = text.rstrip() + "\nproof\n  sorry\nqed\n"
    if not QED_RE.search(text):
        text = text.rstrip() + "\nqed\n"

    # Force an outline (remove inline 'by' if requested by caller)
    if force_outline:
        lines = text.splitlines()
        for i, L in enumerate(lines):
            if " by " in L:
                lines[i] = BY_INLINE_RE.sub(" sorry", L)
        text = "\n".join(lines)
        if "sorry" not in text:
            m_qed = QED_RE.search(text)
            if m_qed:
                insert_at = m_qed.start()
                text = text[:insert_at] + "  sorry\n" + text[insert_at:]

    # Light Isar fixups (order matters)
    #  1) Flip only the meta after 'show', preserving 'then/using/from/with/finally' etc.
    #  2) Ensure every 'have/show' has a body; insert 'sorry' if missing to trigger fill/repair.
    #  3) Prefer 'proof -' when calculational cues are present.
    text = _normalize_show_kinds(text)
    text = _ensure_have_show_bodies(text)
    text = _maybe_proof_dash(text)

    # Trim to the first complete lemma..qed block to avoid trailing splices
    text = _crop_to_first_proof_block(text)

    if not text.endswith("\n"):
        text += "\n"
    return text

def _quick_sketch_score(isabelle, session_id: str, outline_text: str) -> int:
    try:
        thy = build_theory(outline_text.splitlines(), add_print_state=True, end_with="sorry")
        resps = run_theory(isabelle, session_id, thy)
        block = last_print_state_block(resps) or ""
        n = parse_subgoals(block)
        return int(n) if isinstance(n, int) else 9999
    except Exception:
        return 9999

def _state_block_for_goal(isabelle, session_id: str, goal: str) -> str:
    mini = f'lemma "{goal}"\nproof\n  sorry\nqed\n'
    try:
        thy = build_theory(mini.splitlines(), add_print_state=True, end_with="sorry")
        resps = run_theory(isabelle, session_id, thy)
        return last_print_state_block(resps) or ""
    except Exception:
        return ""

# --- Pattern detection / simple priors ---

_PAT_INDUCTION = re.compile(r"(?m)^\s*proof\s*\(induction\b", re.UNICODE)
_PAT_CASES     = re.compile(r"(?m)^\s*proof\s*\(cases\b", re.UNICODE)
_PAT_CASES_RULE= re.compile(r"(?m)^\s*proof\s*\(cases\s+rule:\s*([A-Za-z0-9_\.]+)", re.UNICODE)
_ID = r"[A-Za-z_][A-Za-z0-9_']*"

def _detect_pattern_key(outline: str) -> str:
    if _PAT_INDUCTION.search(outline):
        return "induction"
    m = _PAT_CASES_RULE.search(outline)
    if m:
        return f"cases_rule:{m.group(1)}"
    if _PAT_CASES.search(outline):
        return "cases"
    return "plain"

def _tokenize_goal(goal: str) -> set:
    toks = set(re.findall(_ID, goal))
    if "@" in goal: toks.add("@")
    if "⟹" in goal: toks.add("implies")
    return toks

def _load_priors(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Expect either {"rules":[...]} or a bare list of rules.
        if isinstance(data, dict) and isinstance(data.get("rules"), list):
            return data["rules"]
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []

def _pattern_penalty(goal: str, outline: str, rules: List[Dict[str, Any]]) -> float:
    """
    Lower is better. Simple heuristic + optional JSON rules.
    """
    key = _detect_pattern_key(outline)
    toks = _tokenize_goal(goal)
    pen = 0.0
    # Built-in gentle priors
    if ("@" in toks or "map" in toks) and key != "induction":
        pen += 0.4
    if ({"Suc", "0"} & toks) and key != "induction":
        pen += 0.3
    if ({"True", "False"} & toks) and not key.startswith("cases"):
        pen += 0.25
    if ({"Some", "None", "option"} & toks) and "cases_rule:option.exhaust" != key and key != "cases":
        pen += 0.25
    if ({"Inl", "Inr"} & toks) and "cases_rule:sum.exhaust" != key and key != "cases":
        pen += 0.25
    # Optional external rules
    for r in rules:
        cond = set(map(str, r.get("if_any_tokens", [])))
        prefer = set(map(str, r.get("prefer_patterns", [])))
        weight = float(r.get("weight", 0.3))
        if cond and (cond & toks) and key and prefer and key not in prefer:
            pen += weight
    return pen

def _hint_bonus_from_outline(outline: str, recommended: List[str]) -> int:
    if not recommended:
        return 0
    # Count how many recommended tokens appear in the outline text (rough proxy)
    text = outline
    c = 0
    for h in recommended[:10]:
        if h in text:
            c += 1
    return c

# -----------------------------------------------------------------------------
# NEW: Hint lexicon (micro-RAG) utilities
# -----------------------------------------------------------------------------

def _load_hintlex(path: Optional[str]) -> Dict[str, List[str]]:
    """
    Returns token -> [hint,...]. Accepts either {"token":[["hint",count],...]} or {"token":["hint",...]}.
    """
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}
    out: Dict[str, List[str]] = {}
    for tok, val in raw.items():
        if isinstance(val, list):
            if val and isinstance(val[0], list):
                out[tok] = [h for h, _c in val]
            else:
                out[tok] = [str(h) for h in val]
    return out

def _hints_from_hintlex(goal: str, hintlex: Dict[str, List[str]], top: int = 8) -> List[str]:
    toks = _tokenize_goal(goal)
    got: List[str] = []
    for t in toks:
        hs = hintlex.get(t)
        if not hs:
            continue
        for h in hs[:top]:
            got.append(h)
    # stable de-dup
    return list(dict.fromkeys(got))

# -----------------------------------------------------------------------------
# Outline generators
# -----------------------------------------------------------------------------

def propose_isar_skeleton(
    goal: str,
    model: Optional[str] = None,
    temp: float = 0.35,
    *,
    force_outline: bool = False,
    hints: Optional[List[str]] = None,
) -> Skeleton:
    # Inject tiny hint list when available (keeps default behavior if None/empty)
    prompt = SKELETON_PROMPT.format(goal=goal)
    if hints:
        prompt += "\nHINTS: Prefer using " + ", ".join(sorted(set(hints))) + " if applicable.\n"
    raw = _generate_simple(
        prompt=prompt,
        model=model or DEFAULT_MODEL,
        temperature=temp,
        timeout_s=OLLAMA_TIMEOUT_S,
    )
    cleaned = _sanitize_outline(raw, goal=goal, force_outline=force_outline)
    return Skeleton(text=cleaned, holes=find_sorry_spans(cleaned))

def propose_isar_skeletons(
    goal: str,
    *,
    model: Optional[str] = None,
    temps: Iterable[float] = (0.3, 0.5, 0.8),
    k: Optional[int] = None,
    force_outline: bool = False,
    hints: Optional[List[str]] = None,
) -> List[Skeleton]:
    seen, out = set(), []
    for t in temps:
        prompt = SKELETON_PROMPT.format(goal=goal)
        if hints:
            prompt += "\nHINTS: Prefer using " + ", ".join(sorted(set(hints))) + " if applicable.\n"
        raw = _generate_simple(
            prompt=prompt,
            model=model or DEFAULT_MODEL,
            temperature=float(t),
            timeout_s=OLLAMA_TIMEOUT_S,
        )
        sk = Skeleton(text=_sanitize_outline(raw, goal=goal, force_outline=force_outline),
                      holes=[])
        sk.holes = find_sorry_spans(sk.text)
        key = sk.text.strip()
        if key not in seen:
            seen.add(key)
            out.append(sk)
        if k is not None and len(out) >= int(k):
            break
    if not out:
        return [propose_isar_skeleton(goal, model=model, temp=0.3, force_outline=force_outline, hints=hints)]
    return out

# -----------------------------------------------------------------------------
# IMPROVEMENT: Safe theorem-shape-aware templates and structural safety checks
# -----------------------------------------------------------------------------

def _mk_skeleton(text: str) -> Skeleton:
    """Create a Skeleton object and ensure final newline consistency."""
    if not text.endswith("\n"):
        text += "\n"
    return Skeleton(text=text, holes=find_sorry_spans(text))


def _split_top_level_once(s: str, op: str) -> Optional[Tuple[str, str]]:
    """
    Split a string once on a top-level operator.

    This is intentionally lightweight. It tracks parentheses so we do not split
    inside nested expressions. It is not a full Isabelle parser, but it is safer
    than a plain string split.
    """
    depth = 0
    i = 0
    while i <= len(s) - len(op):
        ch = s[i]

        if ch in "([{":
            depth += 1
            i += 1
            continue

        if ch in ")]}":
            depth = max(0, depth - 1)
            i += 1
            continue

        if depth == 0 and s.startswith(op, i):
            left = s[:i].strip()
            right = s[i + len(op):].strip()
            if left and right:
                return left, right

        i += 1

    return None


def _strip_outer_parens(s: str) -> str:
    """Remove one pair of outer parentheses when they wrap the whole expression."""
    s = s.strip()
    if not (s.startswith("(") and s.endswith(")")):
        return s

    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and i != len(s) - 1:
                return s

    return s[1:-1].strip()


def _looks_like_set_expr(s: str) -> bool:
    """Very rough detector for set expressions."""
    return any(tok in s for tok in ["∩", "∪", "⊆", "`", "Pow", "set "])


def _choose_list_induction_var(goal: str) -> Optional[str]:
    """
    Pick a likely list induction variable.

    This intentionally prefers common dataset variable names first. It is only a
    skeleton choice; Isabelle will still verify/fill the proof later.
    """
    tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_']*\b", goal)

    for preferred in ("xs", "ys", "zs"):
        if preferred in tokens:
            return preferred

    # Fallback: choose a variable that appears near common list operators/functions.
    if any(tok in goal for tok in ["@", "rev", "map", "filter", "take", "drop", "length"]):
        for t in tokens:
            if t not in {"rev", "map", "filter", "take", "drop", "length", "set", "id"}:
                return t

    return None

def _safe_templates_for_goal(goal: str) -> List[Skeleton]:
    """
    Generate conservative theorem-shape-aware Isar skeletons.

    These are not meant to prove everything directly. Their job is to provide
    structurally safe proof outlines that the LLM candidates can compete against.
    """
    g = goal.strip()
    templates: List[Skeleton] = []

    # Iff / equivalence: prove both directions.
    # Avoid this generic iff template for list-recursion goals such as:
    #   (∀x∈set xs. P x) ⟷ filter P xs = xs
    # These usually need induction rather than a plain two-direction proof.
    split = _split_top_level_once(g, "⟷")
    list_recursion_goal = any(tok in g for tok in ["filter", "map", "rev", "@", "length"])

    if split and not list_recursion_goal:
        left, right = map(_strip_outer_parens, split)
        templates.append(_mk_skeleton(
            f'''lemma "{g}"
    proof
      assume H: "{left}"
      show "{right}"
        using H
        sorry
    next
      assume H: "{right}"
      show "{left}"
        using H
        sorry
    qed
    '''))

    # Object-level implication: assume premise, show conclusion.
    split = _split_top_level_once(g, "⟶")
    if split:
        left, right = map(_strip_outer_parens, split)
        templates.append(_mk_skeleton(
f'''lemma "{g}"
proof
  assume H: "{left}"
  show "{right}"
    using H
    sorry
qed
'''))

    # Meta-level implications: introduce assumptions one by one, then show final claim.
    if "⟹" in g:
        parts = [p.strip() for p in g.split("⟹") if p.strip()]
        if len(parts) >= 2:
            assumptions = parts[:-1]
            conclusion = parts[-1]
            lines = [f'lemma "{g}"', "proof -"]
            for i, a in enumerate(assumptions, start=1):
                lines.append(f'  assume H{i}: "{a}"')
            using = " ".join(f"H{i}" for i in range(1, len(assumptions) + 1))
            lines.append(f'  show "{conclusion}"')
            lines.append(f'    using {using}')
            lines.append("    sorry")
            lines.append("qed")
            templates.append(_mk_skeleton("\n".join(lines)))

    # Set equality: prove mutual inclusion, but only when the WHOLE goal is a set equality.
    # Do not trigger this inside iff/implication goals such as:
    #   (∀x∈set xs. P x) ⟷ filter P xs = xs
    # because the "=" belongs to one side of the iff, not to the whole theorem shape.
    eq_split = _split_top_level_once(g, "=")
    has_outer_iff = _split_top_level_once(g, "⟷") is not None
    has_outer_obj_imp = _split_top_level_once(g, "⟶") is not None

    if eq_split and not has_outer_iff and not has_outer_obj_imp and "card" not in g:
        left, right = map(_strip_outer_parens, eq_split)

        # Be conservative: require genuine set operators, not just "set xs" inside
        # a quantifier or list theorem.
        set_equality_cues = ["∩", "∪", "⊆", "⊂", "Pow", "`"]

        if any(tok in left or tok in right for tok in set_equality_cues):
            templates.append(_mk_skeleton(
                f'''lemma "{g}"
    proof
      show "{left} ⊆ {right}"
        sorry
    next
      show "{right} ⊆ {left}"
        sorry
    qed
    '''))

    # Common list-recursion goals: use induction rather than invented have-chains.
    ind_var = _choose_list_induction_var(g)
    if ind_var:
        templates.append(_mk_skeleton(
f'''lemma "{g}"
proof (induction {ind_var})
  case Nil
  show ?case
    sorry
next
  case (Cons x {ind_var})
  show ?case
    sorry
qed
'''))

    # Cardinality of injective image: use assumptions and leave the key library step
    # as the hole. Do not try to prove A = f ` A.
    if "card" in g and "`" in g and ("inj_on" in g or "inj " in g):
        templates.append(_mk_skeleton(
f'''lemma "{g}"
proof -
  show ?thesis
    sorry
qed
'''))

    return templates

def _direct_templates_for_goal(goal: str) -> List[Skeleton]:
    """
    Small complete Isar proofs generated locally.

    These are not scored using sketch heuristics. They are tried first and only
    accepted if Isabelle fully verifies them.
    """
    g = goal.strip()

    templates: List[Skeleton] = []

    # General-purpose direct methods.
    methods = ["auto", "blast", "fastforce", "force", "simp"]

    for method in methods:
        templates.append(_mk_skeleton(
f'''lemma "{g}"
proof -
  show ?thesis by {method}
qed
'''
        ))

    # Library-fact-aware direct template for injective image cardinality.
    # Plain simp/auto is too weak here, but Isabelle can solve this theorem
    # once the assumptions are introduced and card_image is available.
    if "card" in g and "`" in g and "inj_on" in g and "⟹" in g:
        parts = [p.strip() for p in g.split("⟹") if p.strip()]

        if len(parts) >= 2:
            assumptions = parts[:-1]
            conclusion = parts[-1]

            lines = [f'lemma "{g}"', "proof -"]

            for i, asm in enumerate(assumptions, start=1):
                lines.append(f'  assume H{i}: "{asm}"')

            using = " ".join(f"H{i}" for i in range(1, len(assumptions) + 1))

            lines.append(f'  show "{conclusion}"')
            lines.append(f'    using {using}')
            lines.append("    by (simp add: card_image)")
            lines.append("qed")

            templates.append(_mk_skeleton("\n".join(lines)))

    return templates

def _verifies_complete_proof(isabelle, session_id: str, proof_text: str) -> bool:
    """
    Fully verify a complete no-sorry proof candidate.

    This is used only for small direct templates, so it avoids the earlier
    slowdown caused by verifying every long LLM-generated proof.
    """
    if "sorry" in proof_text:
        return False

    try:
        thy = build_theory(proof_text.splitlines(), add_print_state=False, end_with=None)
        ok, _ = finished_ok(run_theory(isabelle, session_id, thy))
        return bool(ok)
    except Exception:
        return False

def _full_verification_penalty(isabelle, session_id: str, outline_text: str) -> float:
    """
    IMPROVEMENT: If a candidate has no sorry holes, it is claiming to be a complete proof.
    In that case, verify it fully. A complete but invalid proof should not beat
    a partial but structurally safe outline.

    Returns 0.0 for verified complete proofs or outlines with holes.
    Returns a large penalty for invalid complete proofs.
    """
    if "sorry" in outline_text:
        return 0.0

    try:
        thy = build_theory(outline_text.splitlines(), add_print_state=False, end_with=None)
        ok, _ = finished_ok(run_theory(isabelle, session_id, thy))
        return 0.0 if ok else 500.0
    except Exception:
        return 500.0

def _normalise_for_comparison(s: str) -> str:
    """Normalise text for rough duplicate/restatement checks."""
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    s = s.strip('"')
    return s


def _safety_penalty(goal: str, outline: str) -> float:
    """
    Penalise structurally risky skeletons observed in our diagnostic tests.

    Lower is better. This does not reject candidates outright; it makes unsafe
    LLM outputs less likely to beat safe templates.
    """
    g_norm = _normalise_for_comparison(goal)
    pen = 0.0
    lines = outline.splitlines()

    # 1. Penalise have-statements that restate the original theorem.
    have_pat = re.compile(r'^\s*have\s+\w*:?[\s]*"([^"]+)"')
    for line in lines:
        m = have_pat.match(line)
        if not m:
            continue
        claim_norm = _normalise_for_comparison(m.group(1))
        if claim_norm == g_norm:
            pen += 8.0
        elif g_norm and (claim_norm in g_norm or g_norm in claim_norm):
            pen += 4.0

    # 2. Wrong disjunction/case pattern seen in testing.
    if re.search(r"proof\s*\(cases\s+rule:\s*disjE\)", outline):
        if re.search(r"(?m)^\s*case\s+True\b", outline) or re.search(r"(?m)^\s*case\s+False\b", outline):
            pen += 8.0

    # 3. Induction cases without an induction proof header.
    if (re.search(r"(?m)^\s*case\s+Nil\b", outline) or re.search(r"(?m)^\s*case\s+\(Cons\b", outline)):
        if not re.search(r"proof\s*\(induction\b", outline):
            pen += 7.0

    # 4. A sorry should not be followed by more tactic/proof commands in the same local block.
    for i, line in enumerate(lines[:-1]):
        if line.strip() == "sorry":
            nxt = lines[i + 1].strip()
            if nxt.startswith(("unfolding", "apply", "case ", "then show", "show ")):
                pen += 6.0

    # 5. Avoid type-unsafe cardinality/image fake set equality or subset reasoning.
    if "card" in goal and "`" in goal:
        if re.search(r"⊆\s*f\s*`", outline) or re.search(r"f\s*`\s*\w+\s*⊆", outline):
            pen += 8.0
        if re.search(r'"\s*\w+\s*=\s*f\s*`', outline) or re.search(r'"\s*f\s*`\s*\w+\s*=\s*\w+', outline):
            pen += 8.0

    # 6. Penalise too many invented have-claims, especially for simple goals.
    have_count = len(re.findall(r"(?m)^\s*have\b", outline))
    if have_count > 2:
        pen += float(have_count - 2)

    # 7. Iff goals should not be proved using a plain proof-block with sequential assumptions.
    # A proper iff skeleton should use:
    # proof
    #   assume ...
    #   show ...
    # next
    #   assume ...
    #   show ...
    # qed
    if "⟷" in goal:
        if re.search(r"(?m)^\s*proof\s*-\s*$", outline):
            if len(re.findall(r"(?m)^\s*assume\b", outline)) >= 2:
                pen += 10.0

    return pen

def _lib_templates_for_goal(goal: str) -> List[Skeleton]:
    toks = _tokenize_goal(goal)
    lib: List[str] = []
    if ("@" in toks or "map" in toks) and "xs" in toks:
        lib.append(
f'''lemma "{goal}"
proof (induction xs)
  case Nil
  show ?case by simp
next
  case (Cons x xs)
  show ?case
    sorry
qed
''')
    if ({"Suc","0"} & toks) and "n" in toks:
        lib.append(
f'''lemma "{goal}"
proof (induction n)
  case 0
  show ?case by simp
next
  case (Suc n)
  show ?case
    sorry
qed
''')
    if ({"True","False"} & toks) and "b" in toks:
        lib.append(
f'''lemma "{goal}"
proof (cases b)
  case True
  show ?thesis
    sorry
next
  case False
  show ?thesis
    sorry
qed
''')
    lib.append(
f'''lemma "{goal}"
proof -
  have f1: "(* fill a useful intermediate statement *)"
    sorry
  have f2: "(* another useful intermediate *)"
    using f1
    sorry
  show ?thesis
    using f1 f2
    sorry
qed
''')
    return [Skeleton(text=s if s.endswith("\n") else s+"\n", holes=find_sorry_spans(s)) for s in lib]

def propose_isar_skeleton_diverse_best(
    goal: str,
    *,
    isabelle,             # required for sketch check
    session_id: str,
    model: Optional[str] = None,
    temps: Iterable[float] = (0.35, 0.55, 0.85),
    k: int = 3,
    force_outline: bool = False,
    # knobs
    priors_path: Optional[str] = None,
    context_hints: bool = False,
    lib_templates: bool = False,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.2,
    # NEW: hint lexicon
    hintlex_path: Optional[str] = None,
    hintlex_top: int = 8,
    trace: bool = False,
) -> Tuple[Skeleton, Dict[str, Any]]:
    """
    Generate K outlines, optionally inject context & hintlex hints, run one-shot sketch checks,
    and return the best using composite score:
      score = alpha * subgoals + beta * pattern_penalty - gamma * hint_bonus
    """
    # Optional context hints from Isabelle state + hint lexicon
    rec_hints: List[str] = []
    if context_hints:
        state_block = _state_block_for_goal(isabelle, session_id, goal)
        rec_hints += _facts_from_state(state_block)[:8]
    hintlex = _load_hintlex(hintlex_path)
    if hintlex:
        rec_hints += _hints_from_hintlex(goal, hintlex, top=hintlex_top)
    rec_hints = list(dict.fromkeys(rec_hints))[:12]  # stable de-dup + cap

    # -------------------------------------------------------------------------
    # Stage 1: verified direct Isar proofs
    # -------------------------------------------------------------------------
    # Try small locally generated proofs first. These are cheap and safe because
    # we accept them only if Isabelle fully verifies them. This prevents the
    # selector from choosing a long unverified LLM proof when a simple method
    # like auto/blast/simp already works.
    if not force_outline:
        direct_templates = _direct_templates_for_goal(goal)

        if trace:
            print()
            print(f"[skeleton] Stage 1: trying {len(direct_templates)} verified direct template(s)")

        for i, sk in enumerate(direct_templates, start=1):
            ok = _verifies_complete_proof(isabelle, session_id, sk.text)

            if trace:
                first_proof_line = next(
                    (ln.strip() for ln in sk.text.splitlines() if ln.strip().startswith("by ") or " by " in ln),
                    "(multi-line proof)"
                )
                print(f"[skeleton] direct candidate {i}: {first_proof_line} -> {'PASS' if ok else 'FAIL'}")

            if ok:
                diag = {
                    "selected_source": "verified_direct",
                    "direct_candidates": len(direct_templates),
                    "selected_text": sk.text,
                }
                return sk, diag
    if trace:
        print()
    # Candidate sources are tracked for debugging/analysis.
    # This makes it clear whether the selected outline came from:
    # - a local safe theorem-shape template,
    # - the older optional library templates,
    # - or the LLM.
    safe_templates = _safe_templates_for_goal(goal)

    llm_candidates = propose_isar_skeletons(
        goal, model=model, temps=temps, k=k,
        force_outline=True, hints=rec_hints
    )

    raw_pairs: List[Tuple[str, Skeleton]] = []

    if lib_templates:
        raw_pairs.extend(("lib_template", sk) for sk in _lib_templates_for_goal(goal))

    raw_pairs.extend(("safe_template", sk) for sk in safe_templates)
    raw_pairs.extend(("llm", sk) for sk in llm_candidates)

    if trace:
        print("[skeleton] Candidate source counts:")
        print(f"  safe templates: {len(safe_templates)}")
        print(f"  llm candidates: {len(llm_candidates)}")
        print(f"  lib templates:  {len(_lib_templates_for_goal(goal)) if lib_templates else 0}")

    # Stable de-duplication after all candidate sources are combined.
    # If two sources produce identical text, keep the earlier source.
    seen_texts = set()
    cands: List[Skeleton] = []
    cand_sources: List[str] = []

    for source, sk in raw_pairs:
        key = sk.text.strip()
        if key and key not in seen_texts:
            seen_texts.add(key)
            cands.append(sk)
            cand_sources.append(source)

    if trace:
        print()
        print(f"[skeleton] Unique candidates after de-duplication: {len(cands)}")
        for i, (source, sk) in enumerate(zip(cand_sources, cands), start=1):
            preview = " ".join(sk.text.strip().splitlines()[:2])
            print(
                f"  candidate {i}: source={source}, "
                f"holes={len(sk.holes)}, chars={len(sk.text)}, preview={preview[:120]}"
            )

    # Load optional priors/rules
    rules = _load_priors(priors_path)

    scored: List[Tuple[float, int, int]] = []  # (score, n_subgoals, idx)

    for i, sk in enumerate(cands):
        n = _quick_sketch_score(isabelle, session_id, sk.text)
        pat_pen = _pattern_penalty(goal, sk.text, rules)
        safe_pen = _safety_penalty(goal, sk.text)
        hint_b = _hint_bonus_from_outline(sk.text, rec_hints)

        n_for_score = n
        if n == 9999 and cand_sources[i] == "safe_template":
            n_for_score = 20

        score = (
                alpha * float(n_for_score)
                + beta * float(pat_pen)
                + 1.0 * float(safe_pen)
                - gamma * float(hint_b)
        )

        scored.append((score, n, i))

        if trace:
            print()
            print(
                f"[skeleton] score candidate {i + 1}: "
                f"source={cand_sources[i]}, "
                f"score={score:.3f}, "
                f"subgoals={n}, score_subgoals={n_for_score}, "
                f"pattern_penalty={pat_pen:.3f}, "
                f"safety_penalty={safe_pen:.3f}, "
                f"hint_bonus={hint_b}, "
                f"holes={len(sk.holes)}"
            )

    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    best_idx = scored[0][2]
    best = cands[best_idx]

    if trace:
        print(
            f"[skeleton] selected candidate {best_idx + 1}: "
            f"source={cand_sources[best_idx]}, "
            f"score={scored[0][0]:.3f}, "
            f"subgoals={scored[0][1]}, "
            f"holes={len(best.holes)}"
        )

    diag = {
        "scores": scored,
        "num_candidates": len(cands),
        "used_hints": rec_hints[:12],
        "priors_rules": len(rules),
        "alpha_beta_gamma": (alpha, beta, gamma),
        "safe_templates": len(safe_templates),
        "selected_source": cand_sources[best_idx],
    }

    return best, diag
