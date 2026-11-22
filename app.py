import re
import json
import os
import time
import difflib
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from sentence_transformers import SentenceTransformer
from groq import Groq

# =============================================================================
# CONFIG & SAFETY CHECKS
# =============================================================================
st.set_page_config(page_title="B2B Lead Scoring RAG", page_icon="🎯", layout="wide")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 **Missing API Key:** add GROQ_API_KEY to Streamlit secrets.")
    st.stop()

if not os.path.exists("./embeddings/chunks.json"):
    st.error("🚨 **Missing Files:** could not find ./embeddings/chunks.json. Please upload your chunks.json file.")
    st.stop()

MODEL_PREFERENCES = [
    st.secrets.get("LLM_MODEL") if st.secrets.get("LLM_MODEL") else None,
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
]
MODEL_PREFERENCES = [m for m in MODEL_PREFERENCES if m]
EMBED_MODEL = st.secrets.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

TOP_K = 5
MIN_RAW_SIM = 0.10            # only used when the query has no domain keywords at all
HISTORY_TURNS = 2             # number of user/assistant exchanges sent to the LLM
HISTORY_MSG_CHARS = 600       # per-message text limit (chart memory lines are added after this cut)
MAX_TOKENS_DEFAULT = 1200     # gpt-oss reasoning tokens count against this budget
MAX_TOKENS_VISUAL = 2500
MAX_TOKENS_EXHAUSTIVE = 4000
INTENT_BOOST_MULT = 2.0
INTENT_BOOST_ADD = 0.20
REPORT_PENALTY = 0.15
MAX_RETRIES = 2               # retries on HTTP 429
MAX_WAIT_SECONDS = 120        # upper bound for a single wait between retries

REFUSAL = "The provided model artifacts do not contain this information."
QUOTE_CHARS = "\"'\u201c\u201d\u2018\u2019"

SELF_CHECK_TRIGGERS = (
    "confirm", "prove", "proves", "right?", "correct?", "is that true",
    "definitely", "always", "never", "guarantee", "means that", "so we should",
)

# Matched against a punctuation-stripped, lower-cased question (see _is_meta_followup)
META_PATTERN = re.compile(
    r"\b(why|explain|how come|what do you mean|whats missing|what is missing|huh|which)\b"
)

HYPOTHETICAL_TRIGGERS = ("suppose", "hypothetical", "if an account", "what if", "assume", "imagine")

EXHAUSTIVE_TRIGGERS = (
    "all plots", "all charts", "every chart", "every plot",
    "in one go", "maximum charts", "generate everything",
)

# =============================================================================
# BACKEND + MODEL RESOLUTION
# =============================================================================
@st.cache_resource
def load_backend():
    embedder = SentenceTransformer(EMBED_MODEL)
    with open("./embeddings/chunks.json") as f:
        chunks = json.load(f)
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    groq_client = Groq(api_key=GROQ_API_KEY)
    return embedder, embeddings, chunks, groq_client

@st.cache_resource
def resolve_llm_model(_client, preferences: tuple):
    try:
        available = {m.id for m in _client.models.list().data}
    except Exception as e:
        return None, [], f"Could not list models: {e}"
    for m in preferences:
        if m in available:
            return m, sorted(available), None
    return None, sorted(available), "None of the preferred models are available"

embedder, EMBEDDINGS, CHUNKS, groq_client = load_backend()
EMBEDDINGS_NORM = EMBEDDINGS / (np.linalg.norm(EMBEDDINGS, axis=1, keepdims=True) + 1e-9)

LLM_MODEL, AVAILABLE_MODELS, MODEL_ERROR = resolve_llm_model(groq_client, tuple(MODEL_PREFERENCES))

if LLM_MODEL is None:
    st.error(f"🚨 **No usable LLM model found.** {MODEL_ERROR}")
    st.stop()

# =============================================================================
# SYSTEM PROMPT
# =============================================================================
SYSTEM_PROMPT = "\n".join([
    "You are a senior RevOps (Revenue Operations) consultant for a B2B sales team.",
    "",
    "STYLE: 2-4 sentences, direct answer first. Never refuse charts — generate JSON.",
    "",
    "CRITICAL EXCEPTIONS TO GROUNDING:",
    "1. META-INSTRUCTION: If the CONTEXT begins with 'META-INSTRUCTION:', you MUST explain what specific data is missing from the artifacts. Do NOT reply with the standard refusal message and do NOT add citations. Explain *why* the previous question could not be answered, and say what the artifacts DO cover.",
    "2. CALCULATIONS: If the user provides hypothetical inputs (e.g., 'If an account has X and Y...') and asks you to calculate a score based on the rules in CONTEXT, YOU MUST DO THE MATH. Begin by stating that the inputs are hypothetical, then show the formula and cite the calculation rules.",
    "3. GREETINGS: If the user just says 'hello' or 'hi', reply politely. Do NOT add a citation to a greeting.",
    "",
    "GROUNDING: For all other factual questions, use ONLY facts in CONTEXT. Do NOT add reasons, rationale, or business explanations that are not written in CONTEXT. If a factual question is not answered by CONTEXT: reply exactly '" + REFUSAL + "'",
    "",
    "PUSHBACK:",
    "1. Do NOT confirm false premises. If a user claims 'Champions always convert', correct them: High fit + engagement correlates with likelihood, but doesn't guarantee conversion.",
    "2. If the user states a score or threshold that contradicts CONTEXT, CORRECT it: 'That figure differs from the artifacts; the correct value is X.'",
    "",
    "TIER LOGIC: Priority MQL requires Score >=70, Premium >=$1M, Engagement >=15, and an active signal. MQL is Score >=50.",
    "ARCHETYPES: Champion (High Fit/Eng/Multi), Achiever (High Fit/Eng/Single), Watcher (High Fit/Mod Eng), Dormant Giant (High Fit/Low Eng), Fader (Low Fit/Mod Eng), Lurker (Low Fit/Low Eng). If CONTEXT lists any other archetype names, use the names exactly as CONTEXT gives them and never invent new ones.",
    "",
    "DERIVED FIGURES: Never estimate or round a total. If a total is not written in CONTEXT, add up the exact counts listed in CONTEXT and label the result '(calculated)', naming the inputs. If a line starting with 'CALCULATION' appears in CONTEXT, use that value and do not recompute it.",
    "COMPOSITE SCORE: it is always the weighted formula in CONTEXT (never a plain sum of Fit and Engagement).",
    "",
    "CHART DATA: Do NOT output an empty chart JSON (e.g., \"data\": []). If the CONTEXT lacks the data to fill a specific chart, DO NOT output that chart block. Just state which data is missing.",
    "CHARTS: If asked for 'all plots', generate the three charts listed in the 'Available Charts' chunk (Tier Distribution, SQL Count by Tier, Archetype Distribution).",
    "",
    "MEMORY: assistant turns include '(Chart N: title — label=value)'. If user references a chart, use history.",
    "CHART REFS: in prose use 'Chart 1' — NEVER square brackets.",
    "",
    "CHART FORMAT: wrap JSON in ```chart ... ```. ALL of \"type\", \"title\" and \"data\" are REQUIRED. Bar/line/pie: {\"type\": \"bar\", \"title\": \"Tier Distribution\", \"data\": [{\"label\": str, \"value\": number}]}.",
    "",
    "FILES: cannot produce files. Say 'I can't generate files, but here is a slide-ready content outline:'",
    "",
    "CITATIONS: after each factual sentence add [Chunk Title]. Use only titles that appear in CONTEXT."
])

SELF_CHECK_PROMPT = "\n".join([
    "You are auditing a proposed answer for statistical integrity.",
    "Check for: 1. Confirming false premises. 2. Missing citations. 3. Fabricated numbers or unsupported reasoning.",
    "DO NOT flag META-EXPLANATIONS or HYPOTHETICAL MATH as hallucinations.",
    "If any failure: REWRITE the answer to fix it. If no failures: return the answer text exactly as given. Return ONLY the answer text."
])

USER_TEMPLATE = "CONTEXT CHUNKS:\n{context}\n\nUSER QUESTION: {question}\n\nGrounded answer with [Title] citations (or Chart JSON if requested):"

# =============================================================================
# TEXT HELPERS
# =============================================================================
def _clean(s: str) -> str:
    """Lower-case and strip everything except letters, digits and spaces."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", (s or "").lower())).strip()

GREETING_RE = re.compile(r"(h+e+l+o+|h+i+|h+e+y+|yo|good (morning|afternoon|evening))( there)?")
THANKS_RE = re.compile(r"(thanks|thank you|thx|ty)( a lot| so much)?")
FAREWELL_RE = re.compile(r"(bye+|goodbye|good bye|see you|see ya|cya|good night)( for now)?")
LEADIN_RE = re.compile(r"i (have|got) (a|one|another) (quick )?(question|query)")

_FUZZY_WORDS = {"hello": "greet", "hiya": "greet", "thanks": "thanks", "goodbye": "bye"}

def _fuzzy_kind(c: str):
    """Single-word typos like 'helo' or 'hellw'. Needs >= 4 letters so real words are not caught by accident."""
    if len(c.split()) != 1 or not (4 <= len(c) <= 9):
        return None
    m = difflib.get_close_matches(c, list(_FUZZY_WORDS), n=1, cutoff=0.75)
    return _FUZZY_WORDS[m[0]] if m else None

def _smalltalk_reply(q: str):
    """Canned replies for greetings, thanks, goodbyes and 'I have a question' lead-ins (no retrieval, no LLM call)."""
    c = _clean(q)
    kind = _fuzzy_kind(c)
    if GREETING_RE.fullmatch(c) or kind == "greet":
        return "Hello! How can I help you with your lead scoring model today?"
    if THANKS_RE.fullmatch(c) or kind == "thanks":
        return "You're welcome!"
    if FAREWELL_RE.fullmatch(c) or kind == "bye":
        return "Goodbye! Come back any time you have questions about your lead scoring model."
    if LEADIN_RE.fullmatch(c):
        return "Sure, go ahead and ask your question."
    return None

def _is_hypothetical(q: str) -> bool:
    ql = q.lower()
    return any(t in ql for t in HYPOTHETICAL_TRIGGERS) or ("calculate" in ql and bool(re.search(r"\d", ql)))

def _is_meta_followup(q: str) -> bool:
    c = _clean(q)
    return 0 < len(c.split()) <= 8 and bool(META_PATTERN.search(c))

def _parse_hypothetical(q: str):
    """For 'Suppose an account has Fit 50, Engagement 30...' questions, compute the composite in code.
    The weights below MUST match the formula in chunks.json (0.4 fit, 0.6 engagement, 0.8 broker bonus)."""
    if not _is_hypothetical(q):
        return None
    ql = q.lower()
    num = r"(\d+(?:\.\d+)?)"
    fit = re.search(r"\bfit(?: score)?\s*(?:of|is|=|:)?\s*" + num, ql)
    eng = re.search(r"\bengagement(?: score)?\s*(?:of|is|=|:)?\s*" + num, ql)
    if not (fit and eng):
        return None
    bonus = re.search(r"broker bonus\s*(?:of|is|=|:)?\s*" + num, ql)
    f, e = float(fit.group(1)), float(eng.group(1))
    b = float(bonus.group(1)) if bonus else 0.0
    return f, e, b, round(0.4 * f + 0.6 * e + 0.8 * b, 2)

def _fmt_points(chart: dict, limit: int = 8) -> str:
    return ", ".join(f"{d.get('label')}={d.get('value')}" for d in chart.get("data", [])[:limit])

# =============================================================================
# INTENT & RETRIEVAL
# =============================================================================
INTENT_KEYWORDS = {
    "caveats": ["limitation", "caveat", "risk", "guarantee", "convert"],
    "recommendations": ["priority", "recommend", "mql", "sql", "tier", "plot", "chart"],
    "segments": ["segment", "archetype", "champion", "achiever", "watcher", "lurker", "plot", "chart"],
    "methodology": ["method", "fit score", "engagement", "weight", "calculate"],
    "model_summary": ["score", "total", "average", "dataset", "rows", "plot", "chart", "distribution"],
}

ANAPHORA_TOKENS = (
    "previous", "that chart", "this chart", "the chart", "above", "last one",
    "slice", "which bar", "the bar", "that bar", "this bar",
)

def detect_intents(query: str) -> set:
    q = query.lower()
    return {t for t, kws in INTENT_KEYWORDS.items() if any(kw in q for kw in kws)}

def retrieve(query: str, top_k: int = TOP_K):
    q_emb = embedder.encode([query], convert_to_numpy=True)[0]
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    raw_sims = EMBEDDINGS_NORM @ q_emb
    raw_max = float(np.max(raw_sims))

    hints = detect_intents(query)
    effective_k = max(top_k, min(len(hints) + 2, 8))

    is_chart_query = any(kw in query.lower() for kw in ["chart", "plot", "graph", "all plots"])

    sims = raw_sims.copy()
    for i, c in enumerate(CHUNKS):
        s = float(raw_sims[i])
        if c["type"] in hints:
            s = s * INTENT_BOOST_MULT + INTENT_BOOST_ADD
        elif c["type"] == "report_page":
            if not is_chart_query:
                s = s - REPORT_PENALTY
            else:
                s = s + 0.5  # strong boost for report pages when asking for charts
        sims[i] = s

    top_idx = list(np.argsort(sims)[::-1][:effective_k])
    results = [
        {
            "title": CHUNKS[i]["title"],
            "type": CHUNKS[i]["type"],
            "text": CHUNKS[i]["text"],
            "score": float(sims[i]),
            "raw_score": float(raw_sims[i]),
        }
        for i in top_idx
    ]
    return results, raw_max

def _get_chart_back(question: str) -> tuple:
    q = question.lower()
    explicit = re.findall(r"chart\s+(\d+)", q)
    if explicit:
        return 0, [int(n) for n in explicit]
    back = 0
    for phrase, n in [("previous to previous", 2), ("previous", 1), ("last chart", 1), ("that chart", 1)]:
        if phrase in q and n > back:
            back = n
    return back, []

def _chart_hint(all_charts: list, question: str) -> str:
    """Only adds chart memory when the question actually refers to an earlier chart."""
    if not all_charts:
        return ""
    back, refs = _get_chart_back(question)
    q = question.lower()
    if not (refs or back or any(t in q for t in ANAPHORA_TOKENS)):
        return ""

    by_num = {c.get("number"): c for c in all_charts if c.get("number") is not None}

    if refs:
        hints = []
        for n in refs:
            c = by_num.get(n)
            if c:
                hints.append(f"Chart {n} ('{c.get('title')}'): {_fmt_points(c, 10)}")
        return "Referring to | " + " | ".join(hints) if hints else ""

    if 1 <= back <= len(all_charts):
        target = all_charts[-back]
    else:
        target = all_charts[-1]
    return f"Chart {target.get('number', '?')} ('{target.get('title')}'): {_fmt_points(target)}."

def _last_was_refusal(history: list) -> bool:
    for m in reversed(history):
        if m.get("role") == "assistant":
            return "not contain this information" in (m.get("content") or "").lower()
    return False

def build_search_query(question: str, history: list, all_charts: list) -> str:
    if not history:
        return question
    if _last_was_refusal(history) and _is_meta_followup(question):
        return ""  # signals META mode

    back, explicit_refs = _get_chart_back(question)
    q_lower = question.lower().strip()
    if len(question.split()) < 4 or any(t in q_lower for t in ANAPHORA_TOKENS) or explicit_refs:
        recent = " ".join(
            m.get("content", "")
            for m in history[-6:]
            if m.get("role") == "user" and not _smalltalk_reply(m.get("content", ""))
        ).strip()
        return (_chart_hint(all_charts, question) + " " + recent + " " + question).strip()
    return question

def sanitize_context(text: str) -> str:
    return text.replace("|", "\u00a6")

def normalize_citations(text: str) -> str:
    text = re.sub(r"\(Source:\s*([^)]+)\)", r"[\1]", text)
    text = re.sub(r"\u3010([^\u3011]*)\u3011", r"[\1]", text)   # 【Title】 -> [Title]
    text = re.sub(r"\s*\[CALCULATION\]", "", text)               # the code-computed line is not a source
    return text.strip()

# =============================================================================
# VERIFICATION
# =============================================================================
NUMTOK = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"

# Strict pattern: big / precise numbers anywhere in the answer
NUM_PATTERN = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+\.\d{2,}\b|\b\d{5,}\b")

# Loose patterns: small numbers, but only when they sit next to a scoring word, a unit or a percent sign
LOOSE_PATTERNS = [
    re.compile(r"(?<![\d,.])" + NUMTOK + r"\s*%"),
    re.compile(r"(?<![\d,.])" + NUMTOK + r"\s*(?:accounts?|SQLs?|MQLs?|points?)\b", re.I),
    re.compile(r"\b(?:score|engagement|bonus|threshold)\b[^.\d\n]{0,20}?" + NUMTOK, re.I),
]

def _extract_nums(text: str) -> set:
    out = set()
    for n in re.findall(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+", text):
        try:
            out.add(round(float(n.replace(",", "")), 4))
        except ValueError:
            pass
    return out

def _with_scaled(nums: set) -> set:
    """Add thousand/million-scaled variants so '$1M' matches '1,000,000'."""
    extra = set()
    for n in nums:
        if n >= 1e6:
            extra.add(round(n / 1e6, 4))
        if n >= 1e3:
            extra.add(round(n / 1e3, 4))
    return nums | extra

def _decimals(raw: str) -> int:
    return len(raw.split(".")[1]) if "." in raw else 0

def _in_context(raw: str, ctx_nums: set) -> bool:
    try:
        v = float(raw.replace(",", ""))
    except ValueError:
        return True
    d = _decimals(raw)
    return any(round(c, d) == round(v, d) for c in ctx_nums)

def _derived_pct_ok(raw: str, ctx_nums: set) -> bool:
    """A percentage is accepted if it equals the ratio of two numbers found in the context."""
    try:
        pct = float(raw.replace(",", ""))
    except ValueError:
        return True
    tol = 0.5 * (10 ** -_decimals(raw)) + 1e-9
    nums = [n for n in sorted(ctx_nums) if n > 0][:200]
    for a in nums:
        for b in nums:
            if b >= a and abs(a / b * 100 - pct) <= tol:
                return True
    return False

def verify_numbers(answer: str, retrieved: list, question: str = "") -> list:
    ctx_nums = set()
    for c in retrieved:
        ctx_nums |= _extract_nums(c["text"])
        if c.get("total") is not None:  # e.g. the sum of a chart's values (a legitimate derived total)
            ctx_nums.add(round(float(c["total"]), 4))
    ctx_nums |= _extract_nums(question)  # numbers supplied by the user are not hallucinations
    ctx_nums = _with_scaled(ctx_nums)

    unverified = []

    for m in NUM_PATTERN.finditer(answer):
        raw = m.group(0)
        if not _in_context(raw, ctx_nums):
            unverified.append(raw)

    if not _is_hypothetical(question):  # hypothetical math produces new numbers on purpose
        for pat in LOOSE_PATTERNS:
            for m in pat.finditer(answer):
                raw = m.group(1)
                if _in_context(raw, ctx_nums):
                    continue
                is_pct = answer[m.end(1):m.end(1) + 3].lstrip().startswith("%")
                if is_pct and _derived_pct_ok(raw, ctx_nums):
                    continue
                unverified.append(raw)

    return unverified

def _norm_title(t: str) -> str:
    return re.sub(r"[\s—–\-_]+", " ", re.sub(r"\s*\([^)]*\)", "", t)).strip().lower()

def verify_citations(answer: str, retrieved: list) -> list:
    titles = {_norm_title(c["title"]) for c in retrieved}
    bad = []
    for cite in re.findall(r"\[([^\]\[]+?)\]", answer):
        if "{" in cite or "}" in cite or '"' in cite or cite.lower().startswith("chart"):
            continue
        if _norm_title(cite) not in titles:
            bad.append(cite)
    return bad

def apply_verification(text: str, retrieved: list, question: str = "") -> str:
    if not text or not retrieved or text.strip() == REFUSAL:
        return text
    warnings = []
    unverified = verify_numbers(text, retrieved, question)
    if unverified:
        warnings.append(f"⚠️ **Unverified figures:** {', '.join(sorted(set(unverified)))}")
    bad = verify_citations(text, retrieved)
    if bad:
        warnings.append(f"⚠️ **Uncited sources:** {', '.join('[' + b + ']' for b in sorted(set(bad)))}")
    if warnings:
        return text + "\n\n> " + "  \n> ".join(warnings)
    return text

# =============================================================================
# GROQ CALL & CORE ASK
# =============================================================================
def _track_usage(resp):
    if hasattr(resp, "usage") and resp.usage:
        u = resp.usage
        st.session_state.setdefault("usage", {"total": 0, "calls": 0})
        st.session_state.usage["total"] += getattr(u, "total_tokens", 0) or 0
        st.session_state.usage["calls"] += 1

def _groq_call(messages, max_tok):
    # gpt-oss models spend part of max_tokens on hidden reasoning; keep it short
    extra = {"reasoning_effort": "low"} if LLM_MODEL.startswith("openai/gpt-oss") else {}
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = groq_client.chat.completions.create(
                model=LLM_MODEL, messages=messages, temperature=0.2, max_tokens=max_tok, **extra
            )
            _track_usage(resp)
            return resp
        except Exception as e:
            msg = str(e).lower()
            if extra and "reasoning" in msg:  # model/API rejected the parameter: retry without it
                extra = {}
                continue
            if "429" in msg and attempt < MAX_RETRIES:
                time.sleep(min(30, MAX_WAIT_SECONDS))
                continue
            raise
    raise RuntimeError("Max retries reached")

def _needs_self_check(q: str) -> bool:
    return any(tok in q.lower() for tok in SELF_CHECK_TRIGGERS)

def _is_exhaustive(q: str) -> bool:
    return any(tok in q.lower() for tok in EXHAUSTIVE_TRIGGERS)

def _history_messages(history: list) -> list:
    out = []
    for m in history:
        text = (m.get("content") or "")[:HISTORY_MSG_CHARS]
        for c in m.get("charts") or []:
            text += f"\n(Chart {c.get('number', '?')}: {c.get('title') or 'Untitled'} — {_fmt_points(c)})"
        if text:
            out.append({"role": m["role"], "content": text})
    return out

def _ask_impl(question: str, history: list, all_charts: list = None):
    """Returns (answer, retrieved, is_visual). RAISES on LLM failure so errors are never cached."""
    all_charts = all_charts or []

    reply = _smalltalk_reply(question)
    if reply:
        return reply, [], False

    search_q = build_search_query(question, history, all_charts)
    is_visual = any(kw in question.lower() for kw in ["chart", "plot", "graph"])
    retrieved = []
    meta_mode = (search_q == "")

    if meta_mode:
        last_q = next((m.get("content", "") for m in reversed(history) if m.get("role") == "user"), "")
        context = (
            f"META-INSTRUCTION: The assistant refused the previous question: '{last_q}'. "
            "Explain what specific information is missing from the artifacts and what the artifacts do cover. "
            "Do NOT switch topics and do NOT add citations."
        )
    else:
        calc = _parse_hypothetical(question)
        if calc:  # make sure the formula and tier-threshold chunks are retrieved
            search_q += " composite score formula weights fit engagement broker bonus tier thresholds"
        retrieved, raw_max = retrieve(search_q)
        # Off-topic guard: nothing in the query looks like our domain AND nothing is similar to any chunk
        if raw_max < MIN_RAW_SIM and not detect_intents(search_q):
            return REFUSAL, [], False
        context = "\n\n".join(f"[{c['title']}]\n{sanitize_context(c['text'])}" for c in retrieved)
        if calc:
            f, e, b, score = calc
            context = (
                f"CALCULATION (computed in code; use this value, do not recompute): "
                f"composite = 0.4 x {f:g} + 0.6 x {e:g} + 0.8 x {b:g} = {score:g}\n\n" + context
            )
        hint = _chart_hint(all_charts, question)
        if hint:
            context = f"CHART CONTEXT: {hint}\n\n{context}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += _history_messages(history)
    messages.append({"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)})

    if _is_exhaustive(question):
        max_tok = MAX_TOKENS_EXHAUSTIVE
    elif is_visual:
        max_tok = MAX_TOKENS_VISUAL
    else:
        max_tok = MAX_TOKENS_DEFAULT

    answer = _groq_call(messages, max_tok).choices[0].message.content or ""
    if not answer.strip():
        raise RuntimeError("The model returned an empty answer (its token budget may have been used up). Please try again.")

    if _needs_self_check(question) and not is_visual and not meta_mode:
        try:
            check = _groq_call(
                [
                    {"role": "system", "content": SELF_CHECK_PROMPT},
                    {"role": "user", "content": f"Q:\n{question}\n\nA:\n{answer}\n\nC:\n{context[:4000]}"},
                ],
                max_tok,
            )
            checked = (check.choices[0].message.content or "").strip()
            if checked and checked.upper() != "UNCHANGED":
                answer = checked
        except Exception:
            pass  # keep the original answer if the audit pass fails

    return normalize_citations(answer), retrieved, is_visual

@st.cache_data(ttl=3600, max_entries=50, show_spinner=False)
def _cached_ask(question: str):
    # An exception raised here is NOT cached by Streamlit, so a transient 429 will not stick for an hour.
    return _ask_impl(question, [], [])

def ask(question: str, history=None, all_charts=None):
    """Returns (answer, retrieved, is_visual, error_message_or_None)."""
    history = history or []
    try:
        if not history and not st.session_state.get("bypass_cache", False):
            answer, retrieved, is_visual = _cached_ask(question)
        else:
            answer, retrieved, is_visual = _ask_impl(question, history, all_charts or [])
        return answer, retrieved, is_visual, None
    except Exception as e:
        return "", [], False, str(e)

# =============================================================================
# PARSER & RENDERER
# =============================================================================
def _try_parse_chart(raw: str):
    raw = raw.strip()
    if not raw.startswith("{") or not raw.endswith("}"):
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if "data" not in data:
        for alias in ("rows", "items", "values"):
            if alias in data:
                data["data"] = data.pop(alias)
                break

    if "data" not in data:
        return None
    if "type" not in data:
        data["type"] = "bar"

    # Reject empty or malformed data arrays
    if not isinstance(data["data"], list) or not data["data"]:
        return None

    for row in data["data"]:
        if not isinstance(row, dict):
            return None
        if "label" not in row and "name" in row:
            row["label"] = row.pop("name")
        if "value" not in row and "count" in row:
            row["value"] = row.pop("count")
        try:
            v = float(row["value"])
        except Exception:
            return None
        row["value"] = int(v) if v.is_integer() else v
    return data

def _find_matching_brace(text: str, start: int) -> int:
    if start >= len(text) or text[start] != "{":
        return -1
    depth, i, in_string, escape = 0, start, False, False
    while i < len(text):
        c = text[i]
        if escape:
            escape = False
        elif c == "\\":
            escape = True
        elif c == '"':
            in_string = not in_string
        elif not in_string:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1

def parse_response(answer: str):
    charts, clean_text = [], answer
    fenced = re.compile(r"```(?:chart|json)?\s*\n?([\s\S]*?)```", re.MULTILINE)

    def _replace(m):
        d = _try_parse_chart(m.group(1))
        if d:
            charts.append(d)
            return ""
        return m.group(0)

    clean_text = fenced.sub(_replace, clean_text)
    i = 0
    while i < len(clean_text):
        if clean_text[i] == "{":
            end = _find_matching_brace(clean_text, i)
            if end != -1:
                d = _try_parse_chart(clean_text[i:end + 1])
                if d:
                    charts.append(d)
                    clean_text = clean_text[:i] + clean_text[end + 1:]
                    continue
        i += 1
    return clean_text.strip(), charts

def _pie_is_valid(values) -> bool:
    s = sum(float(v) for v in values if v is not None)
    return 50 <= s <= 105 or (0.9 <= s <= 1.1)

def render_chart(chart_data):
    df = pd.DataFrame(chart_data["data"])
    if df.empty or "value" not in df.columns:
        return

    chart_type = chart_data.get("type", "bar").lower()
    # Pie charts of raw counts are valid (Altair normalises theta), so a pie request is honoured as-is
    if chart_type == "bar":  # only bars are re-sorted; lines keep the model's order (time/tier order)
        df = df.sort_values("value", ascending=False).reset_index(drop=True)

    order = df["label"].tolist()
    if chart_type == "pie":
        c = alt.Chart(df).mark_arc(innerRadius=50).encode(
            theta="value:Q", color=alt.Color("label:N", title="Category"), tooltip=["label", "value"]
        )
    elif chart_type == "line":
        c = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X("label:N", sort=order, axis=alt.Axis(labelAngle=-45)), y="value:Q", tooltip=["label", "value"]
        )
    else:
        c = alt.Chart(df).mark_bar().encode(
            x=alt.X("label:N", sort=order, axis=alt.Axis(labelAngle=-45)), y="value:Q", tooltip=["label", "value"]
        )

    title = chart_data.get("title") or "Chart"
    if chart_data.get("number") is not None and not str(title).lower().startswith("chart"):
        title = f"Chart {chart_data['number']}: {title}"
    st.altair_chart(c.properties(title=title), use_container_width=True)

# =============================================================================
# UI
# =============================================================================
st.title("🎯 B2B Lead Scoring RAG")
st.caption("Ask questions about Fit Scores, Engagement, Archetypes, and Tiers.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "usage" not in st.session_state:
    st.session_state.usage = {"total": 0, "calls": 0}

with st.sidebar:
    st.session_state.bypass_cache = st.checkbox("Force fresh response", value=False)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.cache_data.clear()
        st.rerun()
    st.markdown(f"**LLM:** `{LLM_MODEL}`")
    st.markdown(f"**Tokens used:** {st.session_state.usage['total']:,}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("content"):
            st.markdown(msg["content"])
        if msg.get("charts"):
            for c in msg["charts"]:
                render_chart(c)
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['title']}** — raw cosine {s.get('raw_score', 0.0):.3f}")

def process(q: str):
    q = q.strip().strip(QUOTE_CHARS).strip()  # pasted prompts often arrive wrapped in quotes
    if not q:
        return

    history = st.session_state.messages[-(2 * HISTORY_TURNS):]
    all_charts = [c for m in st.session_state.messages for c in (m.get("charts") or [])]

    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            raw, retrieved, _, error = ask(q, history=history, all_charts=all_charts)

        if error:
            # Do not store failed turns, so the question can simply be re-sent
            st.error(f"⚠️ Something went wrong: {error}")
            return

        clean, charts = parse_response(raw)
        if charts:  # drop lines that are only a bare citation, left over after the chart JSON was removed
            clean = re.sub(r"(?m)^\s*\[[^\]]+\]\s*$\n?", "", clean).strip()

        # Follow-ups ("which slice is biggest?") legitimately rely on the previous turn's sources and charts,
        # so verify against those too; otherwise correct answers get false warnings.
        prev_sources = [s for m in history for s in (m.get("sources") or [])]
        chart_pool = [
            {
                "title": c.get("title") or "",
                "text": _fmt_points(c, 50),
                "total": sum(float(d.get("value", 0) or 0) for d in c.get("data", [])),
            }
            for c in all_charts
        ]
        clean = apply_verification(clean, retrieved + prev_sources + chart_pool, q)

        next_n = len(all_charts) + 1
        for c in charts:
            c["number"] = next_n
            if not c.get("title"):
                c["title"] = f"Chart {next_n}"
            next_n += 1

        if clean:
            st.markdown(clean)
        for c in charts:
            render_chart(c)
        if retrieved:
            with st.expander(f"Sources ({len(retrieved)})"):
                for c in retrieved:
                    st.markdown(f"**{c['title']}** — raw cosine {c.get('raw_score', 0.0):.3f}")

    st.session_state.messages.append({"role": "user", "content": q})
    st.session_state.messages.append({"role": "assistant", "content": clean, "charts": charts, "sources": retrieved})

if user_q := st.chat_input("Ask about Priority MQLs, Champions, or Fit Scores..."):
    process(user_q)
