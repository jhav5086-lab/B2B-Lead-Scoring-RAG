import re
import json
import os
import time
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

# We ONLY check for chunks.json now. No .npy file needed!
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

TOP_K = 4
MIN_RAW_SIM = 0.10
HISTORY_TURNS = 2
HISTORY_MSG_CHARS = 500
MAX_TOKENS_DEFAULT = 800
MAX_TOKENS_VISUAL = 2000
MAX_TOKENS_EXHAUSTIVE = 3500
INTENT_BOOST_MULT = 2.0
INTENT_BOOST_ADD = 0.20
REPORT_PENALTY = 0.15
MAX_RETRIES = 2
MAX_WAIT_SECONDS = 120

SELF_CHECK_TRIGGERS = (
    "confirm", "prove", "proves", "right?", "correct?", "is that true",
    "definitely", "always", "never", "guarantee", "means that", "so we should",
)

META_FOLLOWUPS = ("why", "why?", "explain", "how come", "what do you mean",
                  "what?", "huh", "why not", "why not?", "which", "how")

EXHAUSTIVE_TRIGGERS = (
    "all plots", "all charts", "every chart", "every plot",
    "in one go", "maximum charts", "generate everything",
)

# =============================================================================
# BACKEND + MODEL RESOLUTION (Embeddings generated on the fly!)
# =============================================================================
@st.cache_resource
def load_backend():
    embedder = SentenceTransformer(EMBED_MODEL)
    
    # Load the JSON
    with open("./embeddings/chunks.json") as f:
        chunks = json.load(f)
        
    # Generate embeddings directly from the text chunks
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
    if AVAILABLE_MODELS:
        st.info("Available models — set `LLM_MODEL` in secrets:\n\n" +
                "\n".join(f"- `{m}`" for m in AVAILABLE_MODELS))
    st.stop()

# =============================================================================
# SYSTEM PROMPT (Tailored for B2B Lead Scoring)
# =============================================================================
SYSTEM_PROMPT = "\n".join([
    "You are a senior RevOps (Revenue Operations) consultant for a B2B sales team.",
    "",
    "STYLE: 2-4 sentences, direct answer first. Never refuse charts — generate JSON.",
    "",
    "GROUNDING: use ONLY facts in CONTEXT. If not in CONTEXT: reply exactly 'The provided model artifacts do not contain this information.'",
    "",
    "PUSHBACK:",
    "1. Do NOT confirm false premises. If a user claims 'Champions always convert', correct them: High fit + engagement correlates with likelihood, but doesn't guarantee conversion.",
    "2. If the user states a score that contradicts CONTEXT, CORRECT it: 'That figure differs from the artifacts; the correct value is X.'",
    "",
    "TIER LOGIC: Priority MQL requires Score >=70, Premium >=$1M, Engagement >=15, and an active signal. MQL is Score >=50.",
    "ARCHETYPES: Champion (High Fit/Eng/Multi), Achiever (High Fit/Eng/Single), Watcher (High Fit/Mod Eng), Dormant Giant (High Fit/Low Eng), Fader (Low Fit/Mod Eng), Lurker (Low Fit/Low Eng).",
    "",
    "MEMORY: assistant turns include '(Chart N: title — label=value)'. If user references a chart, use history.",
    "CHART REFS: in prose use 'Chart 1' — NEVER square brackets.",
    "",
    "CHART FORMAT: wrap JSON in ```chart ... ```. BOTH keys are REQUIRED: \"type\" AND \"data\". Bar/line/pie: {\"type\": \"bar\", \"data\": [{\"label\": str, \"value\": number}]}.",
    "",
    "CAPABILITY QUESTIONS: if user asks 'what can you do', 'what charts', 'what plots' — describe in PROSE only. Do NOT emit chart JSON.",
    "",
    "FILES: cannot produce files. Say 'I can't generate files, but here is a slide-ready content outline:'",
    "",
    "CITATIONS: after each factual sentence add [Chunk Title]."
])

SELF_CHECK_PROMPT = "\n".join([
    "You are auditing a proposed answer for statistical integrity.",
    "Check for: 1. Confirming false premises (e.g., 'Score guarantees conversion').",
    "2. Missing citations. 3. Fabricated numbers.",
    "If any failure: REWRITE the answer to fix it. If no failures: return UNCHANGED. Return ONLY the answer text."
])

USER_TEMPLATE = "CONTEXT CHUNKS:\n{context}\n\nUSER QUESTION: {question}\n\nGrounded answer with [Title] citations (or Chart JSON if requested):"

# =============================================================================
# INTENT & RETRIEVAL
# =============================================================================
INTENT_KEYWORDS = {
    "caveats": ["limitation", "caveat", "risk", "guarantee", "convert"],
    "recommendations": ["priority", "recommend", "mql", "sql", "tier"],
    "segments": ["segment", "archetype", "champion", "achiever", "watcher", "lurker"],
    "methodology": ["method", "fit score", "engagement", "weight", "calculate"],
    "model_summary": ["score", "total", "average", "dataset", "rows"],
}

ANAPHORA_TOKENS = ("previous", "that chart", "this chart", "the chart", "above", "last one")

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
    
    sims = raw_sims.copy()
    for i, c in enumerate(CHUNKS):
        s = float(raw_sims[i])
        if c["type"] in hints: s = s * INTENT_BOOST_MULT + INTENT_BOOST_ADD
        elif c["type"] == "report_page": s = s - REPORT_PENALTY
        sims[i] = s
        
    top_idx = list(np.argsort(sims)[::-1][:effective_k])
    results = [{"title": CHUNKS[i]["title"], "type": CHUNKS[i]["type"], "text": CHUNKS[i]["text"], "score": float(sims[i]), "raw_score": float(raw_sims[i])} for i in top_idx]
    return results, raw_max

def _get_chart_back(question: str) -> tuple:
    q = question.lower()
    explicit = re.findall(r"chart\s+(\d+)", q)
    if explicit: return 0, [int(n) for n in explicit]
    back = 0
    for phrase, n in [("previous to previous", 2), ("previous", 1), ("last chart", 1), ("that chart", 1)]:
        if phrase in q and n > back: back = n
    return back, []

def _chart_hint(history: list, question: str) -> str:
    back, explicit_refs = _get_chart_back(question)
    charts = [c for m in history for c in (m.get("charts") or [])]
    if not charts: return ""
    
    if explicit_refs:
        hints = []
        for n in explicit_refs:
            idx = n - 1
            if 0 <= idx < len(charts):
                c = charts[idx]
                hints.append(f"Chart {n} ('{c.get('title')}'): {', '.join(f'{d.get('label')}={d.get('value')}' for d in c.get('data', [])[:10])}")
        return "Referring to | " + " | ".join(hints) if hints else ""
        
    if back <= 0 or back > len(charts): target = charts[-1]; prefix = "Most recent: "
    else: target = charts[-back]; prefix = f"Chart {len(charts)-back+1}: "
    return f"{prefix}'{target.get('title')}' with data: {', '.join(f'{d.get('label')}={d.get('value')}' for d in target.get('data', [])[:8])}."

def _last_was_refusal(history: list) -> bool:
    for m in reversed(history):
        if m.get("role") == "assistant":
            return "not contain this information" in (m.get("content") or "").lower()
    return False

def build_search_query(question: str, history: list) -> str:
    if not history: return question
    q_lower = question.lower().strip()
    if _last_was_refusal(history) and q_lower in META_FOLLOWUPS: return ""
    
    back, explicit_refs = _get_chart_back(question)
    if len(question.split()) < 4 or any(t in q_lower for t in ANAPHORA_TOKENS) or explicit_refs:
        recent = " ".join(m.get("content", "") for m in history[-6:] if m.get("role") == "user").strip()
        return (_chart_hint(history, question) + " " + recent + " " + question).strip()
    return question

def sanitize_context(text: str) -> str: return text.replace("|", "\u00a6")

def normalize_citations(text: str) -> str:
    text = re.sub(r"\(Source:\s*([^)]+)\)", r"[\1]", text)
    return text.strip()

# =============================================================================
# VERIFICATION
# =============================================================================
NUM_PATTERN = re.compile(r'\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+\.\d{2,}\b|\b\d{5,}\b')
def _extract_nums(text: str) -> set:
    out = set()
    for n in re.findall(r'\d{1,3}(?:,\d{3})+|\d+\.\d+|\d+', text):
        try: out.add(round(float(n.replace(',', '')), 4))
        except: pass
    return out

def verify_numbers(answer: str, retrieved: list) -> list:
    ctx_nums = set()
    for c in retrieved: ctx_nums |= _extract_nums(c["text"])
    unverified = []
    for m in NUM_PATTERN.finditer(answer):
        raw = m.group(0)
        try: v = round(float(raw.replace(',', '')), 4)
        except: continue
        decimals = len(raw.split('.')[1]) if '.' in raw else 0
        if not any(round(c, decimals) == round(v, decimals) for c in ctx_nums): unverified.append(raw)
    return unverified

def _norm_title(t: str) -> str:
    return re.sub(r"[\s—–\-_]+", " ", re.sub(r"\s*\([^)]*\)", "", t)).strip().lower()

def verify_citations(answer: str, retrieved: list) -> list:
    titles = {_norm_title(c["title"]) for c in retrieved}
    bad = []
    for cite in re.findall(r"\[([^\]\[]+?)\]", answer):
        if "{" in cite or "}" in cite or '"' in cite or cite.lower().startswith("chart"): continue
        if _norm_title(cite) not in titles: bad.append(cite)
    return bad

def apply_verification(text: str, retrieved: list) -> str:
    if not text or not retrieved or text.strip() == "The provided model artifacts do not contain this information.": return text
    warnings = []
    unverified = verify_numbers(text, retrieved)
    if unverified: warnings.append(f"⚠️ **Unverified figures:** {', '.join(sorted(set(unverified)))}")
    bad = verify_citations(text, retrieved)
    if bad: warnings.append(f"⚠️ **Uncited sources:** {', '.join(f'[{b}]' for b in sorted(set(bad)))}")
    if warnings: return text + "\n\n> " + "  \n> ".join(warnings)
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
    for attempt in range(3):
        try:
            resp = groq_client.chat.completions.create(model=LLM_MODEL, messages=messages, temperature=0.2, max_tokens=max_tok)
            _track_usage(resp); return resp
        except Exception as e:
            if "429" in str(e) and attempt < 2: time.sleep(30); continue
            raise
    raise Exception("Max retries reached")

def _needs_self_check(q: str) -> bool: return any(tok in q.lower() for tok in SELF_CHECK_TRIGGERS)
def _is_exhaustive(q: str) -> bool: return any(tok in q.lower() for tok in EXHAUSTIVE_TRIGGERS)

def _ask_impl(question: str, history: list):
    search_q = build_search_query(question, history)
    is_visual = any(kw in question.lower() for kw in ["chart", "plot", "graph"])
    retrieved = []
    meta_mode = (search_q == "")

    if meta_mode:
        last_q = next((m.get("content", "") for m in reversed(history) if m.get("role") == "user"), "")
        context = f"META-INSTRUCTION: Refused previous question: '{last_q}'. Explain what specific artifact is missing. Do NOT switch topics."
    else:
        retrieved, raw_max = retrieve(search_q)
        if not is_visual and not history and raw_max < MIN_RAW_SIM and len(question.split()) > 3: return "The provided model artifacts do not contain this information.", [], False
        context = "\n\n".join(f"[{c['title']}]\n{sanitize_context(c['text'])}" for c in retrieved)
        hint = _chart_hint(history, question)
        if hint: context = f"CHART CONTEXT: {hint}\n\n{context}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history:
        content = m.get("content") or ""
        if m.get("charts"):
            content += "\n" + "\n".join(f"(Chart {i}: {c.get('title')} — {', '.join(f'{d.get('label')}={d.get('value')}' for d in c.get('data', [])[:8])})" for i, c in enumerate(m["charts"], 1))
        if content: messages.append({"role": m["role"], "content": content[:HISTORY_MSG_CHARS]})
    messages.append({"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)})

    max_tok = MAX_TOKENS_EXHAUSTIVE if _is_exhaustive(question) else (MAX_TOKENS_VISUAL if is_visual else MAX_TOKENS_DEFAULT)
    try: answer = _groq_call(messages, max_tok).choices[0].message.content or ""
    except Exception as e: return f"LLM error: {e}", [], False

    if _needs_self_check(question) and not is_visual and not meta_mode:
        try:
            check = _groq_call([{"role": "system", "content": SELF_CHECK_PROMPT}, {"role": "user", "content": f"Q:\n{question}\n\nA:\n{answer}\n\nC:\n{context[:4000]}"}], max_tok)
            if check.choices[0].message.content.strip(): answer = check.choices[0].message.content
        except: pass
    return normalize_citations(answer), retrieved, is_visual

@st.cache_data(ttl=3600, max_entries=50, show_spinner=False)
def _cached_ask(question: str): return _ask_impl(question, [])
def ask(question: str, history=None):
    if history is None: history = []
    return _cached_ask(question) if not history and not st.session_state.get("bypass_cache", False) else _ask_impl(question, history)

# =============================================================================
# PARSER & RENDERER
# =============================================================================
def _try_parse_chart(raw: str):
    raw = raw.strip()
    if not raw.startswith("{") or not raw.endswith("}"): return None
    try: data = json.loads(raw)
    except: return None
    if "data" not in data:
        for alias in ("rows", "items", "values"):
            if alias in data: data["data"] = data.pop(alias); break
    if "data" not in data: return None
    if "type" not in data: data["type"] = "bar"
    for row in data["data"]:
        if "label" not in row and "name" in row: row["label"] = row.pop("name")
        if "value" not in row and "count" in row: row["value"] = row.pop("count")
        try: row["value"] = float(row["value"])
        except: return None
    return data

def _find_matching_brace(text: str, start: int) -> int:
    if start >= len(text) or text[start] != "{": return -1
    depth, i, in_string, escape = 0, start, False, False
    while i < len(text):
        c = text[i]
        if escape: escape = False
        elif c == "\\": escape = True
        elif c == '"': in_string = not in_string
        elif not in_string:
            if c == "{": depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0: return i
        i += 1
    return -1

def parse_response(answer: str):
    charts, clean_text = [], answer
    fenced = re.compile(r'```(?:chart|json)?\s*\n?([\s\S]*?)```', re.MULTILINE)
    def _replace(m):
        d = _try_parse_chart(m.group(1))
        if d: charts.append(d); return ""
        return m.group(0)
    clean_text = fenced.sub(_replace, clean_text)
    i = 0
    while i < len(clean_text):
        if clean_text[i] == "{":
            end = _find_matching_brace(clean_text, i)
            if end != -1:
                d = _try_parse_chart(clean_text[i:end + 1])
                if d: charts.append(d); clean_text = clean_text[:i] + clean_text[end + 1:]; continue
        i += 1
    return clean_text.strip(), charts

def _pie_is_valid(values) -> bool:
    s = sum([float(v) for v in values if v is not None])
    return 50 <= s <= 105 or (0.9 <= s <= 1.1)

def render_chart(chart_data):
    df = pd.DataFrame(chart_data["data"])
    chart_type = chart_data.get("type", "bar").lower()
    if chart_type == "pie" and not _pie_is_valid(df["value"].tolist()): chart_type = "bar"
    if chart_type in ("bar", "line"): df = df.sort_values("value", ascending=False).reset_index(drop=True)

    if chart_type == "pie":
        c = alt.Chart(df).mark_arc(innerRadius=50).encode(theta="value:Q", color=alt.Color("label:N", title="Category"), tooltip=["label", "value"])
    elif chart_type == "line":
        c = alt.Chart(df).mark_line(point=True).encode(x=alt.X("label:N", sort=df["label"].tolist(), axis=alt.Axis(labelAngle=-45)), y="value:Q", tooltip=["label", "value"])
    else:
        c = alt.Chart(df).mark_bar().encode(x=alt.X("label:N", sort=df["label"].tolist(), axis=alt.Axis(labelAngle=-45)), y="value:Q", tooltip=["label", "value"])
    st.altair_chart(c.properties(title=chart_data.get("title", "Chart")), use_container_width=True)

# =============================================================================
# UI
# =============================================================================
st.title("🎯 B2B Lead Scoring RAG")
st.caption("Ask questions about Fit Scores, Engagement, Archetypes, and Tiers.")

if "messages" not in st.session_state: st.session_state.messages = []
if "usage" not in st.session_state: st.session_state.usage = {"total": 0, "calls": 0}

with st.sidebar:
    st.session_state.bypass_cache = st.checkbox("Force fresh response", value=False)
    if st.button("Clear chat"): st.session_state.messages = []; st.cache_data.clear(); st.rerun()
    st.markdown(f"**LLM:** `{LLM_MODEL}`")
    st.markdown(f"**Tokens used:** {st.session_state.usage['total']:,}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("content"): st.markdown(msg["content"])
        if msg.get("charts"):
            for c in msg["charts"]: render_chart(c)
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for s in msg["sources"]: st.markdown(f"**{s['title']}** — raw cosine {s.get('raw_score', 0.0):.3f}")

def process(q: str):
    history = st.session_state.messages[-4:]
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"): st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            raw, retrieved, _ = ask(q, history=history)
            clean, charts = parse_response(raw)
            clean = apply_verification(clean, retrieved)
            if clean: st.markdown(clean)
            for c in charts: render_chart(c)
            if retrieved:
                with st.expander(f"Sources ({len(retrieved)})"):
                    for c in retrieved: st.markdown(f"**{c['title']}** — raw cosine {c.get('raw_score', 0.0):.3f}")
            st.session_state.messages.append({"role": "assistant", "content": clean, "charts": charts, "sources": retrieved})

if user_q := st.chat_input("Ask about Priority MQLs, Champions, or Fit Scores..."): process(user_q)
