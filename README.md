<div align="center">

# 🎯 B2B Lead Group Insurance Scoring RAG

**An intelligent RevOps chatbot that translates complex account scoring models into actionable sales strategies.**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Try_it_now-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://b2b-lead-scoring-rag-kazhjrkrfko5tzneybtycs.streamlit.app/)
[![Eval Suite](https://img.shields.io/badge/Evals-32%2F32_Passing-success?style=for-the-badge)](#-evaluation)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Groq](https://img.shields.io/badge/LLM-Groq-F55036?style=for-the-badge)](https://groq.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

[**Overview**](#-overview) · [**Features**](#-features) · [**Quick Start**](#-quick-start) · [**Examples**](#-example-interactions) · [**Architecture**](#️-architecture) · [**Evaluation**](#-evaluation)

</div>

---

## 📌 Overview

B2B sales teams rely on lead scoring models to prioritize accounts, but the underlying logic (Fit Scores, Engagement decay, multi-threading bonuses) is often locked in a black-box dashboard.

This project uses **Retrieval-Augmented Generation (RAG)** to let Account Executives and RevOps managers query the model in plain English. It explains *why* an account is a **Champion** vs. a **Watcher**, and what it would take to move a **Dormant Giant** into an **Achiever**.

> **Commercialized AI Skill:** The core logic of this repository has been packaged as a commercial `SKILL.md` for **Agensi.io**, enabling cross-agent compatibility (Claude Code, Cursor) for RevOps teams.

### 🛡️ The Verification Layer
This app doesn't just trust the LLM. A post-generation layer checks every answer:
- **Numeric guardrails:** flags figures (scores, counts, percentages, dollar amounts) that don't appear in the retrieved source text. Derived percentages are accepted only if they follow from numbers in the sources.
- **Citation checks:** every `[Chunk Title]` the model cites must be a chunk that was actually retrieved.
- **False-premise pushback:** if a user claims *"A score of 80 guarantees they will buy,"* the app corrects them using the methodology caveats.
- **Code-computed math:** for hypothetical accounts ("Suppose Fit is 50 and Engagement is 30..."), the composite score is calculated in code and handed to the LLM, so the arithmetic is not left to the model.

<table>
<tr>
<td width="50%">

### 🎯 What it does
- Explains archetype logic (Champion vs. Lurker)
- Defines Priority MQL vs. standard MQL
- Breaks down Fit Score and Engagement calculations
- Scores hypothetical accounts on request
- Generates charts of account distributions (bar, line, pie)

</td>
<td width="50%">

### 🛡️ What makes it safe
- Grounded strictly in `chunks.json`
- Refuses to answer from general knowledge
- Flags unverified figures with `⚠️`
- Explains *why* it can't answer a question
- Covered by an in-app regression test suite

</td>
</tr>
</table>

---

## ✨ Features

### 🧠 Business Logic
| Feature | Detail |
|---|---|
| **Composite Scoring** | `0.4 × Fit Score + 0.6 × Engagement + 0.8 × Broker Bonus` |
| **Tiering** | Priority MQL, MQL, Responder, Disqualified |
| **Priority MQL** | Score ≥ 70, premium ≥ $1M, engagement ≥ 15, and at least one active signal |
| **MQL** | Score ≥ 50 |
| **Archetypes** | Champion, Achiever, Watcher, Dormant Giant, Fader, Lurker, Detractor (disqualified) |
| **Fit Scoring** | Renewal proximity, premium size, industry fit, carrier, and broker presence |

### 🔍 RAG Pipeline
| Feature | Detail |
|---|---|
| **Grounded Answers** | Every factual claim cites a retrieved chunk (e.g., `[Archetype Definitions]`) |
| **Number Verification** | Rounding-aware check of figures against retrieved text, including small scores, counts and percentages |
| **Self-Check Audit** | Second LLM pass on adversarial queries (e.g., "Does this guarantee a sale?") |
| **Type-Aware Retrieval** | Multiplicative boost for intent-matched chunk types (e.g., `methodology` chunks) |
| **Conversation Memory** | Follow-ups like *"which slice is biggest?"* resolve against the latest chart |
| **Explained Refusals** | Says *"not in artifacts"* instead of inventing rules, and explains what is missing when asked *"why?"* |
| **Cheap Small Talk** | Greetings, thanks and goodbyes (including typos) are answered locally with no LLM call |

---

## 🚀 Quick Start

### Try the Live App

<div align="center">

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://b2b-lead-scoring-rag-kazhjrkrfko5tzneybtycs.streamlit.app/)

</div>

**Sample questions to try:**
```text
💡 What exactly makes an account a Priority MQL versus a regular MQL?
💡 How is the Fit Score calculated?
💡 What does the engagement score reward?
💡 What's the difference between a Champion and an Achiever, and what would it take to upgrade an Achiever?
💡 Suppose an account has Fit 50, Engagement 30, no broker bonus, a $2M premium and an urgent renewal. What tier is it?
💡 Show me all plots.
💡 Show me the tier distribution as a pie chart, then ask "which slice is biggest?"
💡 Champions always convert, right? (it should push back)
💡 What's the average conversion rate for insurance brokers in India? (it should refuse, then explain why)
