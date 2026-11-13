<div align="center">

# 🎯 B2B Lead Scoring RAG

**An intelligent RevOps chatbot that translates complex account scoring models into actionable sales strategies.**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Try_it_now-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](YOUR_STREAMLIT_URL_HERE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Groq](https://img.shields.io/badge/LLM-Groq-F55036?style=for-the-badge)](https://groq.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

[**Overview**](#-overview) · [**Features**](#-features) · [**Quick Start**](#-quick-start) · [**Examples**](#-example-interactions) · [**Architecture**](#️-architecture)

</div>

---

## 📌 Overview

B2B Sales teams rely on Lead Scoring models to prioritize accounts, but the underlying logic (Fit Scores, Engagement decay, Multi-threading bonuses) is often locked in a black-box dashboard. 

This project uses **Retrieval-Augmented Generation (RAG)** to allow Account Executives and RevOps managers to query the model in plain English. It explains *why* an account is a **Champion** vs. a **Watcher**, and what specific actions will move a **Dormant Giant** into an **Achiever**.

### 🛡️ The Verification Layer
Like all my RAG projects, this app doesn't just trust the LLM. It features a post-generation verification layer:
- **Numeric Guardrails:** Checks every score against the source model to prevent hallucinated numbers.
- **Citation Enforcement:** Forces the LLM to cite the exact business rules used to make a claim.
- **False Premise Pushback:** If a user claims *"A score of 80 guarantees they will buy,"* the app corrects them based on the methodology caveats.

<table>
<tr>
<td width="50%">

### 🎯 What it does
- Explains Archetype logic (Champion vs. Lurker)
- Defines Priority MQL vs. standard MQL
- Breaks down Fit Score and Engagement calculations
- Generates dynamic charts of account distributions

</td>
<td width="50%">

### 🛡️ What makes it safe
- Grounded strictly in `chunks.json`
- Refuses to answer from general knowledge
- Flags unverified numbers with `⚠️`
- Explains *why* it can't answer a question

</td>
</tr>
</table>

---

## ✨ Features

### 🧠 Business Logic
| Feature | Detail |
|---|---|
| **Composite Scoring** | 0.4 * Fit Score + 0.6 * Engagement + Broker Bonus |
| **Tiering** | Priority MQL, MQL, Responder, Disqualified |
| **Archetypes** | Champion, Achiever, Watcher, Dormant Giant, Fader, Lurker |
| **Fit Scoring** | Evaluates Renewal proximity, Premium size, Industry fit, Carrier, and Broker presence |

### 🔍 RAG Pipeline
| Feature | Detail |
|---|---|
| **Grounded Answers** | Every factual claim cites a retrieved chunk (e.g., `[Archetype Definitions]`) |
| **Number Verification** | Rounding-aware check against retrieved text (e.g., catches hallucinated scores) |
| **Self-Check Audit** | Second LLM pass on adversarial queries (e.g., "Does this guarantee a sale?") |
| **Type-Aware Retrieval** | Multiplicative boost for intent-matched chunk types (e.g., boosting `methodology` chunks) |
| **Refusal Behavior** | Says *"not in artifacts"* rather than inventing sales rules |

---

## 🚀 Quick Start

### Try the Live App

<div align="center">

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_URL_HERE)
</div>

**Sample questions to try:**
