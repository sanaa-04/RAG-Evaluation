# 🔍 RAG Application & Chatbot Evaluation

A comprehensive evaluation framework for Retrieval-Augmented Generation (RAG) pipelines and chatbot systems — covering data collection, LLM-as-judge methodology, evaluation metrics, and multi-model comparison.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Notebook Walkthrough](#notebook-walkthrough)
  - [1. Gathering Data Points](#1-gathering-data-points)
  - [2. LLM as Judge](#2-llm-as-judge)
  - [3. Evaluation Metrics](#3-evaluation-metrics)
  - [4. Comparison of Different LLM Models](#4-comparison-of-different-llm-models)
- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Results Summary](#results-summary)

---

## 🧠 Overview

Evaluating RAG systems and chatbots is fundamentally different from traditional ML evaluation — there's rarely a single "correct" answer, and quality is often subjective. This project addresses that challenge with a structured pipeline:

1. Systematically gather question-context-answer triplets as evaluation data
2. Use an LLM as an automated judge to score responses
3. Compute standardized evaluation metrics (faithfulness, relevance, etc.)
4. Compare multiple LLMs to identify the best-performing model for your use case

---


## 📓 Notebook Walkthrough

### 1. Gathering Data Points

The first step builds the evaluation dataset — the foundation for all subsequent analysis.

**What this section does:**
- Loads the knowledge base / document corpus used by the RAG system
- Generates diverse evaluation questions (factual, inferential, multi-hop)
- Retrieves relevant context chunks for each question using the RAG pipeline
- Collects ground-truth answers (human-annotated or synthetically generated)
- Saves structured `(question, context, ground_truth, generated_answer)` tuples

**Output:** A clean evaluation dataset in JSON/CSV format, ready for scoring.

```python
# Example data point structure
{
  "question": "What is the refund policy?",
  "context": ["...retrieved chunk 1...", "...retrieved chunk 2..."],
  "ground_truth": "Refunds are processed within 7 business days.",
  "generated_answer": "You can get a refund in 5–10 days per the policy."
}
```

---

### 2. LLM as Judge

Instead of relying solely on string-matching metrics, this section uses a powerful LLM (e.g., GPT-4, Claude) as an automated evaluator — a technique that closely correlates with human judgment.

**What this section does:**
- Designs structured judge prompts with clear scoring rubrics
- Sends each `(question, context, answer)` tuple to the judge LLM
- Collects numerical scores and reasoning explanations per response
- Handles edge cases: hallucinations, incomplete answers, off-topic replies

**Evaluation dimensions scored by the judge:**

| Dimension | Description |
|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? |
| **Answer Relevance** | Does the answer directly address the question? |
| **Context Precision** | Was the right context retrieved? |
| **Context Recall** | Was all necessary information retrieved? |
| **Coherence** | Is the answer well-structured and readable? |

**Example judge prompt structure:**
```
You are an expert evaluator. Given the question, context, and answer below,
score the answer on Faithfulness (1–5) and explain your reasoning.

Question: {question}
Context: {context}
Answer: {answer}
```

---

### 3. Evaluation Metrics

This section computes both automated NLP metrics and LLM-judge scores to give a holistic view of performance.

**Metrics computed:**

| Metric | Type | Description |
|---|---|---|
| **Faithfulness** | LLM-judge | Factual consistency with source context |
| **Answer Relevance** | LLM-judge | Alignment of answer with the question |
| **Context Precision** | LLM-judge | Fraction of retrieved context that is useful |
| **Context Recall** | LLM-judge | Coverage of necessary information |
| **ROUGE-L** | Automated | Longest common subsequence overlap with ground truth |
| **BERTScore** | Automated | Semantic similarity using BERT embeddings |
| **Exact Match** | Automated | Binary match with expected answer |

**Aggregation:** Scores are computed per data point, then averaged across the full dataset and broken down by question category (factual, multi-hop, etc.).

---

### 4. Comparison of Different LLM Models

The final section benchmarks multiple LLMs on the same evaluation dataset, enabling side-by-side performance analysis.

**Models compared** *(configurable — examples below)*:
- `gpt-4o`
- `gpt-3.5-turbo`
- `claude-3-5-sonnet`
- `gemini-1.5-pro`
- Open-source alternatives (Llama 3, Mistral, etc.)

**What this section does:**
- Runs each model through the full RAG pipeline on the same question set
- Collects and scores responses using the LLM-as-judge and automated metrics
- Generates a comparison table and visualizations (radar charts, bar plots)
- Highlights trade-offs: accuracy vs. latency vs. cost

**Sample comparison output:**

| Model | Faithfulness | Answer Relevance | Context Recall | Avg Score | Latency (s) |
|---|---|---|---|---|---|
| GPT-4o | 4.7 | 4.5 | 4.6 | **4.60** | 3.2 |
| Claude 3.5 Sonnet | 4.6 | 4.6 | 4.4 | **4.53** | 2.8 |
| GPT-3.5-turbo | 3.9 | 4.0 | 3.7 | **3.87** | 1.1 |
| Gemini 1.5 Pro | 4.3 | 4.2 | 4.1 | **4.20** | 2.5 |

> ⚡ *Scores are illustrative. Run the notebook to generate results on your dataset.*

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-evaluation.git
cd rag-evaluation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up API keys

```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
# Add other provider keys as needed
```

### 4. Run the notebook

```bash
jupyter notebook rag_evaluation.ipynb
```

Execute cells in order — each section builds on the previous one.

---

## 📦 Requirements

```
openai>=1.0.0
anthropic>=0.20.0
ragas>=0.1.0
langchain>=0.1.0
sentence-transformers>=2.2.0
bert-score>=0.3.13
rouge-score>=0.1.2
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
tqdm>=4.65.0
python-dotenv>=1.0.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📊 Results Summary

After running the full notebook, you'll have:

- ✅ A scored evaluation dataset with per-question breakdowns
- ✅ LLM judge verdicts with reasoning for every response
- ✅ Aggregated metric scores per model
- ✅ Visual comparison charts (radar, bar, scatter plots)
- ✅ A `results/metrics_summary.csv` exportable for further analysis

Use these insights to select the right model, tune your retrieval strategy, and iteratively improve your RAG pipeline.

---

## 📌 Key Takeaways

- **No single metric tells the full story** — combine LLM-judge scores with automated metrics for robust evaluation
- **Retrieval quality matters as much as generation quality** — always measure both context precision and recall
- **Cheaper models can be surprisingly competitive** — use the comparison section to find the optimal cost-performance trade-off for your use case
- **Evaluation is iterative** — re-run the notebook as you update your RAG pipeline to track improvements over time

---

## 📄 License

MIT License — feel free to adapt this framework for your own RAG projects.
