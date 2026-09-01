# Research basis and implementation claims

Reviewed on 2026-09-01. Links below point to primary publications or author-hosted material.
Reported benchmark numbers are evidence for testing a design, not guarantees for this corpus,
hardware, or model.

## Route between full context and retrieval

[LaRA: Benchmarking Retrieval-Augmented Generation and Long-Context LLMs — No Silver Bullet
for LC or RAG Routing](https://proceedings.mlr.press/v267/li25dv.html), ICML 2025, evaluates
2,326 cases across four QA tasks, three long-text types, and eleven models. It finds that the
best choice depends on model capability, context length, task, and retrieval characteristics.

Design implication: preserve both modes and evaluate routing on company query classes. The
current runtime starts with an explicit full-authorized-context threshold and falls back to
retrieval.

[Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval),
Anthropic, 2024, is an industry evaluation rather than a peer-reviewed paper. It recommends
considering the whole corpus when it is below roughly 200,000 tokens. On Anthropic's tested
corpora, contextual embeddings plus contextual BM25 reduced top-20 retrieval failure from
5.7% to 2.9% (49% relative), and reranking reduced it to 1.9% (67% relative).

Design implication: full context is the first baseline. For larger corpora, contextual BM25,
local embeddings, rank fusion, and reranking become measured challengers. The percentages
must not be copied into this project's expected results.

## External and graph-assisted memory

[From RAG to Memory: Non-Parametric Continual Learning for Large Language
Models](https://proceedings.mlr.press/v267/gutierrez25a.html), ICML 2025, introduces
HippoRAG 2. It integrates graph and passage retrieval and reports gains across factual,
associative, and sense-making memory tasks, including a seven-point improvement on the
authors' associative-memory comparison.

Design implication: external memory can combine passages and relationships without rewriting
model weights. A graph is deferred until frozen multi-hop evaluations show that simpler
retrieval is insufficient.

[RAG Meets Temporal Graphs: Time-Sensitive Modeling and Retrieval for Evolving
Knowledge](https://arxiv.org/abs/2510.13590), 2025, is a preprint. It represents timestamped
relations explicitly and updates affected temporal summaries incrementally.

Design implication: represent valid time and updates explicitly from the first milestone.
Its specific generated-summary graph is not adopted until independently reproduced or useful
on this project's evaluation set.

## Bounded agentic retrieval

[AgenticRAG: Agentic Retrieval for Enterprise Knowledge
Bases](https://arxiv.org/abs/2605.05538), Microsoft, May 2026, is a preprint. Its harness uses
search, find, open, and summarize tools. The authors report 92% answer correctness on
FinanceBench with GPT-5-mini, compared with 94% when oracle evidence was supplied, plus gains
on BRIGHT and WixQA.

Design implication: iterative navigation is a later fallback for difficult queries, with tool,
document, token, and time budgets. These results used hosted frontier models and do not
establish the same performance for a local MLX model.

## Authorization before retrieval

[Authorization-First Retrieval: Enforcing Least Privilege in Multi-Agent RAG
Systems](https://aclanthology.org/2026.trustnlp-main.15/), TrustNLP 2026, formalizes the
ordering rule that authorization must constrain the retrieval candidate set before learned
components consume content. Its controlled evaluation reports structural exposure for
retrieve-then-filter and elimination of that class of leak by construction under its setup.

[Securing the Agent: Vendor-Neutral, Multitenant Enterprise Retrieval and Tool
Use](https://doi.org/10.1145/3786335.3813145), ACM 2026, similarly motivates policy-aware
ingestion, retrieval-time gating, and server-side enforcement for multitenant agent systems.

Design implication: the retrieval router accepts only an `AuthorizedCorpus`. Temporal version
resolution precedes status and ACL checks so revocation cannot reveal a stale version. Tests
assert that restricted text never reaches retrieval or prompt compilation.

## Claim tracing and provenance

[VeriTrail: Closed-Domain Hallucination Detection with
Traceability](https://openreview.net/forum?id=Sr0btZuwBi), ICLR 2026, represents multi-step
generative workflows as directed acyclic graphs, extracts claims from terminal outputs, and
traces evidence backward toward source nodes. It returns support verdicts and evidence trails.

Design implication: retain record/version/source identifiers, authority snapshots, selected
evidence-bundle identities, and durable execution traces throughout the runtime, then add
atomic claim support as an independent stage. The runtime currently combines deterministic
citation/contradiction checks with a fail-closed, distinct local model judgment of semantic
support and internal consistency. Same-model self-approval cannot publish. This is useful
defense in depth, but it does not yet provide VeriTrail-like atomic evidence trails or
establish that the judge is reliable.

## Local Apple Silicon execution

[MLX-LM](https://github.com/ml-explore/mlx-lm) is Apple's open-source Python package for
generation and fine-tuning on Apple Silicon. Its public API supports local model loading,
chat-template formatting, deterministic sampling, and generation.

Design implication: MLX-LM is loaded lazily behind a provider protocol. Core storage,
authorization, retrieval, prompt compilation, and tests have no MLX dependency. The default
provider policy requires an existing local model directory; network acquisition is an
explicit opt-in.

## What the literature does not establish

No cited publication validates this complete architecture as one system, proves its security,
or identifies a universally best retriever. Several sources are preprints or vendor
evaluations. Consequently, every advanced component must beat frozen local baselines while
preserving temporal correctness, authorization non-exposure, unknown rejection, citation
integrity, latency, and memory limits on the target Mac.
