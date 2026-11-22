# Software Engineering Chatbot: From-Scratch vs Fine-Tuning vs RAG

A comparative study of three deep-learning approaches to building a chatbot specialized in **software engineering** (OOP, UML, design patterns, testing, DevOps, databases, APIs, Git, agile...). The three systems are implemented, trained and evaluated on the same domain with consistent metrics.

> Academic project, National School of Applied Sciences of Al Hoceima (ENSAH), supervised by Pr. Mohammed Cherradi. Full write-up in [`rapport_deep_learnong.pdf`](rapport_deep_learnong.pdf).

## Approaches

| # | Notebook | Approach | Summary |
|---|----------|----------|---------|
| 1 | [`fromScratch.ipynb`](fromScratch.ipynb) | Seq2Seq + attention (from scratch) | BiLSTM encoder, Bahdanau attention, LSTM decoder in TensorFlow/Keras. GloVe / Word2Vec embeddings. Trained on ~12.7k examples (Stack Overflow, CodeSearchNet, GitHub Issues). |
| 2 | [`finetunedRag.ipynb`](finetunedRag.ipynb) | Fine-tuned FLAN-T5 + retrieval | `flan-t5-base` fine-tuned on a curated SE Q&A set (655 pairs), with FAISS retrieval, CrossEncoder reranking and EN/FR support. |
| 3 | [`rag.ipynb`](rag.ipynb) | Full RAG pipeline (**selected**) | Hybrid retrieval (FAISS with `multilingual-e5-large` + BM25), CrossEncoder reranking, prompt construction, and a LoRA-adapted FLAN-T5 generator. Gradio interface. |

### RAG pipeline

```
User question
   -> cleaning / language detection (EN, FR)
   -> hybrid retrieval: FAISS (E5 embeddings) + BM25, score = 0.6 * dense + 0.4 * BM25
   -> CrossEncoder reranking (ms-marco-MiniLM-L-6-v2)
   -> prompt with top contexts
   -> FLAN-T5-base + LoRA (r=16, alpha=32, dropout=0.05)
   -> grounded answer (Gradio UI)
```

LoRA is applied to the `q` and `v` projections: only ~1.77M of ~249M parameters (0.71%) are trainable.

## Results

| Approach | Main metrics |
|----------|--------------|
| From-scratch Seq2Seq | BLEU = **0.0003** (insufficient data to learn the domain) |
| Fine-tuned FLAN-T5 | Retrieval and generation at 1.0000 on validation (see caveat below) |
| **RAG** | ROUGE-1 = **0.6864**, ROUGE-L = **0.6696**, semantic similarity = **0.9449** (81 validation examples) |

**Caveat.** The perfect scores of the fine-tuned model most likely reflect overlap between validation data and the indexed/training data rather than real generalization, and the model was observed to hallucinate on out-of-corpus questions. This motivated the move to RAG, whose lower ROUGE reflects answers that are reformulated rather than copied, while semantic similarity stays high.

**Conclusion.** RAG was retained: it is the most robust to rephrased questions, produces grounded answers, reduces hallucinations, and lets the knowledge base be updated independently of the model.

## Tech stack

Python, TensorFlow/Keras, PyTorch, Hugging Face Transformers, PEFT (LoRA), Sentence-Transformers, FAISS, rank-bm25, Gradio. Experiments were run on Kaggle with a Tesla T4 GPU.

## Repository structure

```
.
├── fromScratch.ipynb            # Seq2Seq + attention baseline
├── finetunedRag.ipynb           # Fine-tuned FLAN-T5 + FAISS + CrossEncoder
├── rag.ipynb                    # Final hybrid RAG chatbot (LoRA + Gradio)
└── rapport_deep_learnong.pdf    # Full paper (IEEE-style)
```

## Getting started

The notebooks were developed on Kaggle. To run them:

1. Clone the repo and open a notebook on Kaggle or Google Colab (GPU recommended).
2. Install the dependencies used in the notebook, typically:
   ```bash
   pip install transformers sentence-transformers faiss-cpu rank-bm25 peft evaluate gradio
   ```
3. Provide the datasets referenced in each notebook, then run the cells in order.

## Limitations and future work

- Corpora are small; the knowledge base is limited to its content and may miss recent tools.
- Automatic metrics do not capture pedagogical quality: a human evaluation with students is planned.
- Ideas: domain classifier to refuse out-of-scope questions, multi-turn memory, richer web app, larger LLMs (FLAN-T5-large, Mistral, Llama).

## Authors

- Aya HAITI
- Hasnae ELBALLAOUI
- Salma NECHDA
- Rim MAZGOUT

Supervisor: Pr. Mohammed CHERRADI, ENSAH.
