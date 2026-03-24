import streamlit as st
from ui.sidebar import renderSidebar
from rag.inference import runInference
import json
import pandas as pd

from rag.pipeline import getContextFromFiles, chunkAndBuildVectors, retrieveContext
from evaluation.semantic import semanticEval
from evaluation.lexical import computeRougeScore, compute_ngram
from evaluation.hallucination import evaluate_hallucination

st.title("LLMEval")

# Data required to run inference
model, apiKey, context_files, qa_file, n_samples, top_k, chunk_size, run = renderSidebar()

if run:
    if not context_files:
        st.error("Please upload atleast one context file")
        st.stop()
    if not qa_file:
        st.error("Upload qa json")
        st.stop()
    if not model:
        st.error("Please select model")
        st.stop()


    try:
        qaPairs = json.load(qa_file)
        if not isinstance(qaPairs, list):
            st.error("JSON format is wrong")
            st.stop()
    except json.JSONDecodeError as e:
        st.error(f"Ivalid json: {e}")
        st.stop()


    with st.spinner("Building vector embeddings"):
        documents = getContextFromFiles(context_files)
        if not documents:
            st.error("Could not extract data from file")
            st.stop()

        vectors = chunkAndBuildVectors(documents, chunk_size)

    st.success("Embeddings built successfully")


    results = []
    errors = []
    status = st.empty()


    for i, q in enumerate(qaPairs):
        question = q.get("question", "").strip()
        correctAns = q.get("correctAnswer", "").strip()

        if not question or not correctAns:
            errors.append(f"Row {i}: missing 'question' or 'correctAnswer' — skipped.")
            continue
        
        status.text(f"Processing question {i + 1} / {len(qaPairs)}: {question[:60]}...")
        

        try:
            context = retrieveContext(vectors, question, top_k)

            llmAnswer = runInference(
                model=model,
                question=question,
                context=context,
                apiKey=apiKey,
                n_samples=n_samples
            )

            print(llmAnswer)
            llmRes = llmAnswer["answer"]

            semantic    = semanticEval(llmRes, correctAns, context)
            rouge       = computeRougeScore(llmRes, correctAns)
            ngram       = compute_ngram(llmRes, correctAns)
            hallucination = evaluate_hallucination(llmRes, llmAnswer["samples"])

            results.append({
                "question":             question,
                "correct_answer":       correctAns,
                "llm_answer":           llmRes,
                "retrieved_context":    context,
                "sentence_scores":      hallucination["sentence_scores"],

                "accuracy":             semantic["accuracy"],
                "completeness":         semantic["completeness"],
                "truthfulness":         semantic["truthfulness"],
                "groundedness":         semantic["groundedness"],
                "contradiction_score":  semantic["contradiction_score"],

                "hallucination_score":  hallucination["hallucination_score"],
                "sentences_flagged":    hallucination["sentences_flagged"],
                "total_sentences":      hallucination["total_sentences"],

                **rouge,
                **ngram,

                "input_tokens":         llmAnswer["input_tokens"],
                "output_tokens":        llmAnswer["output_tokens"],
                "cost_usd":             llmAnswer["cost_usd"],
            })

        except Exception as e:
            errors.append(f"Q{i + 1} — '{question[:50]}': {e}")

    status.empty()

    if errors:
        for err in errors:
            st.warning(err)

    if results:
        st.session_state["results"] = pd.DataFrame(results)
        st.session_state["model"]   = model
        st.balloons()

    else:
        st.error("No results were produced. Check errors above.")

if "results" in st.session_state:
    df         = st.session_state["results"]
    model_used = st.session_state.get("model", "Unknown model")

    st.divider()
    st.header(f"📊 Aggregate Results — `{model_used}`")
    st.caption(f"{len(df)} questions evaluated")

    # ── Top metric cards ──────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Accuracy",      f"{df['accuracy'].mean():.1%}")
    col2.metric("Avg Completeness",  f"{df['completeness'].mean():.1%}")
    col3.metric("Avg Truthfulness",  f"{df['truthfulness'].mean():.1%}")
    col4.metric("Avg Groundedness",  f"{df['groundedness'].mean():.1%}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Avg Hallucination", f"{df['hallucination_score'].mean():.1%}")
    col6.metric("Avg ROUGE-L F1",    f"{df['rougeL_f1'].mean():.1%}")
    col7.metric("Total Tokens (I/O)",f"{df['input_tokens'].sum():,} / {df['output_tokens'].sum():,}")
    col8.metric("Total Cost",        f"${df['cost_usd'].sum():.4f}")

    # ── Semantic bar chart ────────────────────────────────────────────────────
    st.subheader("Semantic Metrics Overview")
    semantic_means = df[["accuracy", "completeness", "truthfulness", "groundedness"]].mean()
    st.bar_chart(semantic_means)

    # ── Lexical summary table ─────────────────────────────────────────────────
    st.subheader("Lexical Metrics Summary")
    lexical_cols = [
        "rouge1_f1", "rouge2_f1", "rougeL_f1",
        "unigram_f1", "bigram_f1", "trigram_f1",
    ]
    lexical_summary = df[lexical_cols].mean().rename({
        "rouge1_f1":  "ROUGE-1 F1",
        "rouge2_f1":  "ROUGE-2 F1",
        "rougeL_f1":  "ROUGE-L F1",
        "unigram_f1": "Unigram F1",
        "bigram_f1":  "Bigram F1",
        "trigram_f1": "Trigram F1",
    }).to_frame(name="Score")
    st.dataframe(lexical_summary.style.format("{:.1%}"), use_container_width=True)

    # ── Full results table ────────────────────────────────────────────────────
    st.subheader("Full Results Table")
    display_cols = [
        "question", "accuracy", "completeness", "truthfulness",
        "groundedness", "hallucination_score",
        "rouge1_f1", "rougeL_f1",
        "input_tokens", "output_tokens", "cost_usd",
    ]
    st.dataframe(
        df[display_cols].style.format({
            "accuracy":            "{:.1%}",
            "completeness":        "{:.1%}",
            "truthfulness":        "{:.1%}",
            "groundedness":        "{:.1%}",
            "hallucination_score": "{:.1%}",
            "rouge1_f1":           "{:.1%}",
            "rougeL_f1":           "{:.1%}",
            "cost_usd":            "${:.5f}",
        }),
        use_container_width=True,
        height=300,
    )

    csv = df.drop(
        columns=["sentence_scores", "retrieved_context", "llm_answer", "correct_answer"]
    ).to_csv(index=False)
    st.download_button(
        "⬇️ Download Results CSV",
        data=csv,
        file_name=f"rag_eval_{model_used.replace('/', '_')}.csv",
        mime="text/csv",
    )

    # ── Per-question drilldown ────────────────────────────────────────────────
    st.divider()
    st.header("🔍 Per-Question Drilldown")

    for idx, row in df.iterrows():
        q_preview   = row["question"][:80] + ("..." if len(row["question"]) > 80 else "")
        halluc      = row["hallucination_score"]
        halluc_icon = "🔴" if halluc > 0.5 else "🟡" if halluc > 0.25 else "🟢"

        with st.expander(f"Q{idx + 1}: {q_preview}  {halluc_icon} hallucination={halluc:.0%}"):

            ans_col1, ans_col2 = st.columns(2)
            with ans_col1:
                st.subheader("✅ Correct Answer")
                st.write(row["correct_answer"])
            with ans_col2:
                st.subheader("🤖 LLM Answer")
                st.write(row["llm_answer"])

            with st.expander("📄 Retrieved Context"):
                st.text(row["retrieved_context"])

            st.divider()

            # Semantic
            st.subheader("Semantic Metrics")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy",      f"{row['accuracy']:.1%}")
            m2.metric("Completeness",  f"{row['completeness']:.1%}")
            m3.metric("Truthfulness",  f"{row['truthfulness']:.1%}")
            m4.metric("Groundedness",  f"{row['groundedness']:.1%}")
            m5.metric("Contradiction", f"{row['contradiction_score']:.1%}", delta_color="inverse")

            # ROUGE
            st.subheader("ROUGE Metrics")
            r1, r2, r3 = st.columns(3)
            r1.metric("ROUGE-1 F1", f"{row['rouge1_f1']:.1%}")
            r2.metric("ROUGE-2 F1", f"{row['rouge2_f1']:.1%}")
            r3.metric("ROUGE-L F1", f"{row['rougeL_f1']:.1%}")

            with st.expander("Full ROUGE breakdown (precision / recall / F1)"):
                rouge_data = {
                    "Metric":    ["ROUGE-1",            "ROUGE-2",            "ROUGE-L"],
                    "Precision": [row["rouge1_precision"], row["rouge2_precision"], row["rougeL_precision"]],
                    "Recall":    [row["rouge1_recall"],    row["rouge2_recall"],    row["rougeL_recall"]],
                    "F1":        [row["rouge1_f1"],        row["rouge2_f1"],        row["rougeL_f1"]],
                }
                st.dataframe(
                    pd.DataFrame(rouge_data).set_index("Metric").style.format("{:.1%}"),
                    use_container_width=True,
                )

            # NGram
            st.subheader("NGram Metrics")
            n1, n2, n3 = st.columns(3)
            n1.metric("Unigram F1", f"{row['unigram_f1']:.1%}")
            n2.metric("Bigram F1",  f"{row['bigram_f1']:.1%}")
            n3.metric("Trigram F1", f"{row['trigram_f1']:.1%}")

            # Usage
            st.subheader("Usage")
            u1, u2, u3 = st.columns(3)
            u1.metric("Input Tokens",  f"{row['input_tokens']:,}")
            u2.metric("Output Tokens", f"{row['output_tokens']:,}")
            u3.metric("Cost",          f"${row['cost_usd']:.5f}")

            # SelfCheckGPT
            st.subheader("🧠 Hallucination Analysis (SelfCheckGPT)")
            st.caption(
                f"Flagged **{row['sentences_flagged']}** of "
                f"**{row['total_sentences']}** sentences as potentially hallucinated."
            )

            sentence_scores = row["sentence_scores"]
            if sentence_scores:
                for s in sentence_scores:
                    risk = s["risk_level"]
                    icon = "🔴" if risk == "High" else "🟡" if risk == "Medium" else "🟢"
                    ens  = s["ensemble_score"]
                    st.markdown(f"{icon} **{s['sentence']}**  `ensemble={ens:.2f}`")
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    sc1.metric("NLI Score",   s["nli_score"])
                    sc2.metric("BERTScore",   s["bertscore_score"])
                    sc3.metric("NGram Score", s["ngram_score"])
                    sc4.metric("Ensemble",    ens)
            else:
                st.info("No sentence-level data available.")

else:
    st.info(
        "👈 Upload your context files and Q&A JSON in the sidebar, "
        "pick a model, then hit **Run Evaluation**."
    )
    st.subheader("Expected Q&A JSON format")
    st.code(
        json.dumps(
            [
                {"question": "What is the capital of France?", "correctAnswer": "Paris"},
                {"question": "Who wrote Hamlet?",              "correctAnswer": "William Shakespeare"},
            ],
            indent=2,
        ),
        language="json",
    )
