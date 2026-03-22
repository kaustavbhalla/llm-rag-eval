import streamlit as st
from litellm import model_cost

def renderSidebar() -> dict:
    st.sidebar.title("Choose your LLM")

    availableModels = sorted(model_cost.keys())

    selectedModel = st.sidebar.selectbox(
        "Model",
        ["Select Your Model"] + availableModels,
        index=0
    )

    if selectedModel != "Select Your Model":
        modelMetaData = model_cost.get(selectedModel, {})
        if modelMetaData:
            inputCost = modelMetaData.get("input_cost_per_token", 0) * 1_000_000
            outputCost = modelMetaData.get("output_cost_per_token", 0) * 1_000_000

            st.sidebar.caption(
                f"{inputCost:.2f} / 1M input tokens   .   "
                    f"{outputCost:.2f} / 1M input tokens"
            )

    manualOverride = st.sidebar.text_input(
        "Type a model name directly",
        placeholder="eg. groq/llama3-70b-8192",
        help="Overrides selector. PLEASE USE LITELLMS EXACT MODEL STRING!!!",
    )

    finalModel = manualOverride.strip() if manualOverride.strip() else selectedModel

    apiKey = st.sidebar.text_input(
        "API Key",
        type="password",
        help="Enter API key for your selected model",
    )

    st.sidebar.divider()


    contextFile = st.sidebar.file_uploader(
        "Upload context for testing",
        accept_multiple_files=True,
        type=["pdf", "txt", "md"],
    )

    qaFile = st.sidebar.file_uploader(
        "Upload QA file",
        type=["json"],
    )

    st.sidebar.subheader("Eval Settings")

    nSamp = st.sidebar.slider(
        "SelfCheckGPT Samples",
        min_value=3, max_value=10, value=5,
        help="Set sample value for SelfCheckGPT"
    )

    topK = st.sidebar.slider(
        "RAG top-k chunks",
        min_value=1, max_value=10, value=4,
        help="No. of chunks retrieved per question"
    )

    chunkTokens = st.sidebar.slider(
        "Chunk size in tokens",
        min_value=128, max_value=1024, value=512, step=64,
        help="choose chunk size"
    )

    st.sidebar.divider()

    if finalModel == "Select Your Model" or contextFile == None or qaFile == None:
        st.sidebar.warning("Complete above things")
        run = st.sidebar.button("Run Eval", type="primary", use_container_width=True, disabled=True)

    run = st.sidebar.button("Run Eval", type="primary", use_container_width=True, disabled=False)


    return {"model": finalModel, "api": apiKey, "context_file": contextFile, "qa_json": qaFile, "n_sample": nSamp, "top_k": topK, "chunk_tokens": chunkTokens, "runInf": run}
