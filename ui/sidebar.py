import streamlit as st
from litellm import model_cost

def renderSidebar():
    st.sidebar.title("Choose your LLM")
    availableModels = sorted(
    m for m in model_cost.keys()
    if not m.startswith(("vertex_ai", "vertex_ai_beta"))
)
    selectedModel = st.sidebar.selectbox(
        "Model",
        ["Select Your Model"] + availableModels,
        index=0,
        key="model_select",
    )
    if selectedModel != "Select Your Model":
        modelMetaData = model_cost.get(selectedModel, {})
        if modelMetaData:
            inputCost = modelMetaData.get("input_cost_per_token", 0) * 1_000_000
            outputCost = modelMetaData.get("output_cost_per_token", 0) * 1_000_000
            st.sidebar.caption(
                f"{inputCost:.2f} / 1M input tokens   .   "
                f"{outputCost:.2f} / 1M output tokens"
            )
    manualOverride = st.sidebar.text_input(
        "Type a model name directly",
        placeholder="eg. groq/llama3-70b-8192",
        help="Overrides selector. PLEASE USE LITELLMS EXACT MODEL STRING!!!",
        key="model_override",
    )
    finalModel = manualOverride.strip() if manualOverride.strip() else selectedModel
    apiKey = st.sidebar.text_input(
        "API Key",
        type="password",
        help="Enter API key for your selected model",
        key="api_key",
    )
    st.sidebar.divider()
    contextFile = st.sidebar.file_uploader(
        "Upload context for testing",
        accept_multiple_files=True,
        type=["pdf", "txt", "md"],
        key="context_upload",
    )
    qaFile = st.sidebar.file_uploader(
        "Upload QA file",
        type=["json"],
        key="qa_upload",
    )
    st.sidebar.subheader("Eval Settings")
    nSamp = st.sidebar.slider(
        "SelfCheckGPT Samples",
        min_value=3, max_value=10, value=5,
        help="Set sample value for SelfCheckGPT",
        key="n_samples",
    )
    topK = st.sidebar.slider(
        "RAG top-k chunks",
        min_value=1, max_value=10, value=4,
        help="No. of chunks retrieved per question",
        key="top_k",
    )
    chunkTokens = st.sidebar.slider(
        "Chunk size in tokens",
        min_value=128, max_value=1024, value=512, step=64,
        help="choose chunk size",
        key="chunk_size",
    )
    st.sidebar.divider()
    no = False
    if finalModel == "Select Your Model" or contextFile == None or qaFile == None:
        st.sidebar.warning("Complete above things")
        no = True

    run = st.sidebar.button("Run Eval", type="primary", use_container_width=True, disabled=no, key="run_btn")
    return finalModel, apiKey, contextFile, qaFile, nSamp, topK, chunkTokens, run
