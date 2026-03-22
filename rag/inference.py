import litellm

SYSTEM_PROMPT = """You are a helpful assistant. Answer the question using ONLY
the provided context. If the answer cannot be found in the context, say 
"I cannot find this information in the provided context." Do not make up answers."""

def runInference(model: str, question: str, context: str, apiKey: str | None = None, n_samples: int = 5) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context : \n {context} \n\nQuestion: {question}"}
    ]

    primaryResponse = litellm.completion(model=model, messages=messages, temperature=0)
    mainAnswer = primaryResponse.choices[0].message.content

    #for selfcheckgpt

    samples = []
    for i in range(n_samples):
        samp = litellm.completion(model=model, messages=messages, temperature=0.5)
        samples.append(samp.choices[0].message.content)

    return {
        "answer": mainAnswer,
        "samples": samples,
        "input_tokens": primaryResponse.usage.prompt_tokens,
        "output_tokens": primaryResponse.usage.completion_tokens,
        "cost_usd": litellm.completion_cost(primaryResponse)
    }
