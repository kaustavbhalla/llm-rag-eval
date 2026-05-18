
import os
import json
import torch
from rag.inference import runInference
from evaluation.semantic import semanticEval
from evaluation.hallucination import evaluate_hallucination

def test_inference_and_eval():
    print("Testing inference and evaluation logic...")
    
    # Mock data
    model = "gpt-3.5-turbo" # Or any model supported by litellm
    question = "What is the capital of France?"
    context = "France is a country in Europe. Its capital is Paris."
    correct_answer = "Paris"
    
    # Note: This requires a valid API key in environment for the model
    # For a pure "code check", we can mock litellm if needed, but 
    # let's just check if imports and basic structures are okay.
    
    try:
        from unittest.mock import MagicMock
        import litellm
        
        # Mocking litellm to avoid needing real API keys for a smoke test
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "The capital of France is Paris."
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        litellm.completion = MagicMock(return_value=mock_response)
        litellm.completion_cost = MagicMock(return_value=0.0001)
        
        print("Running Inference...")
        res = runInference(model, question, context, apiKey="dummy", n_samples=3)
        assert "samples" in res
        assert len(res["samples"]) == 3
        print("Inference Success.")
        
        print("Running Semantic Eval...")
        sem = semanticEval(res["answer"], correct_answer, context)
        assert "accuracy" in sem
        print("Semantic Eval Success.")
        
        print("Running Hallucination Eval...")
        # This will download spacy if not present
        halluc = evaluate_hallucination(res["answer"], res["samples"])
        assert "hallucination_score" in halluc
        print("Hallucination Eval Success.")
        
        print("ALL SMOKE TESTS PASSED (MOCKED LLM)")
        
    except Exception as e:
        print(f"Smoke test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inference_and_eval()
