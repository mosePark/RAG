from vllm import LLM, SamplingParams

model_name = "facebook/opt-1.3b"
llm = LLM(model=model_name)

params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=30)
prompt = "What is the capital of South Korea?"

outputs = llm.generate([prompt], sampling_params=params)

print("=== 출력 결과 ===")
print(outputs[0].outputs[0].text)
