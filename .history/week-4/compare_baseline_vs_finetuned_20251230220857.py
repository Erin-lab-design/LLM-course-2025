"""
Compare Mental Health Counseling Models: Baseline vs Fine-tuned
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
import torch

# configurations
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Baseline model
FINETUNED_MODEL_NAME = "Erin-lab-design/TinyLlama-1.1B-Chat-v1.0"  # fine-tuned model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("="*80)
print("🧠 Mental Health Model Comparison: Baseline vs Fine-tuned")
print("="*80)
print(f"Baseline Model: {BASE_MODEL_NAME}")
print(f"Fine-tuned Model: {FINETUNED_MODEL_NAME}")
print(f"Device: {DEVICE}")
print()

# config for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

def load_model(model_name, model_type="baseline"):
    """load model and tokenizer"""
    print(f"\n{'='*80}")
    print(f"Loading {model_type.upper()} model: {model_name}")
    print(f"{'='*80}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print(f"✅ {model_type.capitalize()} model loaded!\n")
    return model, tokenizer

def ask_model(model, tokenizer, question: str, model_name: str):
    model.eval()
    
    # to keep prompt format consistent with training
    system_prompt = "You are an empathetic mental health counselor. Please provide advice and support."
    
    B_INST, E_INST = "### Instruction:\n", "\n\n### Response:\n"
    full_prompt = f"{system_prompt}\n\n{B_INST}{question.strip()}{E_INST}"
    
    # emcoding input
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    print(f"\n{'='*80}")
    print(f"📝 {model_name}")
    print(f"{'='*80}")
    print(f"Question: {question}\n")
    print("Response:")
    print("-" * 80)
    
    # streaming generation
    streamer = TextStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            streamer=streamer,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # get full response text
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in full_response:
        answer = full_response.split("### Response:")[-1].strip()
    else:
        answer = full_response
    
    print("\n" + "="*80 + "\n")
    return answer

TEST_QUESTION = """I have an important exam tomorrow, but I'm so nervous that I can't fall asleep. 
My mind keeps racing with worries about failing. I've studied hard, but I still feel anxious. 
What should I do to calm down and get some rest?"""

print("\n" + "="*80)
print("🧪 COMPARISON TEST")
print("="*80)
print(f"\nTest Question (Anxiety for an exam):\n\"{TEST_QUESTION}\"\n")
print("="*80)

# load Baseline model
baseline_model, baseline_tokenizer = load_model(BASE_MODEL_NAME, "baseline")

# load Fine-tuned model
finetuned_model, finetuned_tokenizer = load_model(FINETUNED_MODEL_NAME, "fine-tuned")

# test Baseline model
baseline_answer = ask_model(
    baseline_model, 
    baseline_tokenizer, 
    TEST_QUESTION,
    "BASELINE MODEL"
)

# clean GPU storage
del baseline_model
torch.cuda.empty_cache()

# test Fine-tuned model
finetuned_answer = ask_model(
    finetuned_model, 
    finetuned_tokenizer, 
    TEST_QUESTION,
    "FINE-TUNED MODEL"
)

# 对比总结
print("\n" + "="*80)
print("📊 COMPARISON SUMMARY")
print("="*80)

print("\n🔵 BASELINE MODEL RESPONSE:")
print("-" * 80)
print(baseline_answer[:500] + "..." if len(baseline_answer) > 500 else baseline_answer)

print("\n🟢 FINE-TUNED MODEL RESPONSE:")
print("-" * 80)
print(finetuned_answer[:500] + "..." if len(finetuned_answer) > 500 else finetuned_answer)

print("\n" + "="*80)
print("💡 OBSERVATIONS TO CHECK:")
print("="*80)

print("="*80)
print("✅ Comparison Complete!")
print("="*80)

# --- SAVE RESULTS ---
with open("comparison_results.txt", "w", encoding="utf-8") as f:
    f.write("="*80 + "\n")
    f.write("MENTAL HEALTH MODEL COMPARISON\n")
    f.write("="*80 + "\n\n")
    f.write(f"Question:\n{TEST_QUESTION}\n\n")
    f.write("="*80 + "\n")
    f.write("BASELINE MODEL RESPONSE:\n")
    f.write("="*80 + "\n")
    f.write(baseline_answer + "\n\n")
    f.write("="*80 + "\n")
    f.write("FINE-TUNED MODEL RESPONSE:\n")
    f.write("="*80 + "\n")
    f.write(finetuned_answer + "\n")

print("\n💾 Results saved to: comparison_results.txt")
print("\n🎯 Next Steps:")
print("1. 仔细阅读两个模型的回答")
print("2. 根据上述 5 个观察点进行评估")
print("3. 将你的实际观察结果更新到 Final_report.md 的 Week 4 部分")
print("4. 替换或补充我之前写的假设性评价")
