"""
Compare Mental Health Counseling Models: Baseline vs Fine-tuned
===============================================================
测试同一个问题在 baseline 模型和你训练后的模型上的回答差异
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
import torch

# 配置
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Baseline 模型
FINETUNED_MODEL_NAME = "Erin-lab-design/TinyLlama-1.1B-Chat-v1.0"  # 你训练的模型
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("="*80)
print("🧠 Mental Health Model Comparison: Baseline vs Fine-tuned")
print("="*80)
print(f"Baseline Model: {BASE_MODEL_NAME}")
print(f"Fine-tuned Model: {FINETUNED_MODEL_NAME}")
print(f"Device: {DEVICE}")
print()

# 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

def load_model(model_name, model_type="baseline"):
    """加载模型和 tokenizer"""
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
    """向模型提问并获取回答"""
    # 设置模型为评估模式
    model.eval()
    
    # 格式化 prompt（与训练时一致）
    system_prompt = "You are an empathetic mental health counselor. Please provide advice and support."
    
    B_INST, E_INST = "### Instruction:\n", "\n\n### Response:\n"
    full_prompt = f"{system_prompt}\n\n{B_INST}{question.strip()}{E_INST}"
    
    # 编码输入
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    print(f"\n{'='*80}")
    print(f"📝 {model_name}")
    print(f"{'='*80}")
    print(f"Question: {question}\n")
    print("Response:")
    print("-" * 80)
    
    # 流式输出
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
    
    # 获取完整回答（用于保存）
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in full_response:
        answer = full_response.split("### Response:")[-1].strip()
    else:
        answer = full_response
    
    print("\n" + "="*80 + "\n")
    return answer

# 测试问题：你的真实场景 - 考试前紧张睡不着
TEST_QUESTION = """I have an important exam tomorrow, but I'm so nervous that I can't fall asleep. 
My mind keeps racing with worries about failing. I've studied hard, but I still feel anxious. 
What should I do to calm down and get some rest?"""

print("\n" + "="*80)
print("🧪 COMPARISON TEST")
print("="*80)
print(f"\nTest Question (考试焦虑场景):\n\"{TEST_QUESTION}\"\n")
print("="*80)

# 加载 Baseline 模型
baseline_model, baseline_tokenizer = load_model(BASE_MODEL_NAME, "baseline")

# 加载 Fine-tuned 模型
finetuned_model, finetuned_tokenizer = load_model(FINETUNED_MODEL_NAME, "fine-tuned")

# 测试 Baseline 模型
baseline_answer = ask_model(
    baseline_model, 
    baseline_tokenizer, 
    TEST_QUESTION,
    "BASELINE MODEL (原始未训练)"
)

# 清理 GPU 内存
del baseline_model
torch.cuda.empty_cache()

# 测试 Fine-tuned 模型
finetuned_answer = ask_model(
    finetuned_model, 
    finetuned_tokenizer, 
    TEST_QUESTION,
    "FINE-TUNED MODEL (你训练的心理健康模型)"
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
print("""
请观察以下几个方面的差异：

1. 同理心 (Empathy)
   - Baseline: 是否承认了你的紧张和焦虑情绪？
   - Fine-tuned: 是否表现出更多的理解和共情？
   
2. 建议的实用性 (Practical Advice)
   - Baseline: 建议是否具体可行（如呼吸练习、放松技巧）？
   - Fine-tuned: 建议是否更加详细和针对考试焦虑？
   
3. 专业语气 (Professional Tone)
   - Baseline: 语气是否像一个心理咨询师？
   - Fine-tuned: 是否有改进（更温暖、更专业）？
   
4. 回答深度 (Depth of Response)
   - Baseline: 是否泛泛而谈（如"放轻松"）？
   - Fine-tuned: 是否深入分析了考试焦虑的原因和应对方法？
   
5. 回答结构 (Response Structure)
   - Baseline: 回答是否有条理？
   - Fine-tuned: 是否有更清晰的结构（如步骤1、2、3）？

根据你的实际观察结果，你可以更新 Final_report.md 中对训练效果的描述。
""")

print("="*80)
print("✅ Comparison Complete!")
print("="*80)

# 保存结果到文件
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
