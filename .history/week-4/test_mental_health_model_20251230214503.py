"""
Test Mental Health Counseling Model
====================================
测试微调后的 TinyLlama 心理健康咨询模型
测试真实场景：考试焦虑、睡眠问题等
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
import torch

# 配置
MODEL_NAME = "Erin-lab-design/TinyLlama-1.1B-Chat-v1.0"  # 你的微调模型
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("="*70)
print("🧠 Mental Health Counseling Model Test")
print("="*70)
print(f"Model: {MODEL_NAME}")
print(f"Device: {DEVICE}")
print()

# 加载模型
print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("✅ Model loaded!\n")

def ask_counselor(question: str, show_streaming=True):
    """
    向心理咨询模型提问
    
    Args:
        question: 用户的问题
        show_streaming: 是否显示流式输出
    """
    # 设置模型为评估模式
    model.eval()
    
    # 格式化 prompt（与训练时一致）
    system_prompt = "You are an empathetic mental health counselor. Please provide advice and support."
    
    B_INST, E_INST = "### Instruction:\n", "\n\n### Response:\n"
    full_prompt = f"{system_prompt}\n\n{B_INST}{question.strip()}{E_INST}"
    
    # 编码输入
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    print(f"Question: {question}\n")
    print("Counselor's Response:")
    print("-" * 70)
    
    if show_streaming:
        # 流式输出
        streamer = TextStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        
        with torch.inference_mode():
            _ = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                streamer=streamer,
                eos_token_id=tokenizer.eos_token_id,
            )
    else:
        # 一次性输出
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 只打印 Response 部分
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        print(response)
    
    print("\n" + "="*70 + "\n")

# 测试场景：考试焦虑和睡眠问题
print("🧪 Testing with Real Mental Health Scenarios")
print("="*70 + "\n")

# 场景 1：考试前焦虑失眠
print("📝 Scenario 1: Pre-Exam Anxiety & Insomnia")
print("="*70)
ask_counselor(
    "I have an important exam tomorrow, but I'm so nervous that I can't fall asleep. "
    "My mind keeps racing with worries about failing. What should I do?"
)

# 场景 2：长期压力导致的睡眠问题
print("📝 Scenario 2: Chronic Stress & Sleep Issues")
print("="*70)
ask_counselor(
    "I've been under a lot of stress lately and I can't sleep well at night. "
    "I keep waking up worrying about my responsibilities. How can I improve my sleep?"
)

# 场景 3：考试焦虑的应对策略
print("📝 Scenario 3: Coping with Test Anxiety")
print("="*70)
ask_counselor(
    "I always feel extremely anxious before exams, even when I've studied well. "
    "My heart races and I can't think clearly during tests. What coping strategies can help?"
)

# 场景 4：一般性焦虑
print("📝 Scenario 4: General Anxiety")
print("="*70)
ask_counselor(
    "I often feel anxious for no clear reason. It's affecting my daily life and relationships. "
    "What should I do?"
)

print("\n" + "="*70)
print("✅ Test Complete!")
print("="*70)
print("\n💡 Observations:")
print("- Check if responses show empathy and understanding")
print("- Evaluate if advice is practical and helpful")
print("- Note any generic or off-topic responses")
print("- Compare with your expectations for a mental health counselor")
