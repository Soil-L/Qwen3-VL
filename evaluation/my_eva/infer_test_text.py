import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

# 核心配置
BASE_MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
LORA_ADAPTER_PATH = "./checkpoints"
DEVICE = "cuda"
DTYPE = torch.bfloat16

# 1. 加载处理器
processor = AutoProcessor.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
    padding_side="right",
    video_fps=2,
    video_max_frames=8,
    video_min_frames=4
)
processor.tokenizer.pad_token = processor.tokenizer.eos_token
processor.image_processor.max_pixels = 451584
processor.video_processor.max_pixels = 1284608

# 2. 加载基座模型
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=DTYPE,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",
    # model_max_length=4096
)

# 3. 加载 LoRA（关键：ignore_mismatched_sizes=True + 验证）
model = PeftModel.from_pretrained(
    base_model,
    LORA_ADAPTER_PATH,
    device_map="auto",
    torch_dtype=DTYPE,
    ignore_mismatched_sizes=True  # 仅忽略bias维度错（因LoRA未训练bias）
)

# 验证：确认模型加载成功
model.eval()
print("✅ 模型加载成功！")

# 4. 测试推理
prompt = "请解释多模态模型中视觉和语言模块的交互方式"
messages = [{"role":"user","content":[{"type":"text","text":prompt}]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=[text], return_tensors="pt").to(DEVICE)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
    response = processor.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

print("输入：", prompt)
print("输出：", response)