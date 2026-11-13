import argparse
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

parser = argparse.ArgumentParser()
parser.add_argument("--video", help="path to an mp4 file")
parser.add_argument("--prompt", help="question to ask about the video")
parser.add_argument("--fps", type=float, default=2.0)
args = parser.parse_args()

model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

# load the processor and the model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_name)

# prepare the model input
messages = [{
    "role": "user",
    "content": [
        {"type": "video", "video": args.video, "fps": args.fps},
        {"type": "text", "text": args.prompt},
    ],
}]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos = process_vision_info(messages)
model_inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=2048
)

output_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
content = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

print("content:", content)