"""Example: connect a supported Hugging Face vision-language model to NMR-Bench."""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


class HuggingFaceAdapter:
    def __init__(self, config):
        name = config["model_name"]
        self.device = config.get("device", "cuda:0")
        self.max_new_tokens = config.get("max_new_tokens", 16384)
        self.processor = AutoProcessor.from_pretrained(name, trust_remote_code=False)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                name, dtype=torch.bfloat16, trust_remote_code=False
            )
            .to(self.device)
            .eval()
        )

    @torch.inference_mode()
    def predict(self, image, question):
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": question}],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens)
        return self.processor.decode(
            outputs[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )


def create(config):
    return HuggingFaceAdapter(config)
