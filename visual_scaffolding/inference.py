"""Greedy decoding with model-selected visual evidence."""

from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image

from .model import load_checkpoint
from .prompt import SYSTEM_PROMPT


@dataclass
class Reference:
    position: int
    indices: list[int]
    grid_shape: tuple[int, int]


@dataclass
class Generation:
    text: str
    token_ids: list[int]
    references: list[Reference] = field(default_factory=list)
    stop_reason: str = "max_new_tokens"
    rejected_evidence_count: int | None = None


def incremental_positions(previous, count, device):
    positions = torch.arange(previous + 1, previous + count + 1, dtype=torch.long, device=device)
    return positions.view(1, 1, -1).expand(3, 1, -1).clone()


def repeated_token_run(tokens, excluded, limit=32):
    if len(tokens) < limit:
        return False
    previous, count = None, 0
    for token in reversed(tokens):
        if token in excluded or token == 0:
            continue
        if previous is None:
            previous, count = token, 1
        elif token == previous:
            count += 1
        else:
            break
        if count >= limit:
            return True
    return False


class VisualScaffolding:
    """Inference API for the supplied Path Tracing or Burden Counting checkpoint."""

    def __init__(self, checkpoint, device="cuda:0", max_new_tokens=16384):
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.model, self.processor, self.tokenizer, self.token_ids = load_checkpoint(
            checkpoint, self.device
        )
        self.request_id = self.token_ids["request"]
        self.evidence_id = self.token_ids["evidence"]
        self.end_id = self.token_ids["end"]
        self.eos_id = int(self.tokenizer.eos_token_id)
        end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.im_end_id = int(end) if end is not None else self.eos_id
        self.excluded_tokens = {
            self.request_id,
            self.evidence_id,
            self.end_id,
            self.eos_id,
            self.im_end_id,
        }
        self.context_window = 256
        self.evidence_cap = 8
        self.threshold = 0.5

    def _inputs(self, inputs):
        result = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                result[key] = (
                    value.to(device=self.device, dtype=torch.bfloat16)
                    if key == "pixel_values"
                    else value.to(self.device)
                )
            else:
                result[key] = value
        if "mm_token_type_ids" not in result and "input_ids" in result:
            result["mm_token_type_ids"] = torch.zeros_like(result["input_ids"], dtype=torch.int32)
        return result

    def predict(self, image: Image.Image, question: str) -> str:
        """Return the complete generated response for the benchmark interface."""
        return self.generate(image, question).text

    @torch.inference_mode()
    def generate(self, image: Image.Image | str | Path, question: str) -> Generation:
        if self.max_new_tokens == 0:
            return Generation("", [])
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                image = opened.convert("RGB")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": question}],
            },
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if not prompt.endswith("<think>\n"):
            prompt += "<think>\n"
        inputs = self._inputs(self.processor(text=[prompt], images=[image], return_tensors="pt"))
        attention_mask = inputs["attention_mask"]
        positions = self.model.model.compute_3d_position_ids(
            input_ids=inputs["input_ids"],
            inputs_embeds=None,
            image_grid_thw=inputs["image_grid_thw"],
            attention_mask=attention_mask,
            mm_token_type_ids=inputs["mm_token_type_ids"],
        )
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=attention_mask,
            position_ids=positions,
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            mm_token_type_ids=inputs["mm_token_type_ids"],
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        cache = outputs.past_key_values
        width, height = image.size
        enlarged = image.resize((width * 2, height * 2), Image.Resampling.LANCZOS)
        dense_inputs = self._inputs(
            self.processor(
                text=["<|vision_start|><|image_pad|><|vision_end|>"],
                images=[enlarged],
                return_tensors="pt",
            )
        )
        dense = self.model.model.get_image_features(
            pixel_values=dense_inputs["pixel_values"],
            image_grid_thw=dense_inputs["image_grid_thw"],
            return_dict=True,
        ).pooler_output[0]
        grid = (
            int(dense_inputs["image_grid_thw"][0, 1]) // 2,
            int(dense_inputs["image_grid_thw"][0, 2]) // 2,
        )
        if dense.shape[0] != grid[0] * grid[1]:
            raise ValueError("Dense visual bank and grid shape disagree")
        keys = self.model.retrieval_module.encode_visual_keys(dense, grid)
        history = list(outputs.hidden_states[-1][0].detach()[-257:].unbind(0))
        next_token = int(torch.argmax(outputs.logits[0, -1]).item())
        position = int(positions[0, 0, -1].item())
        tokens, references = [], []
        stop, rejected = "max_new_tokens", None

        while len(tokens) < self.max_new_tokens:
            step_positions = incremental_positions(position, 1, self.device)
            attention_mask = torch.cat(
                [attention_mask, torch.ones(1, 1, dtype=attention_mask.dtype, device=self.device)],
                dim=1,
            )
            step = self.model(
                input_ids=torch.tensor([[next_token]], dtype=torch.long, device=self.device),
                attention_mask=attention_mask,
                position_ids=step_positions,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            cache = step.past_key_values
            hidden = step.hidden_states[-1][0, -1].detach()
            history.append(hidden)
            history = history[-257:]
            tokens.append(next_token)
            position += 1
            if next_token in (self.eos_id, self.im_end_id) and next_token != self.request_id:
                stop = "eos"
                break

            if next_token == self.request_id:
                # The current request forms the query and is excluded from its context.
                context = torch.stack(history[-257:-1], dim=0) if len(history) > 1 else None
                module = self.model.retrieval_module
                dtype = module.context_agg.q_proj.weight.dtype
                h = hidden.to(device=module.context_agg.q_proj.weight.device, dtype=dtype)
                if context is not None:
                    context = context.to(device=h.device, dtype=dtype)
                scores = module.score_precomputed_keys(h, keys, context)["scores"].detach().float()
                selected = scores >= self.threshold
                count = int(selected.sum().item())
                if count == 0:
                    indices = [int(torch.argmax(scores).item())]
                elif count > self.evidence_cap:
                    indices = sorted(torch.topk(scores, self.evidence_cap).indices.tolist())
                else:
                    indices = sorted(selected.nonzero(as_tuple=False).flatten().tolist())
                count = len(indices)
                # Commit the request only when its complete evidence block fits.
                if count + 1 > self.max_new_tokens - len(tokens):
                    tokens.pop()
                    rejected = count
                    break
                visual = dense.to(device=h.device, dtype=dtype)
                evidence = self.model.evidence_mlp(
                    visual[torch.tensor(indices, device=visual.device)]
                )
                end = self.model.get_input_embeddings()(
                    torch.tensor([self.end_id], device=self.device)
                )[0]
                end = end.to(device=evidence.device, dtype=evidence.dtype)
                block = torch.cat([evidence, end.unsqueeze(0)], dim=0).unsqueeze(0)
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(1, count + 1, dtype=attention_mask.dtype, device=self.device),
                    ],
                    dim=1,
                )
                block_output = self.model(
                    input_ids=None,
                    inputs_embeds=block,
                    attention_mask=attention_mask,
                    position_ids=incremental_positions(position, count + 1, self.device),
                    past_key_values=cache,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                cache = block_output.past_key_values
                history.extend(block_output.hidden_states[-1][0].detach().unbind(0))
                history = history[-257:]
                references.append(Reference(len(tokens) - 1, indices, grid))
                tokens.extend([self.evidence_id] * count + [self.end_id])
                position += count + 1
                if len(tokens) >= self.max_new_tokens:
                    break
                next_token = int(torch.argmax(block_output.logits[0, -1]).item())
                continue

            if len(tokens) >= self.max_new_tokens:
                break
            next_token = int(torch.argmax(step.logits[0, -1]).item())
            if repeated_token_run(tokens, self.excluded_tokens):
                stop = "degeneration"
                break
        return Generation(
            self.tokenizer.decode(tokens, skip_special_tokens=False),
            tokens,
            references,
            stop,
            rejected,
        )
