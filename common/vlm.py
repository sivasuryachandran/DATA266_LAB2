"""Thin wrapper around Qwen2.5-VL-3B-Instruct, shared by Part 1 and Part 3.

One instance serves both text-only and image+text calls (greedy decoding, so runs
are reproducible). Weights load lazily on first use.
"""
import json
import re

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def extract_json(text: str):
    """Return the first JSON object found in model output, or None."""
    if not text:
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            depth += text[i] == "{"
            depth -= text[i] == "}"
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break
        start = text.find("{", start + 1)
    return None


class VLM:
    def __init__(self, model_id: str = DEFAULT_MODEL, dtype: str = "float16",
                 max_pixels: int = 640 * 28 * 28, min_pixels: int = 64 * 28 * 28, quant: str | None = None):
        self.model_id, self.dtype_name, self.quant = model_id, dtype, quant      # quant: None | "4bit" (bitsandbytes NF4)
        self.max_pixels, self.min_pixels = max_pixels, min_pixels
        self.model = None
        self.processor = None

    def _load(self, dtype_name=None):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        name = dtype_name or self.dtype_name
        self.dtype_name = name
        kw = {}
        if self.quant == "4bit":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=getattr(torch, name))
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=getattr(torch, name), attn_implementation="sdpa",
            device_map="cuda" if torch.cuda.is_available() else None, **kw)
        self.model.eval()
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, min_pixels=self.min_pixels, max_pixels=self.max_pixels)

    def unload(self):
        import gc
        import torch
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, prompt: str, images=None, system: str | None = None,
                 max_new_tokens: int = 256) -> str:
        import torch
        if self.model is None:
            self._load()
        images = list(images or [])
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        messages.append({"role": "user", "content": content})
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images or None, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device)
        try:
            with torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        except RuntimeError as e:
            # fp16 overflow on T4 shows up as inf/nan probabilities; retry once in bf16
            if "inf" in str(e) or "nan" in str(e):
                if self.dtype_name == "bfloat16":
                    raise
                self.unload()
                self._load("bfloat16")
                return self.generate(prompt, images, system, max_new_tokens)
            raise
        out = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    def generate_json(self, prompt: str, images=None, system: str | None = None,
                      max_new_tokens: int = 300, retries: int = 1):
        raw = ""
        for _ in range(retries + 1):
            raw = self.generate(prompt, images, system, max_new_tokens)
            obj = extract_json(raw)
            if obj is not None:
                return obj, raw
            prompt = prompt + "\n\nReturn ONLY a valid JSON object, nothing else."
        return None, raw


class TextLLM:
    """Text-only answerer (Qwen2.5-7B-Instruct by default). 4-bit NF4 so it fits a 16 GB T4 next to the VLM;
    set LLM_QUANT=none for bf16 on bigger GPUs. Same generate/generate_json interface as VLM (images ignored)."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-7B-Instruct", quant: str | None = None):
        import os
        self.model_id = model_id
        self.quant = quant or os.environ.get("LLM_QUANT", "4bit")
        self.model = None
        self.tok = None

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        kw = {}
        if self.quant == "4bit" and "bnb-4bit" in self.model_id:
            pass                                   # pre-quantised checkpoint: config ships with the weights
        elif self.quant == "4bit":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        else:
            kw["torch_dtype"] = torch.bfloat16
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="cuda", **kw).eval()

    def generate(self, prompt, images=None, system=None, max_new_tokens=256):
        import torch
        if self.model is None:
            self._load()
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                      do_sample=False, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    generate_json = VLM.generate_json