"""Model hub: one shared VLM, CLIP, and ONE diffusion pipeline resident at a time (16 GB budget)."""
import os

SD_BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
IP2P = "timbrooks/instruct-pix2pix"


class Hub:
    def __init__(self, vlm, lora=None, device="cuda"):
        """lora: local path or HF repo id of the Part 2 adapter (env P3_LORA if not given)."""
        self.vlm = vlm
        self.lora = lora or os.environ.get("P3_LORA")
        self.device = device
        self._clip = None
        self._pipe, self._pipe_name = None, None
        self._lora_loaded = False

    @property
    def clip(self):
        if self._clip is None:
            from p2_finetune.evaluate import Clip
            self._clip = Clip()
        return self._clip

    def _free(self):
        import gc
        import torch
        self._pipe, self._pipe_name, self._lora_loaded = None, None, False
        gc.collect()
        torch.cuda.empty_cache()

    def sd(self, domain: bool):
        """Return (txt2img, img2img) sharing weights; optionally with the Part 2 LoRA applied."""
        import torch
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
        if self._pipe_name != "sd":
            self._free()
            p = StableDiffusionPipeline.from_pretrained(SD_BASE, torch_dtype=torch.float16).to(self.device)
            p.scheduler = DPMSolverMultistepScheduler.from_config(p.scheduler.config)
            p.set_progress_bar_config(disable=True)
            self._pipe, self._pipe_name = p, "sd"
        p = self._pipe
        if domain and not self._lora_loaded:
            if not self.lora:
                raise RuntimeError("Part 2 LoRA not configured (set P3_LORA to a local path or Hugging Face repo id).")
            p.load_lora_weights(self.lora)
            self._lora_loaded = True
        elif not domain and self._lora_loaded:
            p.unload_lora_weights()
            self._lora_loaded = False
        return p, StableDiffusionImg2ImgPipeline(**p.components)

    def ip2p(self):
        import torch
        from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline
        if self._pipe_name != "ip2p":
            self._free()
            p = StableDiffusionInstructPix2PixPipeline.from_pretrained(IP2P, torch_dtype=torch.float16, safety_checker=None).to(self.device)
            p.scheduler = EulerAncestralDiscreteScheduler.from_config(p.scheduler.config)
            p.set_progress_bar_config(disable=True)
            self._pipe, self._pipe_name = p, "ip2p"
        return self._pipe
