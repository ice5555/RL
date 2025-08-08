# models/generator.py
from transformers import (
    AutoConfig, AutoTokenizer,
    AutoModelForSeq2SeqLM, AutoModelForCausalLM,
    pipeline,
)
from transformers.utils import logging as hf_logging
import torch

class TextGenerator:
    def __init__(self, model_id="google/flan-t5-small", device=None, max_new_tokens=64, temperature=0.7):
        # 选择设备：优先 mps -> cuda -> cpu
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        hf_logging.set_verbosity_error()

        cfg = AutoConfig.from_pretrained(model_id)
        self.is_seq2seq = bool(getattr(cfg, "is_encoder_decoder", False))

        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.is_seq2seq:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            task = "text2text-generation"
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id)
            task = "text-generation"
            # GPT 类模型常没有 pad，用 eos 兜底，防止警告刷屏
            if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
                self.tok.pad_token = self.tok.eos_token
                self.model.config.pad_token_id = self.tok.eos_token_id

        # 挂到目标设备
        self.model = self.model.to(device)

        # pipeline 的 device：cuda 用 0/1…；mps 传 torch.device("mps")；cpu 用 -1
        if device == "cuda":
            pipe_device = 0
        elif device == "mps":
            pipe_device = torch.device("mps")
        else:
            pipe_device = -1

        # task 区分：只有 causal 才允许 return_full_text
        pipe_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        if not self.is_seq2seq:
            pipe_kwargs["return_full_text"] = False

        self.pipe = pipeline(
            task,
            model=self.model,
            tokenizer=self.tok,
            device=pipe_device,
            **pipe_kwargs,
        )

    
    def generate(self, prompt: str, **gen_kwargs) -> str:
        # 如果 temperature=0.0，则改为 greedy 解码
        if "temperature" in gen_kwargs and gen_kwargs["temperature"] == 0.0:
            gen_kwargs.pop("temperature")
            gen_kwargs["do_sample"] = False
        out = self.pipe(prompt, **gen_kwargs)[0]["generated_text"]
        return out.strip()    