import torch
from transformers import Qwen3_5ForConditionalGeneration

name = "Qwen/Qwen3.5-0.8B-Base"
m = Qwen3_5ForConditionalGeneration.from_pretrained(name, dtype=torch.bfloat16)

print("children:", [n for n, _ in m.named_children()])
print("model children:", [n for n, _ in m.model.named_children()])
print("has lm_head:", hasattr(m, "lm_head"), type(getattr(m, "lm_head", None)).__name__)
if hasattr(m, "lm_head"):
    tied = m.lm_head.weight.data_ptr() == m.get_input_embeddings().weight.data_ptr()
    print("lm_head weight:", tuple(m.lm_head.weight.shape), "tied:", tied)


def count(mod):
    return sum(p.numel() for p in mod.parameters())


tot = count(m)
vis = count(m.model.visual) if hasattr(m.model, "visual") else 0
mtp = count(m.mtp) if hasattr(m, "mtp") else 0
lang = count(m.model.language_model)
print(f"total {tot/1e9:.3f}B | visual {vis/1e6:.0f}M | mtp {mtp/1e6:.0f}M | lang {lang/1e9:.3f}B")
emb = m.get_input_embeddings().weight.numel()
print(f"embedding {emb/1e6:.0f}M ({100*emb/tot:.0f}% of total)")
print(f"trainable if visual+mtp dropped: {(tot-vis-mtp)/1e9:.3f}B")
print("supports grad ckpt:", m.supports_gradient_checkpointing)
print("lang children:", [n for n, _ in m.model.language_model.named_children()])
