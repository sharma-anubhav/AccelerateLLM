from threading import Thread
import torch

class DraftGenerator:
    def __init__(self, draft_models, tokenizer):
        self.draft_models = draft_models
        self.tokenizer = tokenizer

    def _gen_one(self, model, ids, mask, draft_len, out, key):
        dev = next(model.parameters()).device
        o = model.generate(ids.to(dev),
                           attention_mask=mask.to(dev),
                           max_new_tokens=draft_len,
                           do_sample=False,
                           pad_token_id=self.tokenizer.pad_token_id,
                           early_stopping=True,
                           )
        out[key] = o[0, ids.size(1):]

    def generate_drafts(self, prefix_ids, prefix_mask, draft_len=5):
        outs = {n: None for n in self.draft_models}
        ths = []
        for n, m in self.draft_models.items():
            t = Thread(target=self._gen_one,
                       args=(m, prefix_ids, prefix_mask, draft_len, outs, n))
            t.start(); ths.append(t)
        for t in ths: t.join()
        return outs

class Verifier:
    def __init__(self, target_model, tokenizer):
        self.target_model = target_model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def verify_sequence(self, prefix_ids, draft_ids):
        dev = next(self.target_model.parameters()).device
        p = prefix_ids.to(dev)
        d = draft_ids.to(dev)
        x = torch.cat([p, d], dim=1)
        attn = torch.ones_like(x, device=dev)
        logits = self.target_model(x, attention_mask=attn).logits

        pref = p.size(1)
        acc, corr = 0, None
        for i in range(d.size(1)):
            pred = int(logits[0, pref + i - 1].argmax())
            d_id = int(d[0, i])
            if pred == d_id:
                acc += 1
            else:
                corr = pred
                break
        return acc, corr