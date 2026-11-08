"""Lazy, local Gemma activation steering; no model downloads on import."""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTION = ROOT / 'refusal_direction_gen/refusal_direction/pipeline/runs/gemma-2-2b-it'


class SteeringEngine:
    def __init__(self, model_id, direction_dir):
        self.model_id = model_id
        self.direction_dir = Path(direction_dir)
        self.model = None

    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        metadata = json.loads((self.direction_dir / 'direction_metadata.json').read_text())
        direction = torch.load(self.direction_dir / 'direction.pt', map_location='cpu', weights_only=True).flatten()
        if not torch.isfinite(direction).all():
            raise ValueError('Direction contains non-finite values.')
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map='auto',
        )
        model.eval()
        if direction.numel() != model.config.hidden_size:
            raise ValueError('Direction size does not match model hidden size.')
        layer = metadata['layer']
        if not isinstance(layer, int) or not 0 <= layer < len(model.model.layers):
            raise ValueError('Invalid steering layer in direction_metadata.json.')
        self.tokenizer, self.direction, self.layer = tokenizer, direction, layer
        self.model = model

    def generate(self, prompt, alpha, max_tokens):
        import torch
        self.load()
        inputs = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], add_generation_prompt=True,
            return_tensors='pt', return_dict=True,
        ).to(self.model.get_input_embeddings().weight.device)
        def hook(module, args):
            hidden = args[0]
            return (hidden + alpha * self.direction.to(device=hidden.device, dtype=hidden.dtype), *args[1:])
        handle = None
        started = time.perf_counter()
        try:
            if alpha != 0:
                handle = self.model.model.layers[self.layer].register_forward_pre_hook(hook)
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                             pad_token_id=self.tokenizer.eos_token_id)
            ids = output[0, inputs['input_ids'].shape[-1]:]
            eos = self.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            return dict(response=self.tokenizer.decode(ids, skip_special_tokens=True),
                        tokens=len(ids), seconds=round(time.perf_counter() - started, 3),
                        truncated=len(ids) >= max_tokens and int(ids[-1]) not in eos)
        finally:
            if handle is not None:
                handle.remove()


class DemoEngine:
    def load(self):
        pass

    def generate(self, prompt, alpha, max_tokens):
        time.sleep(0.25)
        return dict(response=f'DEMO ONLY · alpha {alpha:g}\n\nYour prompt: {prompt}\n\n'
                    'This is a placeholder response for testing the comparison interface. '
                    'No model was called and no quality or refusal verdict is implied.',
                    tokens=None, seconds=0.25, truncated=False)
