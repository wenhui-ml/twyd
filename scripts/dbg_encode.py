"""Debug: reproduce swift template.encode on a twyd_pt row."""
import sys

sys.path.insert(0, "/home/chenwh/Papers/twyd/_upstream/ms-swift")
sys.path.insert(0, "/home/chenwh/Papers/twyd")

from swift.model import get_model_processor
from swift.template import get_template

MODEL = "/home/chenwh/models/Llama-3.2-1B-Instruct"

_, processor = get_model_processor(MODEL, load_model=False)
template = get_template(processor)
template.use_chat_template = False
template.set_mode('train')
template._loss_scale = 'all'
print("template:", type(template).__name__, "use_chat_template:", template.use_chat_template,
      "is_training:", template.is_training)

row = {"messages": [{"role": "assistant", "content": "The quick brown fox jumps over the lazy dog. " * 10}]}
try:
    out = template.encode(row, return_length=True)
    print("encode OK, keys:", list(out.keys()) if isinstance(out, dict) else type(out))
    if isinstance(out, dict):
        print("input_ids len:", len(out.get("input_ids", [])))
        print("labels len:", len(out.get("labels", [])) if out.get("labels") else None)
        print("lengths:", out.get("lengths"))
except Exception as e:
    import traceback
    traceback.print_exc()
    print("ENCODE FAILED:", e)
