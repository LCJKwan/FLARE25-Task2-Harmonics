import torch
import json

from model.AttnUNet6 import AttnUNet6

# 1) Build/load your eager model
model = AttnUNet6(json.load(open('./model/model.json', 'r')))
model.load_state_dict(torch.load('./model/model.pth', map_location='cpu', weights_only=True))
model.eval().to('cpu')

# 3) Prefer scripting if your model has control flow; otherwise tracing is fine
example = torch.randn(1, 1, 256, 256, 64)
scripted = torch.jit.trace(model, example)

# 4) Freeze (inlines constants, prunes unused stuff)
frozen = torch.jit.freeze(scripted)

# 6) Save (one time)
frozen.save("model_cpu.pth")
