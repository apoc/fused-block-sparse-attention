import torch
from qwen_blocksparse import SelectTwoStage, SelectLastTok, SelectMax
for dt in (torch.float32, torch.bfloat16):
    q = torch.randn(1, 16, 512, 32).to(dt)
    k = torch.randn(1, 2, 512, 32).to(dt)
    ref = SelectMax(2, 128).select(q, k)
    for name, S in [("lasttok", SelectLastTok(2, 128)), ("twostage", SelectTwoStage(2, 128, over=4))]:
        a = S.select(q, k)
        assert a.shape == ref.shape and a.dtype == torch.long, (dt, name, a.shape)
        assert (a >= 0).all() and (a < 4).all()
    print(f"{dt}: ok")
print("SMOKE_OK")
