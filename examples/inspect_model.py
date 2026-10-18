"""No training, downloads, language inference or concrete environment imports."""
import json
import torch
from survival_agent.agent import HistoryAgent
from survival_agent.integrated import LanguageLatentCore

def main():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(0)
        model=HistoryAgent().eval()
        logits,value=model.predict(model.represent(torch.zeros(1,4,5,5,5)))
        core=LanguageLatentCore(feature_width=3,seed=0).eval()
        other_logits,other_value=core.predict(core.represent(torch.zeros(1,3)))
    assert logits.shape==other_logits.shape==(1,5)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    assert torch.isfinite(other_logits).all() and torch.isfinite(other_value).all()
    print(json.dumps({'policy_shape':list(logits.shape),'feature_width':3,'optimizer_updates':0,'trained_result':False}))

if __name__=='__main__':main()
