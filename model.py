import copy

import torch
import torch.nn as nn
from collections import OrderedDict
# from pytorch_transformers import BertModel, BertConfig
# from pytorch_transformers import BertModel, BertConfig
from transformers import BertModel, BartModel, BartForConditionalGeneration
from myProj.bert import BertConfig
from torch.nn.init import xavier_uniform_

from myProj.decoder import TransformerDecoder
from myProj.encoder import Classifier, ExtTransformerEncoder
from myProj.optimiser import Optimizer

def build_optimiser(args,model,checkpt):

    if checkpt==None:
        return
    if args.new_optim==False:
        if args.few_shot==False:
                optim = checkpt['optim'][0]
                saved_optimizer_state_dict = optim.optimizer.state_dict()
                optim.optimizer.load_state_dict(saved_optimizer_state_dict)
                if args.visible_gpus != '-1':
                    for state in optim.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

                if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
                    raise RuntimeError(
                        "Error: loaded Adam optimizer from existing model" +
                        " but optimizer state is empty")

        else:
                optim = Optimizer(
                    args.optim, args.lr, args.max_grad_norm,
                    beta1=args.beta1, beta2=args.beta2,
                    decay_method='noam',
                    warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))


    return optim


class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False, bart=False):
    # def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if(large):
            self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
        else:
            # self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)
            if bart:
                self.model = BartModel.from_pretrained('/home/ybai/downloads/bart', cache_dir=temp_dir, local_files_only=True)
                # self.model = BartForConditionalGeneration.from_pretrained('/home/ybai/downloads/bart', cache_dir=temp_dir, local_files_only=True)
            else:
                self.model = BertModel.from_pretrained('bert-base-multilingual-uncased', cache_dir=temp_dir,
                                                       local_files_only=False)

        self.finetune = finetune

    def forward(self, x, segs, mask):

        if(self.finetune):
            top_vec, _ = self.model(x, attention_mask=mask, token_type_ids=segs)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, attention_mask=mask, token_type_ids=segs)
        return top_vec

class ExtSummr(nn.Module):
    def __init__(self,args,device,checkpt) -> None:
        super(ExtSummr,self).__init__()
        self.args=args
        self.device=device
        self.model=Bert(args.large,args.temp_dir,args.finetune_bert)

        self.ext=ExtTransformerEncoder(self.model.model.config.hidden_size,2048,8,0.2,6)

        if(checkpt is not None):
            self.load_state_dict(checkpt['model'],strict=True)


    def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls
