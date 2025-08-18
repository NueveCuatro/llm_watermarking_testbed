from .base_model import BaseModel
from transformers import AutoModelForCausalLM
import torch
import networks

class CausalLMModel(BaseModel):

    def __init__(self, opt):
        super().__init__(opt)

        self.model = AutoModelForCausalLM.from_pretrained(
            opt.model_name_or_path,
            device_map=opt.device_map_bool,
            torch_dtype=torch.float16
        )

        networks.freeze_model(self.model,
                              num_freezed_layers=opt.get("num_freezed_layers", None),
                              specific_layer_name=opt.get("specefic_layer_name", None),
                              freeze_embeddings=opt.get("freeze_embedding", False),
                              freeze_all=opt.get("freeze_all", False)
                              )

        self.optimizer = #TODO get the list of optimizer and use as parameters, (p for p in self.model.parameters() if p.requires_grad == True)