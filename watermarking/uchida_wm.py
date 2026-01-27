from .base_wm import BaseWm
from models.base_model import BaseModel
from data.base_dataset import BaseDataset
from utils.visualizer import Visualizer
from datasets import Dataset as HFDataset
import torch
import torch.nn as nn
from typing import Dict
import numpy as np

class UchidaWM(BaseWm):
    
    def __init__(self, opt, modality=None, **kwargs):
        super().__init__(opt, modality=modality)

        self.opt = opt
        if kwargs:
            self.kwargs = kwargs
        
        if modality:
            self.model : BaseModel = modality[0]
            self.original_dataset : BaseDataset = modality[1]
            self.visualizer : Visualizer = modality[2]
        
        # Creation of the watermark
        M = self.size_of_M(self.model.hfmodel, self.opt.layer_name)
        T = self.opt.watermark_size
        self.watermark = torch.tensor(np.random.choice([0, 1], size=(T), p=[1. / 3, 2. / 3]))

        # Initialization of the projection matrix
        X = torch.randn((T, M), device=self.model.hfmodel.get_submodule(self.opt.layer_name).weight.device)
        #------------------ W -------------#
        # Normalization of each line of X
        # X = X / (torch.norm(X, dim=1, keepdim=True) + 1e-8)
        # print('min X: ', torch.min(X), 'max X: ', torch.max(X))
        #----------------------------------#
        self.X=X

    @staticmethod
    def modify_commandline_options(parser, isTrain):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument("--lambda_uchida", type=float, default=1., help='the weight applyed to the uchida loss')
        parser.add_argument("--layer_name", type=str, default='lm_head', help='define the layer on which to insert the mark')
        parser.add_argument("--watermark_size", type=int, default=32)
        return parser

    def size_of_M(self, net, weight_name):
        parameters = net.get_submodule(weight_name).weight
        print(f"Weight name: {weight_name}, size: {parameters.size()}")
        # For fully connected layers (nn.Linear)
        if len(parameters.size()) == 2:  # 2D Tensor for r nn.Linear
            return parameters.size()[1]
        # For convolutive layers (nn.Conv2d)
        elif len(parameters.size()) == 4:  # 4D Tensor for nn.Conv2d
            return parameters.size()[1] * parameters.size()[2] * parameters.size()[3]
        else:
            raise ValueError(f"Unsupported parameter shape for {weight_name}: {parameters.size()}")
    
    def insert(self):
        #overwrite the models base funtions
        self.model.set_input = self.new_set_input
        self.model.optimize_parameters = self.new_optimize_parameters

        #visualizer wandb loss plot modification
        self.visualizer.plot_current_loss = self.new_plot_current_loss

    def extract(self):
        pass

    def projection(self, X, w):
        sigmoid_func = nn.Sigmoid()
        res = torch.matmul(X, w)
        sigmoid = sigmoid_func(res)
        return sigmoid

    def flattened_weight(self, net, weights_name):
        parameters = net.get_submodule(weights_name).weight
        f_weights = torch.mean(parameters, dim=0)
        f_weights = f_weights.view(-1, )
        return f_weights
    
    def extraction(self, net, weights_name, X):
        W = self.flattened_weight(net, weights_name)
        return self.projection(X, W)
    
    def hamming(self, s1,s2):
        assert len(s1) == len(s2)
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    def new_set_input(self, input : HFDataset) -> None:
        if self.opt.isTrain:
            self.input = {k:v.to(self.model.hfmodel.device) for k, v in input.items()}
            self.model.input = self.input
    
    def _loss_step(self,
                   batch,
                   lambda_uchi,): #return loss
        hfmodel = self.model.hfmodel
        weights_name = self.opt.layer_name #layer name

        # out_model = hfmodel(input_ids=batch['input_ids'],
        #                     attention_mask=batch["attention_mask"],
        #                     lables=batch["labels"])
        out_model = hfmodel(**batch)
        
        watermark = self.watermark.float().to(self.model.hfmodel.get_submodule(self.opt.layer_name).weight.device)
        
        yj = self.extraction(hfmodel, weights_name, self.X)
        extraction_r = torch.round(yj) # <.5 = 0 and >.5 = 1
        diff = (~torch.logical_xor((extraction_r).cpu()>0, watermark.cpu()>0)) 

        bit_acc_avg = torch.sum(diff, dim=-1) / diff.shape[-1]
        loss_uchida = torch.nn.functional.binary_cross_entropy(yj, watermark, reduction='mean')


        loss_ce = out_model.loss.to(loss_uchida.device)
        loss_total = loss_ce + lambda_uchi*loss_uchida
        # print("ce", loss_ce)
        # print("uchi", loss_uchida)

        return {
            "uchida/loss_total" : loss_total,
            "uchida/loss_uchida" : loss_uchida,
            "uchida/loss_ce" : loss_ce,
            "uchida/bit_acc" : bit_acc_avg,
        }

    def new_optimize_parameters(self, total_steps) -> None:

        self.loss : Dict[str, torch.Tensor] = self._loss_step(batch=self.input,
                                                              lambda_uchi=self.opt.lambda_uchida)
        
        self.model.loss = self.loss
        self.loss["uchida/loss_total"].backward()

        for optimizer in self.model.optimizer:
            optimizer.step()
            optimizer.zero_grad()

    def new_plot_current_loss(self, losses : Dict[str, torch.Tensor], total_steps : int) -> None:
        """
        This function overides the visualizer.plot_current_losses(). And is ment to plot all the new losses on wanbd
        """
        self.visualizer.run.log({k: v.item() if torch.is_tensor(v) else float(v) for k, v in losses.items()}, step=total_steps)