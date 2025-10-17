import wandb

class Visualizer:
    """
    This class is intended to track metrics, and training states for different experiments
    """

    def __init__(self, opt):

        self.opt = opt

        self.use_wandb = opt.use_wandb

        if self.use_wandb:
            self.name = opt.name
            self.wandb_project_name = opt.wandb_project_name
            
            if opt.isTrain:
                config = {
                    "model" : opt.model_name_or_path,
                    "model_type" : opt.model,
                    "watermarking_technique" : getattr(opt, "wm", None),
                    "dataset" : opt.dataset_name,
                    "dataset_config" : opt.dataset_config_name,
                    "epochs" : opt.n_epochs,
                    "batch_size" : opt.batch_size,
                    "learning_rate" : opt.lr,
                }
            else:
                config = {
                    "seed": getattr(self.opt, "seed", 123),
                    "gen": {
                        "do_sample": True,
                        "top_p": getattr(self.opt, "top_p", None),
                        "top_k": getattr(self.opt, "top_k", None),
                        "temperature": getattr(self.opt, "temperature", None),
                        "max_new_tokens": getattr(self.opt, "max_new_tokens", None),
                        },
                }
            
            if opt.isTrain:
                self.run = wandb.init(project=self.wandb_project_name,
                                      name=self.name + "_train",
                                      job_type='training',
                                      group=self.name,
                                      config=config,
                           )
            elif getattr(opt, "isValid", None):
                pass
            else:
                self.run = wandb.init(project=self.wandb_project_name,
                                      name=self.name + str(opt.resume_iter) + "_eval",
                                      job_type='eval',
                                      group=self.name,
                                      config=config
                           )
    
    def plot_current_loss(self, loss, total_steps):
        loss_dict = {"loss" : loss.detach().item()}
        self.run.log(loss_dict, step=total_steps)
    
    def log_eval(self,):
        pass