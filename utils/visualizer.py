import wandb

class Visualizer:
    """
    This class is intended to track metrics, and training states for different experiments
    """

    def __init__(self, opt):

        self.opt = opt

        self.use_wandb = opt.use_wandb
        self.name = opt.name
        self.wandb_project_name = opt.wandb_project_name

        if self.use_wandb:
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
            self.run = wandb.init(project=self.wandb_project_name, name=self.name, config=config)

    
    def plot_current_loss(self, loss, total_steps):
        loss_dict = {"loss" : loss.detach().item()}
        self.run.log(loss_dict, step=total_steps)