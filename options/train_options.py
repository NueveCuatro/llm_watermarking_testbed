from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    """
    This class define the training options. It subclasses base options thus it inclue-des the base options
    """

    def _initialize(self, parser):
        super()._initialize(parser)
        
        #display
        parser.add_argument('--display_freq', type=int, default=10, help='frequency for showing training results on screen')
        parser.add_argument('--Vmetrics', type=str, help="metrics to evalute the model's performence")
        parser.add_argument('--WMmetrics', type=str, help='metrics to evalute the watermark')
        
        #training parameters
        parser.add_argument('--n_epochs', type=int, default=5, help='number of epochs')
        parser.add_argument('--n_epochs_decay', type=int, default=5, help='number of epochs to linearly decay lr to 0')
        parser.add_argument('--optimizer', type=str, default='adamw', help='indicate the optimizer to use : adam | adamw')
        parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum terme of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='second momentum terme of adam')
        parser.add_argument('--weight_decay', type=float, default=1e-2, help='this is the weight decay term from adam')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy')
        parser.add_argument('--warmup_steps', type=int, default=None, help='number of warmup steps')

        self.isTrain = True

        return parser



