from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    """
    This class define the training options. It subclasses base options thus it inclue-des the base options
    """

    def _initialize(self, parser):
        super()._initialize(parser)
        
        #display
        parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        parser.add_argument('--Vmetrics', type=str, help="metrics to evalute the model's performence")
        parser.add_argument('--WMmetrics', type=str, help='metrics to evalute the watermark')
        #training parameters
        parser.add_argument('--n_epochs', type=int, default=5, help='number of epochs')
        parser.add_argument('--n_epochs_decay', type=int, default=5, help='number of epochs to linearly decay lr to 0')
        parser.add_argument('--beta1', type=int, default=0.5, help='momentum terme of adam')
        parser.add_argument('--lr', type=int, default=2e-4, help='learning rate')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy')
        parser.add_argument('--warup_iters', type=int, default=0, help='number of warmup steps')

        self.isTrain = True

        return parser



