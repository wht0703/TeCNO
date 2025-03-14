from torch import nn
import torchvision.models as models

from models.cnn import Identity


class OneHeadEfficientNetModel(nn.Module):
    def __init__(self, hparams):
        super(OneHeadEfficientNetModel, self).__init__()
        self.model = models.efficientnet_b7(pretrained=hparams.pretrained)
        # replace final layer with number of labels
        self.model.fc = Identity()
        self.fc_phase = nn.Linear(1280, hparams.out_features)

    def forward(self, x):
        out_stem = self.model(x)
        phase = self.fc_phase(out_stem)
        return out_stem, phase

    @staticmethod
    def add_model_specific_args(parser):  # pragma: no cover
        resnet50model_specific_args = parser.add_argument_group(
            title='resnet50model specific args options')
        resnet50model_specific_args.add_argument("--pretrained",
                                                 action="store_true",
                                                 help="pretrained on imagenet")
        resnet50model_specific_args.add_argument(
            "--model_specific_batch_size_max", type=int, default=80)
        return parser