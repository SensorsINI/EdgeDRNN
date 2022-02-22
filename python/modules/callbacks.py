__author__ = "Chang Gao"
__copyright__ = "Copyright @ Chang Gao"
__credits__ = ["Chang Gao"]
__license__ = "Private"
__version__ = "0.0.1"
__maintainer__ = "Chang Gao"
__email__ = "gaochangw@outlook.com"
__status__ = "Prototype"

from pytorch_lightning.callbacks import Callback
import modules.util as util
from project import Project


class Quantization(Callback):
    def __init__(self, proj: Project):
        self.qw = proj.qw
        self.wqi = proj.wqi
        self.wqf = proj.wqf

    # Quantize the model before each validation loop per epoch
    def on_validation_epoch_start(self, trainer, pl_module):
        for name, param in pl_module.named_parameters():
            # Quantize Network
            print("### Quantizing Parameter: " + name)
            param.data = util.quantize_tensor(param.data, self.wqi, self.wqf, self.qw)
