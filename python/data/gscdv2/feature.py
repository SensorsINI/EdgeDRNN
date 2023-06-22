__author__ = "Chang Gao"
__copyright__ = "Copyright @ Chang Gao"
__credits__ = ["Chang Gao"]
__license__ = "Private"
__version__ = "0.0.1"
__maintainer__ = "Chang Gao"
__email__ = "gaochangw@outlook.com"
__status__ = "Prototype"

from torch import nn
import pytorch_lightning as pl
import modules.util as util


# Definition of a neural network model
class FeatureExtractor(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        # PyTorch-Lightning Automatic Optimization
        self.automatic_optimization = True

        # Assign Attributes
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.accqi = args.aqi + args.wqi
        self.accqf = args.aqf + args.wqf
        self.debug = 0

        self.reset_parameters()
        self.save_hyperparameters()

    def set_qa(self, x):
        self.qa = x
        if 'Delta' in self.rnn_type:
            self.rnn.set_qa(x)

    def set_eval_sp(self, x):
        self.eval_sp = x
        if 'Delta' in self.rnn_type:
            self.rnn.set_eval_sparsity(x)

    def reset_parameters(self):
        for name, param in self.named_parameters():
            print("### Initializing Parameter: ", name)
            if 'rnn' in name:
                if 'weight' in name:
                    nn.init.orthogonal_(param[:self.rnn_size, :])
                    nn.init.orthogonal_(param[self.rnn_size:2 * self.rnn_size, :])
                    nn.init.orthogonal_(param[2 * self.rnn_size:3 * self.rnn_size, :])
            if 'fc' in name:
                if 'weight' in name:
                    # nn.init.orthogonal_(param)
                    nn.init.xavier_uniform_(param)
            if 'bias' in name:  # all biases
                nn.init.constant_(param, 0)

    def forward(self, x):
        # Attributes
        self.list_log = []
        dict_log_cl = {}
        qc = self.qc and not self.training

        # Flatten RNN Parameters if possible
        if self.rnn_type in ['GRU', 'LSTM']:
            self.rnn.flatten_parameters()

        # print(self.rnn.bias_ih_l0)

        # Forward Propagation
        out_rnn, _ = self.rnn(x)
        out_cl = self.cl(out_rnn)

        # Classification Layer
        out_cl_acc = util.quantize_tensor(out_cl, self.accqi, self.accqf, self.qc)
        qout_cl = util.quantize_tensor(out_cl, self.cqi, self.cqf, qc)

        if not self.training and self.rnn_type == 'DeltaGRU':
            dict_log_cl['cl_inp'] = out_rnn  # Classification Layer Input
            dict_log_cl['cl_out'] = out_cl  # Classification Layer Output
            dict_log_cl['cl_qout'] = qout_cl  # Output quantized to activation precision
            dict_log_cl['cl_out_acc'] = out_cl_acc  # Output quantized to accumulation precision
            self.list_log = self.rnn.list_log
            self.list_log.append(dict_log_cl)

        # Get Statistics
        # if 'Delta' in self.rnn_type:
        #     self.abs_delta_hid = self.rnn.all_layer_abs_delta_hid
        #     self.abs_delta_hid = self.abs_delta_hid.cpu()
        #     self.all_layer_dx = self.rnn.all_layer_dx
        #     self.all_layer_dh = self.rnn.all_layer_dh

        # if self.rnn_type == "DeltaGRU":
        #     self.list_debug = self.rnn.log

        return qout_cl, out_rnn
