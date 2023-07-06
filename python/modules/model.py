__author__ = "Chang Gao"
__copyright__ = "Copyright @ Chang Gao"
__credits__ = ["Chang Gao"]
__license__ = "Private"
__version__ = "0.0.1"
__maintainer__ = "Chang Gao"
__email__ = "gaochangw@outlook.com"
__status__ = "Prototype"

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from nnlayers.deltagru import DeltaGRU
import modules.util as util
from project import Project


# Definition of a neural network model
class Model(pl.LightningModule):
    def __init__(self, args):
        """
        Args:
            - qa, aqi, aqf, wqi, wqf, qaf, afqi, afqf
            - inp_size, rnn_size, rnn_layers, rnn_dropout
            - thx, thh
            - num_classes
        """
        super().__init__()
        # PyTorch-Lightning Automatic Optimization
        self.automatic_optimization = True

        # Assign Attributes
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.accqi = args.aqi + args.wqi
        self.accqf = args.aqf + args.wqf
        self.debug = 0

        # Statistics
        self.statistics = {}
        self.abs_delta_hid = 0
        self.abs_std_delta_hid = 0
        self.all_layer_dx = []
        self.all_layer_dh = []

        # Debug
        self.list_debug = []

        # self.bn = nn.BatchNorm1d(cla_size)

        # Loss
        self.loss = torch.nn.MSELoss()

        # RNN
        self.rnn_type = self.rnn_type
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=self.inp_size,
                              hidden_size=self.rnn_size,
                              num_layers=self.rnn_layers,
                              bias=True,
                              bidirectional=False,
                              batch_first=self.batch_first,
                              dropout=self.rnn_dropout)
        elif self.rnn_type == 'DeltaGRU':
            self.rnn = DeltaGRU(input_size=self.inp_size,
                                hidden_size=self.rnn_size,
                                num_layers=self.rnn_layers,
                                batch_first=self.batch_first,
                                thx=self.thx,
                                thh=self.thh,
                                qa=self.qa,
                                # qaf=self.qaf,
                                aqi=self.aqi,
                                aqf=self.aqf,
                                nqi=self.afqi,
                                nqf=self.afqf,
                                # eval_sp=1,
                                debug=0
                                )
        self.cl = nn.Linear(in_features=self.rnn_size, out_features=self.num_classes, bias=True)
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
                    nn.init.orthogonal_(param[self.rnn_size:2*self.rnn_size, :])
                    nn.init.orthogonal_(param[2*self.rnn_size:3*self.rnn_size, :])
            if 'cl' or 'fc' in name:
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

        # Forward Propagation
        if 'Delta' in self.rnn_type:
            out_rnn, state, reg = self.rnn(x)
        else:
            out_rnn, state = self.rnn(x)
        out_cl = self.cl(out_rnn)

        # Classification Layer
        out_cl_acc = util.quantize_tensor(out_cl, self.accqi, self.accqf, self.qc)
        qout_cl = util.quantize_tensor(out_cl, self.cqi, self.cqf, qc)

        if not self.training and self.rnn_type == 'DeltaGRU':
            dict_log_cl['cl_inp'] = out_rnn  # Classification Layer Input
            dict_log_cl['cl_out'] = out_cl  # Classification Layer Output
            # dict_log_cl['cl_qout'] = qout_cl  # Output quantized to activation precision
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

    def compute_loss(self, batch):
        features, labels = batch
        out_cl, _ = self(features)  # Executes self.forward()
        loss = self.loss(out_cl, labels)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)  # Compute Loss
        self.log("train_loss", loss)  # Log
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)  # Compute Loss
        self.log("val_loss", loss)  # Log

    def test_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)  # Compute Loss
        self.log("test_loss", loss)  # Log

    def configure_optimizers(self):
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        optimizer = optim.AdamW(self.parameters(), lr=self.lr,
                                amsgrad=False, weight_decay=self.weight_decay)

        lr_scheduler_config = {
            "scheduler": ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=5),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config
        }

    def configure_retrain(self, proj: Project):
        self.rnn_type = 'DeltaGRU'
        proj.update_attr('rnn_type', 'DeltaGRU')
        proj.gen_experiment_key()
        self.rnn = DeltaGRU(input_size=self.inp_size,
                            hidden_size=self.rnn_size,
                            num_layers=self.rnn_layers,
                            batch_first=True,
                            thx=self.thx,
                            thh=self.thh,
                            qa=self.qa,
                            qaf=self.qaf,
                            aqi=self.aqi,
                            aqf=self.aqf,
                            afqi=self.afqi,
                            afqf=self.afqf,
                            eval_sp=1,
                            debug=0,
                            gru_layer=self.rnn
                            )

    def load_gru_weight(self, model_path: str):
        state_dict_loaded = torch.load(model_path, map_location='cpu')['state_dict']
        with torch.no_grad():
            for name, param in self.named_parameters():
                loaded_param = state_dict_loaded[name]
                # setattr(self, name, loaded_param)
                param.data = loaded_param.data.to(param.device)
            self.rnn.process_biases()

        return self

    def load_deltagru_weight(self, model_path: str):
        state_dict_loaded = torch.load(model_path, map_location=torch.device('cpu'))['state_dict']
        with torch.no_grad():
            for name, param in self.named_parameters():
                loaded_param = state_dict_loaded[name]
                # setattr(self, name, loaded_param)
                param.data = loaded_param.data.to(param.device)
        return self

    def get_sparsity(self):
        for name, module in self._modules.items():
            self.statistics['SP_W_' + name.upper()] = util.get_layer_sparsity(module)
        temporal_sparsity = self.rnn.get_temporal_sparsity()
        self.statistics.update(temporal_sparsity)
        return self.statistics

    def get_model_size(self):
        self.statistics['NUM_PARAMS'] = 0
        for name, param in self.named_parameters():
            self.statistics['NUM_PARAMS'] += param.data.numel()
        return self.statistics['NUM_PARAMS']