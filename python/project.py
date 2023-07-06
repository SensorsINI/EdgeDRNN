__author__ = "Chang Gao"
__copyright__ = "Copyright @ Chang Gao"
__credits__ = ["Chang Gao"]
__license__ = "Private"
__version__ = "0.0.1"
__maintainer__ = "Chang Gao"
__email__ = "gaochangw@outlook.com"
__status__ = "Prototype"

import importlib
import os
import typing
import argparse
import torch
import pytorch_lightning as pl
from modules import util


class Project:
    def __init__(self):
        # Precision
        torch.set_float32_matmul_precision('high')
        torch.set_printoptions(precision=8)

        # Hardware Info
        self.num_cpu_threads = os.cpu_count()

        # Arguments
        self.parser = argparse.ArgumentParser(description='Train a GRU network.')
        # Basic Setup
        args_basic = self.parser.add_argument_group("Basic Setup")
        args_basic.add_argument('--dataset_name', default='cps', help='Useful for loggers like Comet')
        args_basic.add_argument('--step', default='pretrain', help='A specific step to run.')
        args_basic.add_argument('--run_through', default=0, type=int, help='If true, run all following steps.')
        args_basic.add_argument('--accelerator', default='auto', help='Accelerator to use.')
        args_basic.add_argument('--num_gpus', default=1, type=int,
                                help='Number of gpus to use (Multi-GPU if larger than 1).')
        args_basic.add_argument('--model_path', default='', help='Model path to load. If empty, the experiment key will be used.')
        # Dataset Processing/Feature Extraction
        args_feat = self.parser.add_argument_group("Dataset Processing/Feature Extraction")
        args_feat.add_argument('--normalize_features', default=1, type=int, help='If 1, normalize features')
        args_feat.add_argument('--batch_first', default=1, type=int, help='If 1, put batch dimension as dim 0 for features')
        # Training Hyperparameters
        args_hparam_t = self.parser.add_argument_group("Training Hyperparameters")
        args_hparam_t.add_argument('--seed', default=2, type=int, help='Random seed.')
        args_hparam_t.add_argument('--epochs_pretrain', default=20, type=int, help='Number of epochs to train for.')
        args_hparam_t.add_argument('--epochs_retrain', default=2, type=int, help='Number of epochs to train for.')
        args_hparam_t.add_argument('--batch_size', default=128, type=int, help='Batch size.')
        args_hparam_t.add_argument('--batch_size_test', default=256, type=int, help='Batch size for test. Use larger values for faster test.')
        args_hparam_t.add_argument('--lr', default=3e-4, type=float, help='Learning rate')  # 5e-4
        args_hparam_t.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
        args_hparam_t.add_argument('--grad_clip_val', default=200, type=float, help='Gradient clipping')
        args_hparam_t.add_argument('--ctxt_size', default=50, type=int,
                                   help='The number of timesteps for RNN to look at')
        args_hparam_t.add_argument('--pred_size', default=0, type=int,
                                   help='The number of timesteps to predict in the future')
        # Model Hyperparameters
        args_hparam_m = self.parser.add_argument_group("Model Hyperparameters")

        args_hparam_m.add_argument('--rnn_layers', default=2, type=int, help='Number of RNN nnlayers')
        args_hparam_m.add_argument('--rnn_size', default=16, type=int, help='RNN Hidden layer size (must be a multiple of num_pe, see modules/edgedrnn.py)')
        args_hparam_m.add_argument('--rnn_type_pretrain', default='GRU', help='RNN type for pretrain')
        args_hparam_m.add_argument('--rnn_type_retrain', default='DeltaGRU', help='RNN type for pretrain')
        args_hparam_m.add_argument('--rnn_dropout', default=0.1, type=float, help='RNN Hidden layer size')
        # Quantization
        args_hparam_q = self.parser.add_argument_group("Quantization Hyperparameters")
        args_hparam_q.add_argument('--qa', default=1, type=int, help='Quantize the activations')
        args_hparam_q.add_argument('--qw', default=1, type=int, help='Quantize the weights')
        args_hparam_q.add_argument('--qaf', default=1, type=int,
                                   help='Quantize the activation functions (only AF in Delta Networks)')
        args_hparam_q.add_argument('--qc', default=1, type=int, help='Quantize the classification layer (CL)')
        args_hparam_q.add_argument('--aqi', default=8, type=int,
                                   help='Number of integer bits before decimal point for activation')
        args_hparam_q.add_argument('--aqf', default=8, type=int,
                                   help='Number of integer bits after decimal point for activation')
        args_hparam_q.add_argument('--wqi', default=1, type=int,
                                   help='Number of integer bits before decimal point for weight')
        args_hparam_q.add_argument('--wqf', default=7, type=int,
                                   help='Number of integer bits after decimal point for weight')
        args_hparam_q.add_argument('--afqi', default=2, type=int,
                                   help='Number of integer bits before decimal point for AF')
        args_hparam_q.add_argument('--afqf', default=4, type=int,
                                   help='Number of integer bits after decimal point for AF')
        args_hparam_q.add_argument('--cqi', default=8, type=int,
                                   help='Number of integer bits before decimal point for CL')
        args_hparam_q.add_argument('--cqf', default=8, type=int,
                                   help='Number of integer bits after decimal point for CL')
        # Delta Networks
        args_hparam_d = self.parser.add_argument_group("Delta Network Hyperparameters")
        args_hparam_d.add_argument('--thx', default=0 / 256, type=float, help='Delta threshold for inputs')
        args_hparam_d.add_argument('--thh', default=0 / 256, type=float, help='Delta threshold for hidden states')
        # Get EdgeDRNN-Specific Arguments
        args_edgedrnn = self.parser.add_argument_group("EdgeDRNN Arguments")
        args_edgedrnn.add_argument('--num_pe', default=8, type=int,
                                   help='#Processing elements (rnn_size must be a multiple of num_pe)')
        args_edgedrnn.add_argument('--stim_head', default=1000, type=int, help='Starting index of the HDL test stimuli')
        args_edgedrnn.add_argument('--stim_len', default=1000, type=int, help='#Timesteps of the HDL test stimuli')

        self.args = self.parser.parse_args()
        self.args.batch_first = bool(self.args.batch_first)

        # Get Hyperparameter Dictionary
        self.hparams = vars(self.args)
        for k, v in self.hparams.items():
            setattr(self, k, v)

        # Define abbreviations of hparams
        self.args_to_abb = {
            'seed': 'S',
            'inp_size': 'I',
            'rnn_size': 'H',
            'rnn_type': 'T',
            'rnn_layers': 'L',
            'num_classes': 'C',
            'ctxt_size': 'CT',
            'pred_size': 'PD',
            'qa': 'QA',
            'aqi': 'AQI',
            'aqf': 'AQF',
            'qw': 'QW',
            'wqi': 'WQI',
            'wqf': 'WQF',
        }
        self.abb_to_args = dict((v, k) for k, v in self.args_to_abb.items())
        self.experiment_key = None

        # Manage Steps
        self.list_steps = ['pretrain', 'retrain', 'export']
        self.update_attr('step_idx', self.list_steps.index(self.step))

    def update_args(self, args: argparse.Namespace):
        self.args = args
        for k, v in vars(args).items():
            setattr(self, k, v)

    def update_attr(self, key, value):
        setattr(self, key, value)
        setattr(self.args, key, value)
        self.hparams[key] = value

    def step_in(self):
        if self.run_through:
            self.update_attr('step_idx', self.step_idx + 1)
            self.update_attr('step', self.list_steps[self.step_idx])

    def gen_experiment_key(self, **kwargs) -> str:
        from operator import itemgetter

        # Add extra arguments if needed
        args_to_abb = {**self.args_to_abb, **kwargs}

        # Model ID
        list_args = list(args_to_abb.keys())
        list_abbs = list(itemgetter(*list_args)(args_to_abb))
        list_vals = list(itemgetter(*list_args)(self.hparams))
        list_vals_str = list(map(str, list_vals))
        experiment_key = list_abbs + list_vals_str
        experiment_key[::2] = list_abbs
        experiment_key[1::2] = list_vals_str
        experiment_key = '_'.join(experiment_key)
        self.experiment_key = experiment_key
        return experiment_key

    def decode_exp_id(self, exp_id: str):
        args = exp_id.split('_')
        vals = args[1::2]
        args = args[0::2]
        args = [self.abb_to_args[x] for x in args]
        dict_arg = dict(zip(args, vals))
        return dict_arg

    def create_loggers(self):
        """
        Create loggers
        """
        # Create Loggers
        csv_logger = pl.loggers.CSVLogger("logs/" + self.step, name=self.experiment_key)
        # comet_logger = pl.loggers.CometLogger(save_dir="logs_cmt/")
        # tb_logger = pl.loggers.TensorBoardLogger("logs_tb/" + proj.step, name=proj.experiment_key)

        # Log Hyperparameters
        csv_logger.log_hyperparams(self.hparams)
        # tb_logger.log_hyperparams(proj.hparams)
        # comet_logger.log_hyperparams(dict_args)

        return [csv_logger]  # add comet to the list if you want to use it

    def create_callbacks(self):
        """
        Create callbacks which will be executed sequentially
        """
        from pytorch_lightning.callbacks import ModelCheckpoint

        # Callback 1 - Quantize model after each training epoch
        from modules.callbacks import Quantization
        callback_quantization = Quantization(self)

        # Callback 2 - Save the best model at the lowest val_loss
        callback_best_model = ModelCheckpoint(
            filename=self.dataset_name + "-{epoch:02d}-{val_loss:.4f}.pt",
            monitor="val_loss",  # Monitor val_loss for saving the best models
            dirpath=os.path.join("save", self.step, self.experiment_key),
            save_top_k=1,  # Save the top-1 model
            mode="min",  # The lower the val_loss, the better the model
            save_weights_only=True
        )

        # Callback 3 - Learning Rate Monitor
        from pytorch_lightning.callbacks import LearningRateMonitor
        lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=True)

        # Callback 4
        from modules.callbacks import CustomLoggingCallback
        logger = CustomLoggingCallback()

        return [callback_quantization, callback_best_model, lr_monitor, logger]

    def prepare_dataloader(self):
        try:
            mod_dataloader = importlib.import_module('data.' + self.dataset_name + '.dataloader')
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Please select a supported dataset or check your name spelling.')
        dataloaders = mod_dataloader.DataLoader(self)
        return dataloaders

    def prepare_model(self):
        """
        Prepare the model to train
        """
        from modules.model import Model

        # Instantiate Models


        # Method to select the best pretrained model
        def select_best_model(ckpt_paths: typing.List) -> str:
            num_metrics = 2
            # Get basenames
            basenames = [os.path.basename(x) for x in ckpt_paths]
            # Get filenames
            filenames = [x.split('.ckpt')[0] for x in basenames]
            # Decode
            all_val_loss = [util.decode_ckpt_name(x, num_metrics=num_metrics)['val_loss'] for x in filenames]
            # Select
            min_val_loss = min(all_val_loss)
            idx_min_val_loss = all_val_loss.index(min_val_loss)
            best_model_path = ckpt_paths[idx_min_val_loss]
            return best_model_path

        # Retrain
        dir_path = self.model_path
        if self.step == 'pretrain':
            self.update_attr('rnn_type', self.rnn_type_pretrain)
            self.update_attr('max_epochs', self.epochs_pretrain)
            self.gen_experiment_key()
            model = Model(self.args)
            model.save_hyperparameters(self.args)
        elif self.step == 'retrain':
            self.update_attr('rnn_type', self.rnn_type_pretrain)
            self.update_attr('max_epochs', self.epochs_retrain)
            self.gen_experiment_key()
            # Load pretrained Model
            if dir_path == '':
                dir_path = "save/pretrain/" + self.experiment_key
            model_paths = util.get_filepaths(dirpath=dir_path, search_key='*.ckpt')
            pretrained_model_path = select_best_model(model_paths)
            print("### Loading Pretrained Model: ", pretrained_model_path)
            self.update_attr('rnn_type', self.rnn_type_retrain)
            self.gen_experiment_key()
            model = Model(self.args)
            model = model.load_gru_weight(pretrained_model_path)
            model.save_hyperparameters(self.args)
        # Export
        elif self.step == 'export':
            self.update_attr('rnn_type', self.rnn_type_retrain)
            self.gen_experiment_key()
            model = Model(self.args)
            if dir_path == '':
                dir_path = "save/retrain/" + self.experiment_key
            model_paths = util.get_filepaths(dirpath=dir_path, search_key='*.ckpt')
            retrained_model_path = select_best_model(model_paths)
            model = model.load_deltagru_weight(retrained_model_path)
            # model = model.load_from_checkpoint(retrained_model_path)
            # model.configure_retrain(self)

        return model

    def reproducible(self):
        pl.seed_everything(self.seed, workers=True)
        torch.backends.cudnn.deterministic = True