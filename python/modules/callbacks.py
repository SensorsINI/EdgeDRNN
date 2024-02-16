__author__ = "Chang Gao"
__copyright__ = "Copyright @ Chang Gao"
__credits__ = ["Chang Gao"]
__license__ = "Private"
__version__ = "0.0.1"
__maintainer__ = "Chang Gao"
__email__ = "gaochangw@outlook.com"
__status__ = "Prototype"

import pytorch_lightning as pl
import pandas as pd
import modules.util as util
from project import Project


class Quantization(pl.Callback):
    def __init__(self, proj: Project):
        self.qw = proj.qw
        self.wqi = proj.wqi
        self.wqf = proj.wqf

    # Quantize the model before each validation loop per epoch
    def on_validation_epoch_start(self, trainer, pl_module):
        for name, param in pl_module.named_parameters():
            # Quantize Network
            # print("### Quantizing Parameter: " + name)
            param.data = util.quantize_tensor(param.data, self.wqi, self.wqf, self.qw)


class CustomLoggingCallback(pl.Callback):
    def __init__(self):
        self.log_file_path = None
        self.log_data = []

    def on_train_start(self, trainer, pl_module):
        # Get the file path of the log CSV file
        self.log_file_path = trainer.logger.log_dir + '/' + 'train_log.csv'

    def on_validation_epoch_end(self, trainer, pl_module):
        print(self.log_file_path)
        # Get the current epoch and the metrics logged by the Trainer
        current_epoch = trainer.current_epoch
        train_metrics = trainer.callback_metrics

        # Extract the training and validation losses from the metrics
        train_loss = train_metrics['train_loss'].item()
        val_loss = train_metrics['val_loss'].item()
        learning_rate = pl_module.lr

        # Create a dictionary to store the metrics for the current epoch
        epoch_metrics = {'epoch': current_epoch,
                         'learning_rate': learning_rate,
                         'train_loss': train_loss,
                         'val_loss': val_loss}

        # Append the epoch metrics to the log data list
        self.log_data.append(epoch_metrics)

        # Create a pandas DataFrame from the log data
        df = pd.DataFrame(self.log_data)

        # Write the DataFrame to the log CSV file
        df.to_csv(self.log_file_path, index=False)
