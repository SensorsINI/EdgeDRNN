__author__ = "Chang Gao"
__copyright__ = "Copyright @ Chang Gao"
__credits__ = ["Chang Gao"]
__license__ = "Private"
__version__ = "0.0.1"
__maintainer__ = "Chang Gao"
__email__ = "gaochangw@outlook.com"
__status__ = "Prototype"

import os.path

import torch
import numpy as np
from project import Project
import modules.util as util
from scipy import io


def cast_x_to_least_multiple(x: int, constant: int):
    return int(constant * np.ceil(x / constant))


class EdgeDRNN:
    def __init__(self, proj: Project):
        # Create Export Dir
        proj.update_attr(key='dir_tb', value='../hdl/tb')
        proj.update_attr(key='dir_sdk', value='sdk')
        util.create_folder([proj.dir_tb])
        util.create_folder([proj.dir_sdk])

        # Prepare Dataloader (Force Batch-1)
        # proj.update_attr(key='batch_size_test', value=1)

        # Create PyTorch Dataloader
        try:
            mod_dataset = importlib.import_module('data.' + self.dataset_name + '.dataset')
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Please select a supported dataset or check your name spelling.')
        setattr(self, "train_set", mod_dataset.AmproDataset(self, "training"))
        setattr(self, "dev_set", mod_dataset.AmproDataset(self, "validation"))
        setattr(self, "test_set", mod_dataset.AmproDataset(self, "testing"))
        self.dataloaders = DataLoader(proj)

        # Load DeltaGRU Model
        self.model = proj.prepare_model()
        self.model = self.model.eval()

        # Print Info
        print('###################################################################################\n'
              '# EdgeDRNN Hardware Configurations\n'
              '###################################################################################')
        inp_size_hw = cast_x_to_least_multiple(proj.inp_size, proj.num_pe)
        proj.update_attr('inp_size_hw', inp_size_hw)
        print(":::#Processing Elements:  ", proj.num_pe)
        print(":::RNN Input Size   (SW): ", proj.inp_size)
        print(":::RNN Input Size   (HW): ", proj.inp_size_hw)

        # Save Project Info
        self.proj = proj

        # EdgeDRNN
        self.edgedrnn_params = None

    def collect_params(self):
        print("### Collecting Network Parameters......")
        # Regroup parameters into rnn_weight, rnn_bias, fc_weight, fc_bias
        list_rnn_weight = []
        list_rnn_bias = []
        list_fc_weight = []
        list_fc_bias = []
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param_np = param.cpu().numpy()
                if 'rnn' in name and 'weight' in name:
                    list_rnn_weight.append(param_np)
                if 'rnn' in name and 'bias_ih' in name:
                    list_rnn_bias.append(np.reshape(param_np, (-1, 1)))
                if 'fc' in name and 'weight' in name:
                    list_fc_weight.append(param_np)
                if 'fc' in name and 'bias' in name:
                    list_fc_bias.append(param_np)

        # Concat RNN parameters for hardware mapping
        pad_len = int(self.proj.inp_size_hw - self.proj.inp_size)
        rnn_weight_ix_pad = np.zeros((int(3 * self.proj.rnn_size), pad_len))
        edgedrnn_params = []
        for i in range(self.proj.rnn_layers):
            edgedrnn_params.append(list_rnn_bias[i])
            edgedrnn_params.append(list_rnn_weight[2 * i])
            if i == 0:
                edgedrnn_params.append(rnn_weight_ix_pad)  # Pad zero columns to the first input layer
            edgedrnn_params.append(list_rnn_weight[2 * i + 1])
        self.edgedrnn_params = np.hstack(edgedrnn_params)
        self.edgedrnn_params_hw = util.cast_fp_to_fxp(self.edgedrnn_params, self.proj.wqi, self.proj.wqf)

        # Generate C Libraries for ilinx SDK
        dict_params = {
            'rnn_param': (self.edgedrnn_params, 1, self.proj.wqi, self.proj.wqf, 'param', 16),
            'rnn_layers': (self.proj.rnn_layers, 0, 16, 0, 'const', 1),
            'inp_size': (self.proj.inp_size_hw, 0, 16, 0, 'const', 1),
            'rnn_size': (self.proj.rnn_size, 0, 16, 0, 'const', 1),
            'thx': (self.proj.thx, 1, self.proj.aqi, self.proj.aqf, 'const', 1),
            'thh': (self.proj.thh, 1, self.proj.aqi, self.proj.aqf, 'const', 1),
            'num_pe': (self.proj.num_pe, 0, 16, 0, 'const', 1),
            'aqi': (self.proj.aqi, 0, 16, 0, 'const', 1),
            'aqf': (self.proj.aqf, 0, 16, 0, 'const', 1),
            'wqi': (self.proj.wqi, 0, 16, 0, 'const', 1),
            'wqf': (self.proj.wqf, 0, 16, 0, 'const', 1),
        }
        util.gen_xilinx_sdk_clib(self.proj.dir_sdk, 'edgedrnn_params', dict_params)

    def collect_test_data(self):
        print("### Collecting EdgeDRNN Test Data......")

        test_stim = []
        for i, (batch, _) in enumerate(self.dataloaders.test_loader):
            if i == 0:
                test_stim.append(batch[0, :, :])
                test_stim.append(batch[1:, -1, :])
            else:
                test_stim.append(batch[:, -1, :])
        test_stim = torch.cat(test_stim, dim=0)
        test_stim = test_stim[self.proj.stim_head:self.proj.stim_head + self.proj.stim_len, :]
        # If you want to plot the stimuli:
        import matplotlib.pyplot as plt
        # x = np.arange(test_stimuli.shape[0])
        # y = test_stimuli
        # plt.plot(x, y)
        # plt.show()

        # Feed the test stimuli to the model to get test_gold
        with torch.no_grad():
            test_stim = test_stim.unsqueeze(0)  # (batch, time, feature)
            test_gold_fc, test_gold_rnn = self.model(test_stim)
            test_stim = np.squeeze(test_stim.numpy()).transpose()  # (feature, time)
            test_gold_fc = np.squeeze(test_gold_fc.numpy()).transpose()  # (feature, time)
            test_gold_rnn = np.squeeze(test_gold_rnn.numpy()).transpose()  # (feature, time)

            # Get logged variables for debugging hardware
            dict_log = {}
            for i, dict_log_layer in enumerate(self.model.list_log):
                for key, value in dict_log_layer.items():
                    dict_log['l' + str(i) + '_' + key] = np.transpose(np.squeeze(value.numpy()))

        # Log for Hardware Debugging
        dict_log_hw = {}
        aqi = self.proj.aqi
        aqf = self.proj.aqf
        wqi = self.proj.wqi
        wqf = self.proj.wqf
        accqi = 16  # Precision of the product between activation and weight (use a number larger than wqi+wqf)
        accqf = aqf + wqf
        aaqi = aqi + aqi  # Precision of the product of two numbers with the activation precision
        aaqf = aqf + aqf
        for key, val in dict_log.items():
            if 'pacc' in key:
                dict_log_hw[key] = util.cast_fp_to_fxp(val, accqi, accqf)
            elif 'paa' in key:
                dict_log_hw[key] = util.cast_fp_to_fxp(val, aaqi, aaqf)
            elif 'pa' in key:
                dict_log_hw[key] = util.cast_fp_to_fxp(val, aqi, aqi)
            else:
                dict_log_hw[key] = util.cast_fp_to_fxp(val, wqi, wqf)

        # Pad test stimuli to inp_size_hw
        inp_pad = np.zeros((int(self.proj.inp_size_hw - self.proj.inp_size), int(self.proj.stim_len)))
        test_stim = np.concatenate((test_stim, inp_pad), axis=0)

        # Export Xilinx SDK Test Data
        test_gold_rnn_hw = util.cast_fp_to_fxp(test_gold_rnn, self.proj.aqi, self.proj.aqf)
        dict_test = {
            'edgedrnn_test_stim': (test_stim, 1, self.proj.aqi, self.proj.aqf, 'tensor', test_stim.shape[0]),
            'edgedrnn_test_gold_fc': (test_gold_fc, 0, 0, 0, 'tensor', test_gold_fc.shape[0]),
            'edgedrnn_test_gold_rnn': (
            test_gold_rnn, 1, self.proj.aqi, self.proj.aqf, 'tensor', test_gold_rnn.shape[0]),
        }
        util.gen_xilinx_sdk_clib(self.proj.dir_sdk, 'edgedrnn_test', dict_test)

        # Export Verilog Testbench Stimuli
        dict_param_tb = {
            'edgedrnn_tb_params': self.edgedrnn_params
        }
        dict_test_tb = {
            'edgedrnn_tb_stim': test_stim,
            'edgedrnn_tb_gold_rnn': test_gold_rnn,
        }
        cfg_reg = np.zeros((32, 1))
        cfg_reg[0] = 0  # Base address of weights, 0 for verilog simulation
        cfg_reg[1] = self.proj.thx
        cfg_reg[2] = self.proj.thh
        cfg_reg[3] = self.proj.rnn_layers
        cfg_reg[4] = self.proj.inp_size_hw
        for i in range(5, 5 + self.proj.rnn_layers):
            cfg_reg[i] = self.proj.rnn_size
        dict_cfg = {
            'edgedrnn_tb_creg': cfg_reg,
        }
        util.gen_sim_file(self.proj.dir_tb, dict_param_tb, self.proj.wqi, self.proj.wqf, row_blocks=8)
        util.gen_sim_file(self.proj.dir_tb, dict_test_tb, self.proj.aqi, self.proj.aqf, row_blocks=8)
        util.gen_sim_file(self.proj.dir_tb, dict_cfg, 16, 0)

        # Export to MATLAB
        dict_mat = {
            'rnn_param': self.edgedrnn_params,
            'rnn_layers': self.proj.rnn_layers,
            'inp_size': self.proj.inp_size_hw,
            'rnn_size': self.proj.rnn_size,
            'thx': self.proj.thx,
            'thh': self.proj.thh,
            'num_pe': self.proj.num_pe,
            'aqi': self.proj.aqi,
            'aqf': self.proj.aqf,
            'wqi': self.proj.wqi,
            'wqf': self.proj.wqf,
        }
        io.savemat(self.proj.dir_sdk + '/edgedrnn_params.mat', dict_mat)
        io.savemat(self.proj.dir_sdk + '/edgedrnn_log.mat', dict_log)

    def gen_lut(self):
        ###################################################################################
        # Create Sigmoid LUT
        ###################################################################################
        # Create file
        try:
            f_sig = open(os.path.join(self.proj.dir_tb, 'sigmoid_lut.txt'), 'w')
        except IOError as err:
            print("I/O error({0}): {1}".format(err.errno, err.strerror))

        # Create Sigmoid LUT
        min_in = int(util.cast_fp_to_fxp(-8, self.proj.aqi, self.proj.aqf))
        max_in = int(util.cast_fp_to_fxp(8, self.proj.aqi, self.proj.aqf))
        sigmoid_lut_in = np.arange(min_in, max_in).astype(float) / float((2 ** self.proj.aqf))
        sigmoid_lut_out = torch.sigmoid(torch.Tensor(sigmoid_lut_in))
        sigmoid_lut_out = util.quantize_array(sigmoid_lut_out.numpy(), self.proj.afqi, self.proj.afqf, enable=1)
        for i in range(0, len(sigmoid_lut_in)):
            f_sig.write('%d : lut_out[i] = %d;\n' % (
                int(util.cast_fp_to_fxp(sigmoid_lut_in[i], self.proj.aqi, self.proj.aqf)),
                int(util.cast_fp_to_fxp(sigmoid_lut_out[i], self.proj.aqi, self.proj.aqf))))
        f_sig.close()

        print('###################################################################################\n'
              '# Create Tanh LUT\n'
              '###################################################################################\n')
        # Create file
        try:
            f_tanh = open(os.path.join(self.proj.dir_tb, 'tanh_lut.txt'), 'w')
        except IOError as err:
            print("I/O error({0}): {1}".format(err.errno, err.strerror))

        # Create Tanh LUT
        min_in = int(util.cast_fp_to_fxp(-4, self.proj.aqi, self.proj.aqf))
        max_in = int(util.cast_fp_to_fxp(4, self.proj.aqi, self.proj.aqf))
        tanh_lut_in = np.arange(min_in, max_in).astype(float) / float((2 ** self.proj.aqf))
        tanh_lut_out = torch.tanh(torch.Tensor(tanh_lut_in))
        tanh_lut_out = util.quantize_array(tanh_lut_out.numpy(), self.proj.afqi, self.proj.afqf, enable=1)
        for i in range(0, len(tanh_lut_in)):
            f_tanh.write('%d : lut_out[i] = %d;\n' % (
                int(util.cast_fp_to_fxp(tanh_lut_in[i], self.proj.aqi, self.proj.aqf)),
                int(util.cast_fp_to_fxp(tanh_lut_out[i], self.proj.aqi, self.proj.aqf))))
        f_tanh.close()