import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function


def quantizeTensor(x, m, n, en):
    """
    :param x: input tensor
    :param m: number of integer bits before the decimal point
    :param n: number of fraction bits after the decimal point
    :return: tensor quantized to fixed-point precision
    """
    if en == 0:
        return x
    power = 2. ** n
    clip_val = 2. ** (m + n)
    value = GradPreserveRoundOp.apply(x * power)
    # value = t.round(x * power)
    value = t.clamp(value, -clip_val, clip_val - 1)  # saturation arithmetic
    value = value / power
    return value


class GradPreserveRoundOp(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input):
        output = t.round(input)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        grad_input = grad_output

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        # print(grad_output.size())
        # if not t.equal(grad_output, QuantizeT(grad_output, dW_qp)): print("grad_output not quantized")
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        # Return same number of parameters as "def forward(...)"
        return grad_input


def get_temporal_sparsity(list_layer, seq_len, threshold):
    # Evaluate Sparsity
    num_zeros = 0
    num_elems = 0
    # Iterate through nnlayers
    for layer in list_layer:
        all_delta_vec = layer.transpose(0, 1)
        all_delta_vec = t.abs(all_delta_vec)  # Take absolute values of all delta vector elements
        for i, delta_vec in enumerate(all_delta_vec):
            seq = delta_vec[:seq_len[i], :]
            zero_mask = seq < threshold
            num_zeros += t.sum(zero_mask)
            num_elems += t.numel(zero_mask)
    sparsity = float(num_zeros) / float(num_elems)
    return sparsity


def hard_sigmoid(x, qi, qf, q_enable):
    """
    Hard sigmoid function: y(x) = 0.25*x + 0.5

    :param x: input tensor
    :param qi: number of bit before decimal points for quantization
    :param qf: number of bit after decimal points for quantization
    :param q_enable: If = 1, enable quantization
    :return: output of the hard sigmoid funtion
    """
    x = quantizeTensor(0.25 * x, qi, qf, q_enable) + 0.5
    x = t.clamp(x, 0, 1)
    return x


class DeltaGRUCell(nn.Module):
    def __init__(self,
                 n_inp,
                 n_hid,
                 th_x=0.25,
                 th_h=0.25,
                 aqi=8,
                 aqf=8,
                 wqi=8,
                 wqf=8,
                 eval_sparsity=0,
                 quantize_act=0,
                 debug=0,
                 cuda=1):
        super(DeltaGRUCell, self).__init__()

        # Properties
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.th_x = th_x
        self.th_h = th_h
        self.aqi = aqi
        self.aqf = aqf
        self.wqi = wqi
        self.wqf = wqf
        self.eval_sparsity = eval_sparsity
        self.quantize_act = quantize_act
        self.cuda = cuda

        # Network parameters
        self.weight_ih = t.nn.Parameter(t.empty(3 * n_hid, n_inp))
        self.weight_hh = t.nn.Parameter(t.empty(3 * n_hid, n_hid))
        self.bias_ih = t.nn.Parameter(t.empty(3 * n_hid))
        self.bias_hh = t.nn.Parameter(t.empty(3 * n_hid))

        # Statistics
        self.dx = []
        self.dh = []

        # Regularizer
        self.abs_delta_hid = 0

        # Debug
        self.debug = debug
        self.dict_debug = {}
        self.dict_debug['x'] = []
        self.dict_debug['h'] = []
        self.dict_debug['dx'] = []
        self.dict_debug['dh'] = []
        self.dict_debug['q_mem_r'] = []
        self.dict_debug['q_mem_u'] = []
        self.dict_debug['mem_cx'] = []
        self.dict_debug['q_mem_ch'] = []
        self.dict_debug['q_acc_c'] = []
        self.dict_debug['r'] = []
        self.dict_debug['u'] = []
        self.dict_debug['c'] = []
        self.dict_debug['a'] = []
        self.dict_debug['b'] = []
        self.dict_debug['one_minus_u'] = []

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        nn.init.constant_(self.bias_ih, 0)
        nn.init.constant_(self.bias_hh, 0)

    def timestep(self, delta_inp, delta_hid, hidden, mem_x_prev, mem_ch_prev):
        acc_x_curr = F.linear(delta_inp, self.weight_ih) + mem_x_prev
        acc_h_curr = F.linear(delta_hid, self.weight_hh)
        acc_x_chunks = acc_x_curr.chunk(3, dim=1)
        acc_h_chunks = acc_h_curr.chunk(3, dim=1)
        mem_ch_curr = acc_h_chunks[2] + mem_ch_prev

        mem_r_curr = acc_x_chunks[0] + acc_h_chunks[0]
        mem_u_curr = acc_x_chunks[1] + acc_h_chunks[1]
        mem_cx_curr = acc_x_chunks[2]
        mem_x_curr = t.cat((mem_r_curr, mem_u_curr, mem_cx_curr), 1)

        # Quantize Delta Memories
        q_mem_r_curr = quantizeTensor(mem_r_curr, self.aqi, self.aqf, self.quantize_act)
        q_mem_u_curr = quantizeTensor(mem_u_curr, self.aqi, self.aqf, self.quantize_act)

        # Calculate reset gate and update gate
        # r = hard_sigmoid(q_mem_r_curr, self.aqi, self.aqf, self.quantize_act)
        # u = hard_sigmoid(q_mem_u_curr, self.aqi, self.aqf, self.quantize_act)
        r = t.sigmoid(q_mem_r_curr)
        u = t.sigmoid(q_mem_u_curr)
        r = quantizeTensor(r, self.aqi, self.aqf / 2, self.quantize_act)
        u = quantizeTensor(u, self.aqi, self.aqf / 2, self.quantize_act)

        # Quantize mem_ch_curr
        q_mem_ch_curr = quantizeTensor(mem_ch_curr, self.aqi, self.aqf, self.quantize_act)

        # Calculate candidate accumulation
        acc_c = mem_cx_curr + t.mul(r, q_mem_ch_curr)
        q_acc_c = quantizeTensor(acc_c, self.aqi, self.aqf, self.quantize_act)

        # Calculate candidate
        # c = F.hardtanh(q_acc_c)
        c = t.tanh(q_acc_c)
        c = quantizeTensor(c, self.aqi, self.aqf / 2, self.quantize_act)

        # Quantize candidate
        one_minus_u = (t.ones(u.size(0), u.size(1)).cuda() - u)
        a = quantizeTensor(t.mul(one_minus_u, c), self.aqi, self.aqf, self.quantize_act)
        b = quantizeTensor(t.mul(u, hidden), self.aqi, self.aqf, self.quantize_act)
        next_hidden = a + b

        # Debug
        if self.debug:
            self.dict_debug['q_mem_r'].append(t.squeeze(q_mem_r_curr))
            self.dict_debug['q_mem_u'].append(t.squeeze(q_mem_u_curr))
            self.dict_debug['mem_cx'].append(t.squeeze(mem_cx_curr))
            self.dict_debug['q_mem_ch'].append(t.squeeze(q_mem_ch_curr))
            self.dict_debug['q_acc_c'].append(t.squeeze(q_acc_c))
            self.dict_debug['r'].append(t.squeeze(r))
            self.dict_debug['u'].append(t.squeeze(u))
            self.dict_debug['c'].append(t.squeeze(c))
            self.dict_debug['a'].append(t.squeeze(a))
            self.dict_debug['b'].append(t.squeeze(b))
            self.dict_debug['one_minus_u'].append(t.squeeze(one_minus_u))

        return next_hidden, mem_x_curr, mem_ch_curr

    def iter(self, x, max_seq_len, n_batch):

        # Initialize result accumulator
        self.abs_delta_hid = 0
        hidden = t.zeros(n_batch, self.n_hid).float()  # Initialize hidden state
        input_prev = t.zeros(n_batch, self.n_inp).float()  # Initialize previous input state
        hidden_prev = t.zeros(n_batch, self.n_hid).float()  # Initialize previous hidden state
        mem_x = t.unsqueeze(self.bias_ih, dim=0).repeat(n_batch, 1).float()  # Initialize mem_x
        mem_ch = t.zeros(n_batch, self.n_hid).float()  # Initialize mem_ch
        output = t.zeros(max_seq_len, n_batch, self.n_hid).float()
        if self.cuda:
            hidden = hidden.cuda()
            input_prev = input_prev.cuda()
            hidden_prev = hidden_prev.cuda()
            mem_x = mem_x.cuda()
            mem_ch = mem_ch.cuda()
            output = output.float().cuda()

        # Quantize input X
        x = quantizeTensor(x, self.aqi, self.aqf, 1)

        # Save history of delta vectors to evaluate sparsity
        self.dx = []
        self.dh = []

        # Iterate through time steps
        for i, input_curr in enumerate(x.chunk(max_seq_len, dim=0)):
            # Get current input vectors
            input_curr = t.squeeze(input_curr, dim=0)
            hidden_curr = hidden

            # Get raw delta vectors
            delta_inp = input_curr - input_prev
            delta_hid = hidden_curr - hidden_prev

            # Zero-out elements of delta input vector below the threshold
            delta_inp_abs = t.abs(delta_inp)
            delta_inp = delta_inp.masked_fill_(delta_inp_abs < self.th_x, 0)

            # Zero-out elements of delta hidden vector below the threshold
            delta_hid_abs = t.abs(delta_hid)
            delta_hid = delta_hid.masked_fill_(delta_hid_abs < self.th_h, 0)

            # Get L1 Penalty
            self.abs_delta_hid += t.abs(delta_hid)

            # Run forward pass for one time step
            hidden, mem_x, mem_ch = self.timestep(delta_inp,
                                                  delta_hid,
                                                  hidden,
                                                  mem_x,
                                                  mem_ch)

            # Quantize hidden
            hidden = quantizeTensor(hidden, self.aqi, self.aqf, 1)

            # Evaluate Temporal Sparsity
            if self.eval_sparsity:
                self.dx.append(delta_inp)
                self.dh.append(delta_hid)

            # Debug
            if self.debug:
                self.dict_debug['x'].append(t.squeeze(input_curr))
                self.dict_debug['h'].append(t.squeeze(hidden))
                self.dict_debug['dx'].append(t.squeeze(delta_inp))
                self.dict_debug['dh'].append(t.squeeze(delta_hid))

            # Update previous input vector memory on indices that had above-threshold change
            input_prev = t.where(delta_inp_abs >= self.th_x, input_curr, input_prev)

            # Update previous hidden vector memory on indices that had above-threshold change
            hidden_prev = t.where(delta_hid_abs >= self.th_h, hidden_curr, hidden_prev)

            # Append current DeltaGRU hidden output to the list
            output[i, :, :] = hidden

        if self.eval_sparsity:
            self.dx = t.stack(self.dx).detach().cpu()
            self.dh = t.stack(self.dh).detach().cpu()

        self.abs_delta_hid = self.abs_delta_hid.cpu()

        return output

    def set_quantize_act(self, x):
        self.quantize_act = x

    def set_eval_sparsity(self, x):
        self.eval_sparsity = x

    def forward(self, x):
        """
        :param input: 3D-Input tensor of feature of all time steps with size (seq_len, n_batch, n_feature)
        :param feat_len: 1D-Tensor of size (n_batch) having sequence length of each sample in the batch
        :param show_sp: Whether to return sparsity of delta vectors
        :return:
            - output_seq: 3D-Tensor of all time steps of the rnn outputs with size (seq_len, n_batch, n_hid)
            - nz_dx: Number of nonzero elements in delta input vectors of all time steps
            - nz_dh Number of nonzero elements in delta hidden vectors of all time steps
            - abs_delta_hid: L1 mean penalty
            - sp_W: Weight sparsity
        """

        # Get Input Tensor Dimensions
        max_seq_len = x.size()[0]
        n_batch = x.size()[1]

        # Clear debug dictionary
        for key, value in self.dict_debug.items():
            self.dict_debug[key] = []

        output = self.iter(x, max_seq_len, n_batch)

        # Store data for debugging
        if self.debug:
            for key, value in self.dict_debug.items():
                self.dict_debug[key] = t.stack(value, dim=0)

        return output


class DeltaGRU(nn.Module):
    def __init__(self,
                 n_inp,
                 n_hid,
                 num_layers,
                 th_x=0.25,
                 th_h=0.25,
                 aqi=8,
                 aqf=8,
                 wqi=8,
                 wqf=8,
                 eval_sparsity=0,
                 quantize_act=0,
                 debug=0,
                 cuda=1):
        super(DeltaGRU, self).__init__()

        # Properties
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.num_layers = num_layers
        self.th_x = th_x
        self.th_h = th_h
        self.aqi = aqi
        self.aqf = aqf
        self.wqi = wqi
        self.wqf = wqf
        self.eval_sparsity = eval_sparsity
        self.quantize_act = 0
        self.debug = debug
        self.cuda = cuda
        self.layer_list = nn.ModuleList()

        # Statistics
        self.all_layer_dx = []
        self.all_layer_dh = []
        self.all_layer_abs_delta_hid = t.zeros(1)

        self.dropout = nn.Dropout(p=0.5)

        # Debug
        self.list_rnn_debug = []

        # Instantiate DeltaGRU nnlayers
        for i in range(self.num_layers):
            if i == 0:
                layer = DeltaGRUCell(n_inp=self.n_inp,
                                     n_hid=self.n_hid,
                                     th_x=self.th_x,
                                     th_h=self.th_h,
                                     aqi=self.aqi,
                                     aqf=self.aqf,
                                     wqi=self.wqi,
                                     wqf=self.wqf,
                                     eval_sparsity=self.eval_sparsity,
                                     quantize_act=quantize_act,
                                     debug=self.debug,
                                     cuda=self.cuda)
            else:
                layer = DeltaGRUCell(n_inp=self.n_hid,
                                     n_hid=self.n_hid,
                                     th_x=self.th_x,
                                     th_h=self.th_h,
                                     aqi=self.aqi,
                                     aqf=self.aqf,
                                     wqi=self.wqi,
                                     wqf=self.wqf,
                                     eval_sparsity=self.eval_sparsity,
                                     quantize_act=quantize_act,
                                     debug=self.debug,
                                     cuda=self.cuda)
            self.layer_list.append(layer)

    def set_quantize_act(self, x):
        self.quantize_act = x
        for i in range(self.num_layers):
            self.layer_list[i].set_quantize_act(self.quantize_act)

    def set_eval_sparsity(self, x):
        self.eval_sparsity = x
        for i in range(self.num_layers):
            self.layer_list[i].set_eval_sparsity(self.eval_sparsity)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("The input vector x must be 3-dimensional (len, batch, n_feat)")
        self.list_rnn_debug = []
        # Propagate through nnlayers
        self.all_layer_dx = []
        self.all_layer_dh = []
        self.all_layer_abs_delta_hid = 0
        for i, rnn in enumerate(self.layer_list):
            # if i < len(self.layer_list) - 1:
            x = rnn(x)
            #     x = self.dropout(x)
            if self.debug:
                self.list_rnn_debug.append(rnn.dict_debug)
            self.all_layer_abs_delta_hid += rnn.abs_delta_hid
            self.all_layer_dx.append(rnn.dx)
            self.all_layer_dh.append(rnn.dh)

        return x, x[-1, :, :]
