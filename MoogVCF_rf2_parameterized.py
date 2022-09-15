import math

import torch
import torch.nn as nn


class MoogVCFRFModelCell(nn.Module):
    def __init__(self, r, f, gain, sample_rate):
        args = locals()
        del args['self']
        del args['__class__']
        super(MoogVCFRFModelCell, self).__init__()

        for key in args:
            self.register_buffer(key, torch.FloatTensor([args[key]]))

        # Trainable parameters
        self.gf = nn.Parameter(torch.FloatTensor([1.6071]), requires_grad=True)#3.58273.5170
        self.gr = nn.Parameter(torch.FloatTensor([0.9993]), requires_grad=True)#0.93691.0376
        self.gain = nn.Parameter(torch.FloatTensor([3.9415]), requires_grad=True)#3.77922.9607
        self.alpha = nn.Parameter(torch.FloatTensor([0.8782]), requires_grad=True) #0.8682 alpha = 1, traperzoidal rule

        # Helper Buffers
        self.register_buffer("k", torch.FloatTensor([1 / self.sample_rate]))
        self.register_buffer("I", torch.eye(4))
        self.register_buffer("zero_element", torch.FloatTensor([0.0]))
        self.register_buffer("one_element", torch.FloatTensor([1.0]))

    def update_parameters(self, batch_size):

        self.gr.data.clamp_(min=1e-3)
        self.gf.data.clamp_(min=1e-3)
        self.alpha.data.clamp_(min=0, max=1)

        self.update_matrices(batch_size)

    def update_matrices(self, batch_size):

        self.A = torch.stack((torch.stack((-2 * math.pi * self.gf * self.f, self.zero_element, self.zero_element, - 4 * 2 * math.pi * self.gf * self.f * self.gr * self.r), dim = 1),
                              torch.stack((2 * math.pi * self.gf * self.f, -2 * math.pi * self.gf * self.f, self.zero_element, self.zero_element), dim=1),
                              torch.stack((self.zero_element, 2 * math.pi * self.gf * self.f, -2 * math.pi * self.gf * self.f, self.zero_element), dim=1),
                              torch.stack((self.zero_element, self.zero_element, 2 * math.pi * self.gf * self.f, -2 * math.pi * self.gf * self.f), dim=1)),
                             dim=0).squeeze(1)

        self.B = torch.stack((2 * math.pi * self.gf * self.f, self.zero_element, self.zero_element, self.zero_element), dim=1).t()

        self.C = torch.stack((self.zero_element, self.zero_element, self.zero_element, self.gain * self.one_element), dim=1)

        self.k_A_div_two = self.k * self.A / 2
        self.inv_I_minus_kAdt_m_a = torch.inverse(self.I - self.k_A_div_two * self.alpha)
        self.I_plus_kA_minus_a = self.I + self.k_A_div_two * 2 - self.k_A_div_two * self.alpha
        self.k_mul_b_div_two = self.k * self.B / 2
        self.kmbdt_m_a = self.k_mul_b_div_two * self.alpha
        self.kmbdt_m_one_m_a = self.k_mul_b_div_two * (1 - self.alpha / 2)


    def forward(self, u_n, x_n1, u_n1, batch_size, device):
        batch_size = batch_size
        x_n = torch.zeros((batch_size, 4, 1), dtype=torch.float32, device=device)
        y_n = torch.zeros((batch_size, 1, 1), dtype=torch.float32, device=device)

        for i in range(0, batch_size):
            x_n[i] = self.inv_I_minus_kAdt_m_a.matmul(self.I_plus_kA_minus_a.matmul(x_n1[i]) + self.kmbdt_m_a.matmul(u_n[i]) + self.kmbdt_m_one_m_a.matmul(u_n1[i]))
            y_n[i] = self.C.matmul(x_n[i])
        return y_n, x_n, u_n

class MoogVCFRFModel(nn.Module):
    def __init__(self, r, f, gain, sample_rate):
        super(MoogVCFRFModel, self).__init__()
        self.cell = MoogVCFRFModelCell(r, f, gain, sample_rate)

    def init_states(self, batch_size, device):
        previous_input = torch.zeros((batch_size, 1, 1), dtype=torch.float32, device=device)
        previous_state = torch.zeros((batch_size, 4, 1), dtype=torch.float32, device=device)
        return previous_input, previous_state

    def forward(self, input_tensor):
        device = input_tensor.device
        batch_size      = input_tensor.shape[0]
        sequence_length = input_tensor.shape[1]

        previous_input, previous_state = self.init_states(batch_size, device)
        self.cell.update_matrices(batch_size)

        out_sequence = torch.zeros((batch_size, sequence_length, 1)).to(device)
        for n in range(sequence_length):
            output, previous_state, previous_input = self.cell(input_tensor[:, n:n+1, :], previous_state, previous_input, batch_size, device)
            out_sequence[:, n, :] = output.view(-1, 1)
        return out_sequence
