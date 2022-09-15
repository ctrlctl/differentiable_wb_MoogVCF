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
        self.gf = nn.Parameter(torch.FloatTensor([1.5755]), requires_grad=True)#3.2285
        self.gr = nn.Parameter(torch.FloatTensor([1.2305]), requires_grad=True)#-1.8575
        self.gain = nn.Parameter(torch.FloatTensor([3.0912]), requires_grad=True)#-1.8575
        self.alpha = nn.Parameter(torch.FloatTensor([0.7624]), requires_grad=True)


        # Helper Buffers
        self.register_buffer("k", torch.FloatTensor([1 / self.sample_rate]))
        self.register_buffer("I", torch.eye(4))
        self.register_buffer("E", torch.ones((4,1)))
        self.register_buffer("zero_element", torch.FloatTensor([0.0]))
        self.register_buffer("one_element", torch.FloatTensor([1.0]))

    def update_parameters(self, previous_state, previous_input):

        self.gr.data.clamp_(min=1e-3)
        self.gf.data.clamp_(min=1e-3)

        self.update_matrices(previous_state, previous_input)

    def update_matrices(self, previous_state, previous_input):

        if previous_state[0,3] == 0:
            self.A = torch.stack((torch.stack((-2 * math.pi * self.gf * self.f, self.zero_element, self.zero_element, - 2 * math.pi * self.gf * self.f * 4 * self.gr * self.r), dim = 1),
                                  torch.stack((2 * math.pi * self.gf * self.f, -2 * math.pi * self.gf * self.f, self.zero_element, self.zero_element), dim=1),
                                  torch.stack((self.zero_element, 2 * math.pi * self.gf * self.f, -2 * math.pi * self.gf * self.f, self.zero_element), dim=1),
                                  torch.stack((self.zero_element, self.zero_element, 2 * math.pi * self.gf * self.f, -2 * math.pi * self.gf * self.f), dim=1)),
                                 dim=0).squeeze(1)
        else:
            x4_temp = previous_state[0,3]
            self.A = torch.stack((torch.stack((-2 * math.pi * self.gf * self.f, self.zero_element, self.zero_element, - 2 * math.pi * self.gf * self.f * math.tanh(4 * self.gr * self.r * x4_temp) / math.tanh(x4_temp)), dim = 1),
                                  torch.stack((2 * math.pi * self.gf * self.f, -2 * math.pi * self.gf * self.f, self.zero_element, self.zero_element), dim=1),
                                  torch.stack((self.zero_element, 2 * math.pi * self.gf * self.f, -2 * math.pi * self.gf * self.f, self.zero_element), dim=1),
                                  torch.stack((self.zero_element, self.zero_element, 2 * math.pi * self.gf * self.f, -2 * math.pi * self.gf * self.f), dim=1)),
                                 dim=0).squeeze(1)


        self.B = torch.stack((2 * math.pi * self.gf * self.f * self.beta_function(self.gr * self.r * previous_state[0, 3], previous_input[0]), self.zero_element, self.zero_element, self.zero_element), dim=1).t()

        self.C = torch.stack((self.zero_element, self.zero_element, self.zero_element, self.gain * self.one_element), dim=1)

        self.k_A_mul = self.k * self.A * (1 - self.alpha / 2)
        self.k_b_mul = self.k * self.B * (1 - self.alpha / 2)
        self.k_b_al  = self.k * self.B * self.alpha / 2
        self.K = self.k * self.A * self.alpha / 2


    def beta_function(self, x, u):
        result = self.miu_function(x) * (1 - math.pow(math.tanh(x),2)) / (1 - math.tanh(x) * math.tanh(u))
        return result

    def miu_function(self, x):
        if x == 0:
            return 1
        else:
            return math.tanh(x) / x

    def forward(self, u_n, x_n1, u_n1, batch_size, device):
        batch_size = batch_size
        x_n = torch.zeros((batch_size, 4, 1), dtype=torch.float32, device=device)
        p_n = torch.zeros((batch_size, 4, 1), dtype=torch.float32, device=device)
        y_n = torch.zeros((batch_size, 1, 1), dtype=torch.float32, device=device)

        # Newton Raphson parameters
        tol = 1e-9
        max_iter = 50

        for i in range(0, batch_size):
            x_n1_th = torch.zeros_like(x_n1[i])
            for m in range(0, x_n1.shape[1]):
                x_n1_th[m] = math.tanh(x_n1[i,m])
            p_n[i] = x_n1[i] + self.k_A_mul.matmul(x_n1_th) + self.k_b_mul.matmul(u_n[i]) + self.k_b_al.matmul(u_n1[i])
            x_temp = x_n1[i]

            # Newton Raphson
            for j in range(0, max_iter):
                # Calculate g
                x_t_th = torch.zeros_like(x_n[i])
                for m in range(0, x_n.shape[1]):
                    x_t_th[m] = math.tanh(x_temp[m])
                g = p_n[i] + self.K.matmul(x_t_th) - x_temp

                # Calculate J
                x_t_one_m_p_th = torch.zeros_like(x_n[i])
                for m in range(0, x_n.shape[1]):
                    x_t_one_m_p_th[m] = 1 - pow(math.tanh(x_temp[m]), 2)
                J = self.K.matmul(x_t_one_m_p_th) - self.E
                deltax = g / J
                x_temp = x_temp - deltax
                if math.fabs(deltax[0]) <= tol and math.fabs(deltax[1]) <= tol \
                        and math.fabs(deltax[2]) <= tol and math.fabs(deltax[3]) <= tol:
                    break

            x_n[i] = x_temp
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

        out_sequence = torch.zeros((batch_size, sequence_length, 1)).to(device)
        for n in range(sequence_length):
            self.cell.update_matrices(previous_state, previous_input)
            output, previous_state, previous_input = self.cell(input_tensor[:, n:n+1, :], previous_state, previous_input, batch_size, device)
            out_sequence[:, n, :] = output.view(-1, 1)
        return out_sequence
