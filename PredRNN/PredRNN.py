#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
#%%


class LayerNorm2D(nn.LayerNorm):
    """专门为2D卷积设计的层归一化（与原版一致）"""
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps
        ).permute(0, 3, 1, 2)


class ActionSTLSTMCell(nn.Module):
    """FSTR模型的核心循环单元：带有动作输入的空间-时间LSTM单元（与原版一致）"""
    def __init__(self, in_channel, action_channel, num_hidden, filter_size, stride):
        super().__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 7, filter_size, stride, self.padding, padding_mode='reflect'),
            LayerNorm2D(num_hidden * 7)
        )
        self.conv_a = nn.Sequential(
            nn.Conv2d(action_channel, num_hidden * 4, filter_size, stride, self.padding, padding_mode='reflect'),
            LayerNorm2D(num_hidden * 4)
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, filter_size, stride, self.padding, padding_mode='reflect'),
            LayerNorm2D(num_hidden * 4)
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, filter_size, stride, self.padding, padding_mode='reflect'),
            LayerNorm2D(num_hidden * 3)
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, filter_size, stride, self.padding, padding_mode='reflect'),
            LayerNorm2D(num_hidden)
        )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)

    def forward(self, x_t, a_t, h_t, c_t, m_t):
        x_conv = self.conv_x(x_t)
        a_conv = self.conv_a(a_t)
        h_conv = self.conv_h(h_t)
        m_conv = self.conv_m(m_t)
        i_x, f_x, g_x, i_xp, f_xp, g_xp, o_x = torch.split(x_conv, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_conv * a_conv, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_conv, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        i_tp = torch.sigmoid(i_xp + i_m)
        f_tp = torch.sigmoid(f_xp + f_m + self._forget_bias)
        g_tp = torch.tanh(g_xp + g_m)

        delta_c = i_t * g_t
        c_new = f_t * c_t + delta_c
        delta_m = i_tp * g_tp
        m_new = f_tp * m_t + delta_m

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))
        return h_new, c_new, m_new, delta_c, delta_m


class ForcedSTRNN(nn.Module):
    """FSTR主模型 v2：与原版 HydroFrame 保持一致，包含 decouple_loss"""
    def __init__(
        self,
        num_layers,
        num_hidden,
        img_channel,
        act_channel,
        init_cond_channel,
        static_channel,
        out_channel,
        filter_size=5,
        stride=1,
    ):
        super().__init__()

        self.input_channel = img_channel
        self.action_channel = act_channel
        self.init_cond_channel = init_cond_channel
        self.static_channel = static_channel
        self.frame_channel = img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.out_channel = out_channel
        self.decouple_loss = None   # 用于存储解耦损失（原版特性）

        # 创建多个ActionSTLSTMCell层
        cell_list = []
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(ActionSTLSTMCell(
                in_channel, act_channel, num_hidden[i], filter_size, stride,
            ))
        self.cell_list = nn.ModuleList(cell_list)

        # 输出卷积层
        self.conv_last = nn.Conv2d(
            num_hidden[num_layers - 1],
            self.out_channel,
            kernel_size=1,
            bias=False
        )

        # 用于归一化 delta_c 和 delta_m 的适配器（与原版一致）
        self.adapter = nn.Conv2d(num_hidden[0], num_hidden[0], kernel_size=1, bias=False)

        # 记忆状态和细胞状态编码器
        self.memory_encoder = nn.Conv2d(
            self.init_cond_channel, num_hidden[0], kernel_size=1, bias=True
        )
        self.cell_encoder = nn.Conv2d(
            self.static_channel, sum(num_hidden), kernel_size=1, bias=True
        )

    def update_state(self, state):
        """对delta状态进行归一化，用于解耦损失（与原版一致）"""
        out_shape = (state.shape[0], state.shape[1], -1)
        return F.normalize(self.adapter(state).view(out_shape), dim=2)

    def calc_decouple_loss(self, c, m):
        """计算余弦相似度损失，促进c和m的解耦（与原版一致）"""
        return torch.mean(torch.abs(torch.cosine_similarity(c, m, dim=2)))

    def forward(self, forcings, init_cond, static_inputs):
        # 输入形状: (batch, length, channel, height, width)
        batch, timesteps, channels, height, width = forcings.shape

        # 初始化状态列表
        next_frames = []
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width],
                               device=forcings.device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        # 初始化记忆状态和第一层细胞状态
        memory = self.memory_encoder(init_cond[:, 0])
        c_t = list(torch.split(
            self.cell_encoder(static_inputs[:, 0]),
            self.num_hidden, dim=1
        ))

        # 第一个输入是初始条件
        x = init_cond[:, 0]
        decouple_losses = []
        for t in range(timesteps):
            a = forcings[:, t]

            # 第一层LSTM
            h_t[0], c_t[0], memory, dc, dm = self.cell_list[0](x, a, h_t[0], c_t[0], memory)
            delta_c_list[0] = self.update_state(dc)
            delta_m_list[0] = self.update_state(dm)

            # 后续层LSTM
            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, dc, dm = self.cell_list[i](h_t[i - 1], a, h_t[i], c_t[i], memory)
                delta_c_list[i] = self.update_state(dc)
                delta_m_list[i] = self.update_state(dm)

            # 输出（与原版一致：x = self.conv_last(h_t[-1]) + x）
            x = self.conv_last(h_t[-1]) + x
            next_frames.append(x)

            # 收集解耦损失
            for i in range(self.num_layers):
                decouple_losses.append(self.calc_decouple_loss(delta_c_list[i], delta_m_list[i]))

        # 堆叠输出: [batch, length, channel, height, width]
        next_frames = torch.stack(next_frames, dim=1)

        # 计算平均解耦损失并存储为属性（与原版一致）
        self.decouple_loss = torch.mean(torch.stack(decouple_losses))

        return next_frames
#%%