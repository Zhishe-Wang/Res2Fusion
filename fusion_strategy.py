import torch
EPSILON = 1e-10
from torch.nn import Softmax

'''atten strategy'''
def attention_fusion_weight(tensor1, tensor2):
    f_channel = channel_fusion(tensor1, tensor2)
    f_spatial = spatial_fusion(tensor1, tensor2)
    tensor_f = (f_channel + f_spatial) / 2
    return tensor_f

# channel atten
def channel_fusion(tensor1, tensor2):
    # calculate channel attention
    attention_map1 = channel_attention(tensor1)
    attention_map2 = channel_attention(tensor2)
    # get weight map
    attention_p1_w1 = attention_map1 / (attention_map1 + attention_map2 + EPSILON)
    attention_p2_w2 = attention_map2 / (attention_map1 + attention_map2 + EPSILON)

    tensor_f = attention_p1_w1 * tensor1 + attention_p2_w2 * tensor2
    return tensor_f

def channel_attention(tensor):
    B, C, H, W = tensor.size()
    out1 = tensor.view(B, C, -1) #(C,N)
    out2 = tensor.view(B, C, -1).permute(0, 2, 1) #(N,C)
    energy = (torch.bmm(out1, out2))
    min = torch.min(energy)
    max = torch.max(energy)
    energy_norm = (energy - min) / (max - min)

    ca_softmax = Softmax(dim=-1)
    energy_norm = ca_softmax(energy_norm)

    out3 = tensor.view(B, C, -1) #(C,N)
    attention_p = torch.bmm(energy_norm, out3)
    attention_map = attention_p.view(B, C, H, W)
    attention_map = tensor + attention_map
    return attention_map


# spatial atten
def spatial_fusion(tensor1, tensor2):
    # calculate spatial attention
    spatial1 = spatial_attention(tensor1)
    spatial2 = spatial_attention(tensor2)
    # get weight map
    spatial_w1 = spatial1 / (spatial1 + spatial2 + EPSILON)
    spatial_w2 = spatial2 / (spatial1 + spatial2 + EPSILON)

    tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2
    return tensor_f

def spatial_attention(tensor):

    spatial = tensor
    B, C, H, W = spatial.size()
    sa_avg_pooling = torch.nn.AvgPool2d(8, stride=8)
    spatial1 = spatial.view(B, -1, H * W).permute(0, 2, 1)  # [B,(HW),C]

    spatial2 = sa_avg_pooling(spatial)  # [B,C,H/8,W/8]
    spatial2_H = spatial2.size()[2]
    spatial2_W = spatial2.size()[3]
    spatial2 = spatial2.view(B, -1, spatial2_H * spatial2_W)  # [B,C,(HW//64)]
    energy = torch.bmm(spatial1, spatial2) # [B,(HW),(HW//64)]

    min = torch.min(energy)
    max = torch.max(energy)
    energy_norm = (energy - min) / (max - min)

    sa_softmax = Softmax(dim=-1)
    energy_norm= sa_softmax(energy_norm )# [B,(HW),(HW//64)]

    spatial3 = sa_avg_pooling(spatial)  # [B,C,H/8,W/8]
    spatial3_H = spatial3.size()[2]
    spatial3_W = spatial3.size()[3]
    spatial3 = spatial3.view(B, -1, spatial3_H * spatial3_W).permute(0, 2, 1)  # [B,(HW//64),C]
    spatial_atten = torch.bmm(energy_norm , spatial3)  # [B,(HW),C]
    spatial_map = spatial_atten.permute(0, 2, 1).view(B, -1, H, W)  # [B,C,H,W]
    spatial_map = tensor + spatial_map
    return spatial_map


'''add srtategy'''
def addition_fusion(tensor1, tensor2):
    return (tensor1 + tensor2) / 2





