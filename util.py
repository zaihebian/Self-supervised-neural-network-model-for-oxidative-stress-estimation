import torch
# 两种惩罚函数

def pearsonr(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val

# def Punish(x,low_lim,high_lim):
#     if x>high_lim:
#         p = x - high_lim
#     elif x<low_lim:
#         p = low_lim - x
#     else:
#         p = 0
#     return p

def Punish(x,mid):
    return abs(x-mid)