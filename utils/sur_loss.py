import torch

def sur_loss(h,sur_time,censor):
    if sur_time==0:
        h = torch.squeeze(h)
        h = torch.split(h, [1, h.shape[0] - 1])
        s1 = -torch.sum(torch.log(1.0 - h[0] + 1e-30))
        if censor == 1:
            s1=s1-s1
            return s1
        else:
            return s1
    else :
        h=torch.squeeze(h)
        h=torch.split(h,[sur_time,1,h.shape[0]-1-sur_time])

        s0=-torch.sum(torch.log(h[0]+1e-30))
        s1=-torch.sum(torch.log(1.0-h[1]+1e-30))
        if censor==1:
            return s0
        else :
            return s0+s1
