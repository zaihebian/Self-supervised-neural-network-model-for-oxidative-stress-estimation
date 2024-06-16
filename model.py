# import sys
import torch
import torch.nn as nn


class OXIS(nn.Module):
    #     def __init__(self,f_in_Or,f_in_Rs,f_in_Os,f_out = 1):
    #         super(OXIS, self).__init__()
    #         # gene to flux
    #         self.inSize_Or = f_in_Or
    #         self.inSize_Rs = f_in_Rs
    #         self.inSize_Os = f_in_Os
    #         self.m_encoder = nn.ModuleList([nn.Sequential(nn.Linear(self.inSize_Or,f_out, bias = False)),
    #                                         nn.Sequential(nn.Linear(self.inSize_Rs,f_out, bias = False)),
    #                                         nn.Sequential(nn.Linear(self.inSize_Os,f_out, bias = False))])

    def __init__(self, f_in_Or, f_in_Rs, f_in_Os, f_out=1):
        super(OXIS, self).__init__()
        # gene to flux
        self.inSize_Or = f_in_Or
        self.inSize_Rs = f_in_Rs
        self.inSize_Os = f_in_Os
        self.alfa = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alfa.data.fill_(1.0)
        self.beta.data.fill_(1.0)
        self.m1 = nn.Linear(self.inSize_Or, f_out, bias=True)
        # self.m1.weight = torch.nn.Parameter(torch.ones((self.inSize_Or,f_out)))
        self.m2 = nn.Linear(self.inSize_Rs, f_out, bias=True)
        # self.m2.weight = torch.nn.Parameter(torch.ones((self.inSize_Rs,f_out)))
        self.m_encoder = nn.ModuleList([nn.Sequential(self.m1),
                                        nn.Sequential(self.m2)])

    def forward(self, Expr_Or, Expr_Rs, Expr_Os):
        values_Or = self.m_encoder[0](Expr_Or)
        values_Rs = self.m_encoder[1](Expr_Rs)
        values_Os = self.alfa * values_Or - self.beta * values_Rs
        return values_Or, values_Rs, values_Os


from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, Expr_Or, Expr_Rs, Expr_Os, transform=None):
        self.data_Or = Expr_Or
        self.data_Rs = Expr_Rs
        self.data_Os = Expr_Os
        self.transform = transform

    def __getitem__(self, index):
        Or = self.data_Or[index]
        Rs = self.data_Rs[index]
        Os = self.data_Os[index]

        if self.transform:
            Or = self.transform(Or)
            Rs = self.transform(Rs)
            Os = self.transform(Os)
        return Or, Rs, Os

    def __len__(self):
        return len(self.data_Or)