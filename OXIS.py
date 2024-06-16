import os
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import magic
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
import random
from sklearn.preprocessing import PolynomialFeatures
import itertools
from model import OXIS,MyDataset
from util import *

# 设置随机数
GLOBAL_SEED = 1
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(GLOBAL_SEED)
# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.chdir("C:/Users/Dell/AnacondaDamao/OXIS")   #修改当前工作目录
# hyper parameters
EPOCH = 2 # 需要修改
LR = 0.001
lamb123 = 0.1
lamb4 = 0.1
all_lamb5 = [1,5]
all_C = [0.05,0.08,0.1,0.12,0.15]
# 读入表达矩阵
geneExpr = pd.read_csv("./all.tpm.16968.csv",index_col = 0)
geneExpr = geneExpr.T
geneExpr = geneExpr*1.0 # DataFrame
if geneExpr.max().max() > 50:
    geneExpr = (geneExpr +1).apply(np.log2)
BATCH_SIZE = geneExpr.shape[0]

# 读入模块基因
moduleGene = pd.read_csv("./module_genes_input.csv",index_col = 0)
data_gene_all = set(geneExpr.columns)
# 读入样本信息
colinfo = pd.read_csv("./all.colinfo.csv",index_col = 0)
labels  = colinfo.loc[:,"Sample_type"].values

Os_ref = pd.read_csv("./all.17possible.ref.OS.levels.csv",index_col = 0).iloc[:,0]
Os_ref = torch.FloatTensor(Os_ref.values).to(device)

OrGene = moduleGene.iloc[0,:]
OrGene_ok = []
for i in range(len(OrGene)):
    if OrGene[i] in data_gene_all:
        tmpdata = geneExpr.loc[:,OrGene[i]]
        if (tmpdata<1.0).sum()/geneExpr.shape[0]<0.5:
            OrGene_ok.append(OrGene[i])
RsGene = moduleGene.iloc[1,:]
RsGene_ok = []
for i in range(len(RsGene)):
    if RsGene[i] in data_gene_all:
        tmpdata = geneExpr.loc[:,RsGene[i]]
        if (tmpdata<1.0).sum()/geneExpr.shape[0]<0.5:
            RsGene_ok.append(RsGene[i])
OsGene = moduleGene.iloc[2,:]
OsGene_ok = []
for i in range(len(OsGene)):
    if OsGene[i] in data_gene_all:
        tmpdata = geneExpr.loc[:,OsGene[i]]
        if (tmpdata<1.0).sum()/geneExpr.shape[0]<0.5:
            OsGene_ok.append(OsGene[i])
Expr_Or = geneExpr[OrGene_ok]
Expr_Rs = geneExpr[RsGene_ok]
Expr_Os = geneExpr[OsGene_ok]
Num_Or = Expr_Or.shape[1]
Num_Rs = Expr_Rs.shape[1]
Num_Os = Expr_Os.shape[1]
# 不带交互项的输入已经准备好
poly = PolynomialFeatures(interaction_only=True,include_bias = False) #定义了一个转化器

temp = pd.DataFrame(poly.fit_transform(Expr_Or))
temp.index = Expr_Or.index
original_names = list(Expr_Or.columns)
inter_names = ['_'.join(x) for x in list(itertools.combinations(original_names,2))]
all_names = original_names+inter_names
temp.columns = all_names
Expr_Or = temp

temp = pd.DataFrame(poly.fit_transform(Expr_Rs))
temp.index = Expr_Rs.index
original_names = list(Expr_Rs.columns)
inter_names = ['_'.join(x) for x in list(itertools.combinations(original_names,2))]
all_names = original_names+inter_names
temp.columns = all_names
Expr_Rs = temp

temp = pd.DataFrame(poly.fit_transform(Expr_Os))
temp.index = Expr_Os.index
original_names = list(Expr_Os.columns)
inter_names = ['_'.join(x) for x in list(itertools.combinations(original_names,2))]
all_names = original_names+inter_names
temp.columns = all_names
Expr_Os = temp
Num_Or = Expr_Or.shape[1]
Num_Rs = Expr_Rs.shape[1]
Num_Os = Expr_Os.shape[1]
# 到目前位置 Expr_Or，Expr_Rs，Expr_Os为三个输入矩阵，Num_Or，Num_Rs，Num_Os为三个矩阵列数，行数为BATCH_SIZE
# 带着交互项的输入

def myLoss(Expr_Or, Expr_Rs, Expr_Os, values_Or, values_Rs, values_Os, Os_ref, labels, lamb1=1, lamb2=1, lamb3=1,
           lamb4=1, lamb5=1, C0=1, C1=2, C2=3):
    #  相关性限制Os
    cor_Os = torch.FloatTensor(np.ones(Expr_Os.shape[1]))
    for i in range(Expr_Os.shape[1]):
        cor_Os[i] = pearsonr(Expr_Os[:, i], values_Os[:, 0])
    penal_Os_var = torch.FloatTensor(np.ones(Expr_Os.shape[1])) - cor_Os
    total1 = penal_Os_var
    # 相关性限制Or
    cor_Or = torch.FloatTensor(np.ones(Expr_Or.shape[1]))
    for i in range(Expr_Or.shape[1]):
        cor_Or[i] = pearsonr(Expr_Or[:, i], values_Or[:, 0])
    penal_Or_var = torch.FloatTensor(np.ones(Expr_Or.shape[1])) - cor_Or
    total2 = penal_Or_var

    # 相关性限制Rs
    cor_Rs = torch.FloatTensor(np.ones(Expr_Rs.shape[1]))
    for i in range(Expr_Rs.shape[1]):
        cor_Rs[i] = pearsonr(Expr_Rs[:, i], values_Rs[:, 0])
    penal_Rs_var = torch.FloatTensor(np.ones(Expr_Rs.shape[1])) - cor_Rs
    total3 = penal_Rs_var

    # 不同类型大小排序
    totalP = torch.tensor(0.)
    for i in range(len(labels)):
        label = labels[i]
        if label == "normal":
            pun = Punish(values_Os[i], 0, C0)
        elif label == "Solid Tissue Normal":
            pun = Punish(values_Os[i], C0, C1)
        else:
            pun = Punish(values_Os[i], C1, C2)
        totalP = totalP + pun

    # 相关性限制Os 结果规整
    total5 = 1 - pearsonr(Os_ref, values_Os[:, 0])
    # loss
    loss1 = torch.mean(lamb1 * total1)
    loss2 = torch.mean(lamb2 * total2)
    loss3 = torch.mean(lamb3 * total3)
    loss4 = lamb4 * totalP / len(labels)
    loss5 = lamb5 * total5
    loss = loss1 + loss2 + loss3 + loss4 + loss5
    return loss, loss1, loss2, loss3, loss4, loss5

#Dataloader---------------------------------------------------------------------------
dataloader_params = {'batch_size': BATCH_SIZE,
                         'shuffle': False,
                         'num_workers': 0,
                         'pin_memory': False}
Expr_Or = torch.FloatTensor(Expr_Or.values).to(device)#torch.Tensor
Expr_Rs = torch.FloatTensor(Expr_Rs.values).to(device)
Expr_Os = torch.FloatTensor(Expr_Os.values).to(device)
dataSet = MyDataset(Expr_Or,Expr_Rs,Expr_Os)
train_loader = torch.utils.data.DataLoader(dataset=dataSet,
                                               **dataloader_params)

sample_type_results = pd.DataFrame()
grade_results = pd.DataFrame()
stage_results = pd.DataFrame()
train_names = []
#-------------------------------------------------------------------------------------
for i in range(4):
    for j in range(i+1,5):
        for k in range(j+1,5):
            for lamb5 in all_lamb5:
                net = OXIS(f_in_Or=Num_Or,f_in_Rs=Num_Rs,f_in_Os=Num_Os).to(device)
#--------------------------------------------------------------------------------------
                start = time.time()
                #   training
                loss_v = []
                loss_v1 = []
                loss_v2 = []
                loss_v3 = []
                loss_v4 = []
                loss_v5 = []
                net.train()
                timestr = time.strftime("%Y%m%d-%H%M%S")
                lossName = "./output/lossValue_" + timestr + ".txt"
                file_loss = open(lossName, "a")
                for epoch in tqdm(range(EPOCH)):
                    loss, loss1, loss2, loss3, loss4, loss5 = 0, 0, 0, 0, 0, 0
                    for i, (Or, Rs, Os) in enumerate(train_loader):
                        Or_batch = Variable(Or.float().to(device))
                        Rs_batch = Variable(Rs.float().to(device))
                        Os_batch = Variable(Os.float().to(device))
                        Or_batch = F.normalize(Or_batch, p=2, dim=1)
                        Rs_batch = F.normalize(Rs_batch, p=2, dim=1)
                        Os_batch = F.normalize(Os_batch, p=2, dim=1)
                        for key in net.state_dict().keys():
                            if "bias" not in key:
                                myupdate = torch.abs(net.state_dict()[key])
                                net.state_dict()[key].copy_(myupdate)
                        values_Or, values_Rs, values_Os = net(Or_batch, Rs_batch, Os_batch)
                        loss_batch, loss1_batch, loss2_batch, loss3_batch, loss4_batch, loss5_batch = myLoss(Expr_Or,
                                                                                                             Expr_Rs,
                                                                                                             Expr_Os,
                                                                                                             values_Or,
                                                                                                             values_Rs,
                                                                                                             values_Os,
                                                                                                             Os_ref,
                                                                                                             labels,
                                                                                                             lamb1=lamb123,
                                                                                                             lamb2=lamb123,
                                                                                                             lamb3=lamb123,
                                                                                                             lamb4=lamb4,
                                                                                                             lamb5=lamb5,
                                                                                                             C0=all_C[i],
                                                                                                             C1=all_C[j],
                                                                                                             C2=all_C[k])
                        if epoch < 20:
                            lr_adaptive = LR
                        elif epoch < 30:
                            lr_adaptive = LR * 0.1
                        else:
                            lr_adaptive = LR * 0.01

                        optimizer = torch.optim.Adam(net.parameters(), lr=lr_adaptive)
                        optimizer.zero_grad()
                        loss_batch.backward()
                        optimizer.step()
                        loss += loss_batch.cpu().data.numpy()
                        loss1 += loss1_batch.cpu().data.numpy()
                        loss2 += loss2_batch.cpu().data.numpy()
                        loss3 += loss3_batch.cpu().data.numpy()
                        loss4 += loss4_batch.cpu().data.numpy()
                        loss5 += loss5_batch.cpu().data.numpy()

                    print('epoch: %02d, loss1: %.8f, loss2: %.8f, loss3: %.8f,loss4: %.8f,loss5: %.8f,  loss: %.8f' % (
                    epoch + 1, loss1, loss2, loss3, loss4, loss5, loss))
                    file_loss.write(
                        'epoch: %02d, loss1: %.8f, loss2: %.8f, loss3: %.8f, loss4: %.8f,loss5: %.8f, loss: %.8f. \n' % (
                        epoch + 1, loss1, loss2, loss3, loss4, loss5, loss))

                    loss_v.append(loss)
                    loss_v1.append(loss1)
                    loss_v2.append(loss2)
                    loss_v3.append(loss3)
                    loss_v4.append(loss4)
                    loss_v5.append(loss5)
                end = time.time()
                print("Training time: ", end - start)

                file_loss.close()
                results = np.hstack((values_Or.detach().numpy(),values_Rs.detach().numpy(),values_Os.detach().numpy()))
                results = pd.DataFrame(results)
                results.columns = ["OR","RS","OS"]
                results.index = colinfo.index
                results = pd.concat([results,colinfo],axis=1)
                # 根据sample_type 分组统计
                table_sample_type = results.loc[results.loc[:, "Sample_type"].isin(
                    ['normal', 'Solid Tissue Normal', 'Primary Tumor', "Metastatic"]), :]
                sample_type_results = pd.concat([sample_type_results, table_sample_type.groupby(by=["Sample_type"])["OS"].mean()], axis=1)

                table_sample_grade = results.loc[results.loc[:, "Grade"].isin(['G1', 'G2', 'G3', "G4"]), :]
                grade_results = pd.concat([grade_results, table_sample_grade.groupby(by=["Grade"])["OS"].mean()], axis=1)

                table_sample_stage = results.loc[results.loc[:,"Stage"].isin(['normal','control','Stage I',"Stage II","Stage III","Stage IV"]),:]
                stage_results = pd.concat([stage_results, table_sample_stage.groupby(by=["Stage"])["OS"].mean()], axis=1)

                train_names.append(str(all_C[i])+'+'+str(all_C[j])+'+'+str(all_C[k])+'+'+str(lamb5))
sample_type_results.columns = train_names
sample_type_results.to_csv('./output/sample_type.csv')
grade_results.columns = train_names
grade_results.to_csv('./output/grade.csv')
stage_results.columns = train_names
stage_results.to_csv('./output/stage.csv')
