import numpy as np
import os
import cv2 as cv
import pandas as pd
from numpy import matlib
import nibabel as nib
from ASBO import ASBO
from BWO import BWO
from CO import CO
from GOA import GOA
from Global_Vars import Global_Vars
from Model_DBN import Model_DBN
from Model_GCN import Model_GCN
from Model_ResUnet import Model_ResUnet
from Model_Resnet import Model_RESNET
from Model_Trans_R2_UNET import Model_Trans_R2Unet_plusplus
from Model_Trans_Unet import Model_Trans_Unet
from Model_Trans_Unet_CycleGAN import Model_Trans_Unet_CycleGAN
from Model_UNET import Model_Unet
from Model_WFP_MGCN_DBN import Model_WFP_MGCN_DBN
from Objective_Function import objfun_cls
from PROPOSED import PROPOSED
from Plot_results import *


# Read the Dataset
an = 0
if an == 1:
    Dataset = './nilearn_data/ABIDE_pcp/cpac/nofilt_noglobal'
    file = './nilearn_data/ABIDE_pcp/Dataset.txt'
    path = os.listdir(Dataset)  # Directory of the dataset
    IMAGE = []
    Target = []
    uni = []
    Tar = []
    df = pd.read_csv(file, sep=" ")
    Values = df.values
    for t in range(len(Values)):
        print(t, len(Values))
        val = Values[t].astype(str)
        targ = val[0].split('\t')
        Tar.append(targ[2])
    Tar = np.asarray(Tar)
    for i in range(len(path)):
        print(i)
        fold = Dataset + '/' + path[i]
        if '.gz' in fold:
            images_fold = nib.load(str(fold))
            image_file = images_fold.get_fdata()
            img = image_file[:, :, :, 77]
            for j in range(32, 40):
                print(i, len(path), j, img.shape)
                imge = img[:, :, j]
                imae = np.uint8(imge)
                uni.append(len(np.unique(imae)))
                image = cv.resize(imae, (512, 512))

                IMAGE.append(image)
                Target.append((Tar[i]).astype(int))

    np.save('Dataset.npy', IMAGE)
    np.save('Target.npy', np.reshape(Target, (-1, 1)))

# preprocess
an = 0
if an == 1:
    Images = np.load('Dataset.npy', allow_pickle=True)
    Preprocess = []
    for n in range(len(Images)):
        print(n, len(Images))
        image_bw = Images[n]
        clahe = cv.createCLAHE(clipLimit=5)  # Clahe Filtering
        final_img = clahe.apply(image_bw) + 30
        bilateral = cv.bilateralFilter(final_img, 15, 75, 75)  # Bilateral Filtering
        Preprocess.append(bilateral)
    np.save('Preprocess.npy', Preprocess)


# Segmentation
an = 0
if an == 1:
    Data_path = './Image_Dataset/'
    Data = np.load('Preprocess.npy', allow_pickle=True)  # Load the Data
    Target = np.load('Ground_Truths.npy', allow_pickle=True)  # Load the ground truth
    Unet = Model_Unet(Data_path)
    Res_Unet = Model_ResUnet(Data, Target)
    Trans_Unet = Model_Trans_Unet(Data, Target)
    Ada_F_ANN = Model_Trans_Unet_CycleGAN(Data, Target)
    Proposed = Model_Trans_R2Unet_plusplus(Data, Target,)
    Seg = [Unet, Res_Unet, Trans_Unet, Ada_F_ANN, Proposed]
    np.save('Segmented_Images.npy', Proposed)
    np.save('Seg_imgs.npy', Seg)


# OPTIMIZATION
an = 0
if an == 1:
    Seg = np.load('Segmented_Images.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Ground_Truth = np.load('Ground_Truths.npy', allow_pickle=True)
    Global_Vars.Feat = Seg
    Global_Vars.Target = Target
    Global_Vars.Ground_Truth = Ground_Truth
    Npop = 10
    Ch_len = 3  # Hidden neuron count, epoches, Activation Function in DBN
    xmin = matlib.repmat(np.asarray([5, 5, 1]), Npop, Ch_len)
    xmax = matlib.repmat(np.asarray([255, 50, 5]), Npop, Ch_len)
    initsol = np.zeros(xmax.shape)
    for p1 in range(Npop):
        for p2 in range(Ch_len):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    fname = objfun_cls
    max_iter = 50

    print('BWO....')
    [bestfit1, fitness1, bestsol1, Time1] = BWO(initsol, fname, xmin, xmax, max_iter)  # BWO

    print('GOA....')
    [bestfit2, fitness2, bestsol2, Time2] = GOA(initsol, fname, xmin, xmax, max_iter)  # GOA

    print('ASBO....')
    [bestfit3, fitness3, bestsol3, Time3] = ASBO(initsol, fname, xmin, xmax, max_iter)  # ASBO

    print('CO....')
    [bestfit4, fitness4, bestsol4, Time4] = CO(initsol, fname, xmin, xmax, max_iter)  # CO

    print('PROPOSED....')
    [bestfit5, fitness5, bestsol5, Time5] = PROPOSED(initsol, fname, xmin, xmax, max_iter)  # PROPOSED

    BestSol = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(), bestsol5.squeeze()]
    fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]

    np.save('Fitness.npy', np.asarray(fitness))
    np.save('BestSol.npy', np.asarray(BestSol))


# Classification
an = 0
if an == 1:
    Feat = np.load('Segmented_Images.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    bests = np.load('BestSols.npy', allow_pickle=True)
    EVAL = []
    Batchsize = [4, 8, 16, 32, 64, 128]
    for BS in range(len(Batchsize)):
        learnperc = round(Feat.shape[0] * 0.75)
        Train_Data = Feat[:learnperc, :]
        Train_Target = Target[:len(Feat)][:learnperc, :]
        Test_Data = Feat[learnperc:, :]
        Test_Target = Target[:len(Feat)][learnperc:, :]
        Eval = np.zeros((10, 25))
        for j in range(bests.shape[0]):
            soln = bests[j]
            Eval[j, :], Pred0 = Model_WFP_MGCN_DBN(Feat, Target, BS=Batchsize[BS], sol=soln)
        Eval[5, :], pred1 = Model_RESNET(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batchsize[BS])
        Eval[6, :], pred2 = Model_GCN(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batchsize[BS])
        Eval[7, :], pred3 = Model_DBN(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batchsize[BS])
        Eval[8, :], pred4 = Model_WFP_MGCN_DBN(Feat, Target, BS=Batchsize[BS])
        Eval[9, :] = Eval[4, :]
        EVAL.append(Eval)
    np.save('Eval_all_BS.npy', np.asarray(EVAL))


plot_conv()
ROC_curve()
Plot_KFold()
Plot_Batchsize()
plot_results_Seg()
Image_segment_comparision()
GUI()
