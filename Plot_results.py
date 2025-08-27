from itertools import cycle
import numpy as np
import warnings
import cv2 as cv
from prettytable import PrettyTable
from matplotlib import pylab
from sklearn.metrics import roc_curve

from GUI import GUI

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt


def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_conv():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    fig = pylab.gcf()
    fig.canvas.manager.set_window_title('Convergence')
    Algorithm = ['TERMS', 'BWO', 'GOA', 'ASBO', 'CO', 'RMCO']
    Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
    Conv_Graph = np.zeros((Fitness.shape[-2], 5))
    for j in range(len(Algorithm) - 1):
        Conv_Graph[j, :] = stats(Fitness[j, :])

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('-------------------------------------------------- Statistical Report ',
          ' Dataset --------------------------------------------------')
    print(Table)
    length = np.arange(Fitness.shape[-1])
    Conv_Graph = Fitness

    plt.plot(length, Conv_Graph[0, :], color='#e50000', linewidth=3, marker='.', markerfacecolor='red',
             markersize=12, label='BWO')
    plt.plot(length, Conv_Graph[1, :], color='#0504aa', linewidth=3, marker='.', markerfacecolor='green',
             markersize=12, label='GOA')  # c
    plt.plot(length, Conv_Graph[2, :], color='#76cd26', linewidth=3, marker='.', markerfacecolor='cyan',
             markersize=12, label='ASBO')
    plt.plot(length, Conv_Graph[3, :], color='#b0054b', linewidth=3, marker='.', markerfacecolor='magenta',
             markersize=12, label='CO')  # y
    plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='.', markerfacecolor='black',
             markersize=12, label='RMCO')
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    plt.savefig("./Results/Convergence.png")
    plt.show()


def ROC_curve():
    lw = 2
    cls = ['Resnet', 'GCN', 'DBN', 'WMGC-DBN', 'RMCO-WMGC-DBN']
    Actual = np.load('Target.npy', allow_pickle=True).astype('int')
    from keras.utils import to_categorical
    Actual = to_categorical(Actual, dtype="uint8")
    colors = cycle(["#fe2f4a", "#0165fc", "#ffff14", "lime", "black"])
    for i, color in zip(range(len(cls)), colors):
        Predicted = np.load('Y_Score.npy', allow_pickle=True)[i]
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
        plt.plot(
            false_positive_rate1,
            true_positive_rate1,
            color=color,
            lw=lw,
            label=cls[i],
        )
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    path = "./Results/ROC.png"
    plt.savefig(path)
    plt.show()


def Plot_Batchsize():
    eval = np.load('Eval_all_BS.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'FOR', 'NPV', 'FDR', 'F1_score',
             'MCC',
             'pt',
             'ba', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']
    Graph_Term = [0, 3, 7, 12, 14, 16, 20]
    for j in range(len(Graph_Term)):
        Graph = np.zeros((eval.shape[0], eval.shape[1]))
        for k in range(eval.shape[0]):
            for l in range(eval.shape[1]):
                Graph[k, l] = eval[k, l, Graph_Term[j] + 4]

        length = np.arange(Graph.shape[0])
        fig = plt.figure()
        ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])

        ax.plot(length, Graph[:, 0], color='#010fcc', linewidth=3, marker='D', markerfacecolor='red',  # 98F5FF
                markersize=6, label='BWO-WMGC-DBN')
        ax.plot(length, Graph[:, 1], color='#08ff08', linewidth=3, marker='s', markerfacecolor='green',  # 7FFF00
                markersize=6, label='GOA-WMGC-DBN')
        ax.plot(length, Graph[:, 2], color='#fe420f', linewidth=3, marker='H', markerfacecolor='cyan',  # C1FFC1
                markersize=8, label='ASBO-WMGC-DBN')
        ax.plot(length, Graph[:, 3], color='#00ffff', linewidth=3, marker='p', markerfacecolor='#fdff38',
                markersize=8, label='CO-WMGC-DBN')
        ax.plot(length, Graph[:, 4], color='k', linewidth=3, marker='*', markerfacecolor='w', markersize=12,
                label='RMCO-WMGC-DBN')
        plt.xticks(length, ('4', '8', '16', '32', '64', '128'))
        plt.xlabel('Batch size', fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
        plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Batch size vs ' + Terms[Graph_Term[j]])
        path = "./Results/Batch size_%s_lrean.png" % (Terms[Graph_Term[j]])
        plt.savefig(path)
        plt.show()

        fig = plt.figure()
        ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
        X = np.arange(Graph.shape[0])

        ax.bar(X + 0.00, Graph[:, 5], color='#0165fc', edgecolor='w', width=0.15, label="Resnet")
        ax.bar(X + 0.15, Graph[:, 6], color='#ff474c', edgecolor='w', width=0.15, label="GCN")
        ax.bar(X + 0.30, Graph[:, 7], color='#be03fd', edgecolor='w', width=0.15, label="DBN")
        ax.bar(X + 0.45, Graph[:, 8], color='#21fc0d', edgecolor='w', width=0.15, label="WMGC-DBN")
        ax.bar(X + 0.60, Graph[:, 4], color='k', edgecolor='w', width=0.15, label="RMCO-WMGC-DBN")
        plt.xticks(X + 0.15, ('4', '8', '16', '32', '64', '128'))
        plt.xlabel('Batch size', fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
        plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Batch size vs ' + Terms[Graph_Term[j]])
        path = "./Results/Batch size_%s_bar.png" % (Terms[Graph_Term[j]])
        plt.savefig(path)
        plt.show()


def Plot_KFold():
    eval = np.load('Eval_all_Fold.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'FOR', 'NPV', 'FDR', 'F1_score',
             'MCC',
             'pt',
             'ba', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']
    Table_Term = [0, 13, 14, 15, 17]
    Algorithm = ['TERMS', 'BWO-WMGC-DBN', 'GOA-WMGC-DBN', 'ASBO-WMGC-DBN', 'CO-WMGC-DBN', 'RMCO-WMGC-DBN']
    Classifier = ['TERMS', 'Resnet', 'GCN', 'DBN', 'WMGC-DBN', 'RMCO-WMGC-DBN']
    for i in range(eval.shape[0]):
        value = eval[i, :, 4:]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], (np.asarray(Terms))[np.asarray(Table_Term)])
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[j, Table_Term])
        print('--------------------------------------------------' + str(i + 1) + ' Fold',
              'Algorithm Comparison',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], (np.asarray(Terms))[np.asarray(Table_Term)])
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, Table_Term])
        print('-------------------------------------------------- ' + str(i + 1) + ' Fold',
              'Classifier Comparison',
              '--------------------------------------------------')
        print(Table)


def plot_results_Seg():
    Eval_all = np.load('Eval_all_seg.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'PSNR', 'MSE', 'Sensitivity', 'Specificity', 'Precision', 'FPR',
             'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]
        stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
        for i in range(4, value_all[0].shape[1] - 9):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i])
                    stats[i, j, 1] = np.min(value_all[j][:, i])
                    stats[i, j, 2] = np.mean(value_all[j][:, i])
                    stats[i, j, 3] = np.median(value_all[j][:, i])
                    stats[i, j, 4] = np.std(value_all[j][:, i])

            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
            Graph = stats[i, :5, 2]

            labels = ['UNET', 'ResUnet', 'TransUnet', 'Trans-Unet\n-CycleGAN', 'Trans-\nR2Unet++']
            colors = ['green', 'orange', 'purple', '#ff474c', 'k']
            ax.bar(labels, Graph, color=colors)
            plt.xticks(labels, rotation=5)
            plt.ylabel(Terms[i - 4])
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('Segmented Mean vs ' + Terms[i - 4])
            path = "./Results/Dataset_%s_Mean_Seg_%s_mtd.png" % (n + 1, Terms[i - 4])
            plt.savefig(path)
            plt.show()


def Image_segment_comparision():
    Original = np.load('Images.npy', allow_pickle=True)
    Ground_truth = np.load('Ground_Truth.npy', allow_pickle=True)
    segmented = np.load('Seg_img.npy', allow_pickle=True)
    Preprocess = np.load('Preproces.npy', allow_pickle=True)
    Images = [0, 1, 2, 3, 4]
    for i in range(len(Images)):
        # print(n, i, len(Original))
        Orig = Original[Images[i]]
        Prep = Preprocess[Images[i]]
        Seg = segmented[Images[i]]
        GT = Ground_truth[Images[i]]
        for j in range(1):
            Orig_1 = Seg[j]
            Orig_2 = Seg[j + 1]
            Orig_3 = Seg[j + 2]
            Orig_4 = Seg[j + 3]
            Orig_5 = Seg[j + 4]
            plt.suptitle('Segmented Images', fontsize=20)

            plt.subplot(3, 3, 1).axis('off')
            plt.imshow(GT)
            plt.title('Ground Truth', fontsize=10)

            plt.subplot(3, 3, 2).axis('off')
            plt.imshow(Orig_1)
            plt.title('UNET', fontsize=10)

            plt.subplot(3, 3, 3).axis('off')
            plt.imshow(Orig_2)
            plt.title('RESUnet', fontsize=10)

            plt.subplot(3, 3, 4).axis('off')
            plt.imshow(Orig)
            plt.title('Original', fontsize=10)

            plt.subplot(3, 3, 6).axis('off')
            plt.imshow(Prep)
            plt.title('Preprocess', fontsize=10)

            plt.subplot(3, 3, 7).axis('off')
            plt.imshow(Orig_3)
            plt.title('TransUnet ', fontsize=10)

            plt.subplot(3, 3, 8).axis('off')
            plt.imshow(Orig_4)
            plt.title('Trans-Unet-CycleGAN', fontsize=10)

            plt.subplot(3, 3, 9).axis('off')
            plt.imshow(Orig_5)
            plt.title('Trans-R2Unet++', fontsize=10)

            path = "./Results/Image_Results/Image_%s.png" % (i + 1)
            plt.savefig(path)
            plt.show()

            # cv.imwrite('./Results/Image_Results/Orig_image_' + str(i + 1) + '.png',
            #            Orig)
            # cv.imwrite('./Results/Image_Results/Prep_image_' + str(i + 1) + '.png',
            #            Prep)
            # cv.imwrite('./Results/Image_Results/Ground_Truth_' + str(i + 1) + '.png',
            #            GT)
            # cv.imwrite('./Results/Image_Results/Segm_Unet_' + str(i + 1) + '.png',
            #            Orig_1)
            # cv.imwrite('./Results/Image_Results/Segm_ResUnet_' + str(i + 1) + '.png',
            #            Orig_2)
            # cv.imwrite(
            #     './Results/Image_Results/Segm_TransUnet_' + str(i + 1) + '.png',
            #     Orig_3)
            # cv.imwrite(
            #     './Results/Image_Results/Segm_Trans-Unet-CycleGAN_' + str(i + 1) + '.png',
            #     Orig_4)
            # cv.imwrite(
            #     './Results/Image_Results/Segm_Trans-R2Unet_' + str(i + 1) + '.png',
            #     Orig_5)

def Plot_Confusion():
    from sklearn import metrics
    from sklearn.metrics import confusion_matrix
    Actual = np.load('Actual.npy', allow_pickle=True)
    Predict = np.load('Predict.npy', allow_pickle=True)
    cm = confusion_matrix(np.asarray(Actual).argmax(axis=1), np.asarray(Predict).argmax(axis=1))
    Classes = ['Normal', 'Abnormal']
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(8, 6))
    cm_display.plot(ax=ax, cmap='Blues', values_format='d')

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('Actual labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(Classes)
    ax.yaxis.set_ticklabels(Classes)
    path = "./Results/Confusion_Matrix.png"
    plt.savefig(path)
    plt.show()


def Actual_Pred_values():
    import pandas as pd
    Actual = np.load('Actual.npy', allow_pickle=True)
    Predict = np.load('Predict.npy', allow_pickle=True)
    targ_act = np.argmax(Actual, axis=1).reshape(-1)
    targ_pred = np.argmax(Predict, axis=1).reshape(-1)
    data = {
        "Actual Values": targ_act,
        "Predicted Values": targ_pred,
    }
    df = pd.DataFrame(data)
    output_path = './Results/Actual_vs_Predicted.xlsx'
    df.to_excel(output_path, index=False)


if __name__ == '__main__':
    plot_conv()
    ROC_curve()
    Plot_KFold()
    Plot_Batchsize()
    plot_results_Seg()
    Image_segment_comparision()
    Plot_Confusion()
    # Actual_Pred_values()
    GUI()
