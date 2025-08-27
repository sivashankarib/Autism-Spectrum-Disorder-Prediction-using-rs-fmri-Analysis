import numpy as np
from Evaluation import evaluation
from Global_Vars import Global_Vars
from Model_WFP_MGCN_DBN import Model_WFP_MGCN_DBN


def objfun_cls(Soln):
    Feat = Global_Vars.Feat
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(Feat.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Test_Target = Tar[learnper:, :]
            Eval, pred = Model_WFP_MGCN_DBN(Feat, Tar, sol=sol)
            Eval = evaluation(pred, Test_Target)
            Fitn[i] = 1 / (Eval[4] + Eval[7] + Eval[11])  # (Accuracy + Precision + NPV)
        return Fitn
    else:
        learnper = round(Feat.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Test_Target = Tar[learnper:, :]
        Eval, pred = Model_WFP_MGCN_DBN(Feat, Tar, sol=sol)
        Eval = evaluation(pred, Test_Target)
        Fitn = 1 / (Eval[4] + Eval[7] + Eval[11])  # (Accuracy + Precision + NPV)
        return Fitn