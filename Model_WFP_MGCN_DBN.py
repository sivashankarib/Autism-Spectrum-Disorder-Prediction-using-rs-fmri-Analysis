from Model_DBN import Model_DBN
from Model_GCN import Model_GCN_Feat
from sklearn.model_selection import train_test_split


def Model_WFP_MGCN_DBN(Data, Target, BS=None, sol=None):
    if sol is None:
        sol = [5, 5, 1]
    if BS is None:
        BS = 4
    Feature = Model_GCN_Feat(Data, Target, BS)
    X_train, X_test, y_train, y_test = train_test_split(Feature, Target, random_state=104, test_size=0.25, shuffle=True)
    Eval, Pred = Model_DBN(X_train, y_train, X_test, y_test, BS, sol=sol)
    return Eval, Pred

