import numpy as np


def compute_oscr(pred_k, pred_u, labels, out_curve=False):
    results = {}

    x1, x2 = np.max(pred_k, axis=1), np.max(
        pred_u, axis=1
    )  # known predictions, unknown predictions
    # closed-set classifiation
    pred = np.argmax(pred_k, axis=1)  # get the predictions
    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1

    k_target = np.concatenate(
        (m_x1, np.zeros(len(x2))), axis=0
    )  # known样本的预测目标值
    u_target = np.concatenate(
        (np.zeros(len(x1)), np.ones(len(x2))), axis=0
    )  # unkown的预测目标
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)  # unknown+known样本数量

    # Cutoffs are of prediction values
    CCR = [0 for x in range(n + 2)]
    FPR = [0 for x in range(n + 2)]

    idx = predict.argsort()  # 从小到大排序
    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n - 1):  # k是阈值，概率大于k位置的样本判断为know,否则为unknown
        CC = s_k_target[k + 1 :].sum()
        FP = s_u_target[k:].sum()

        # True Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    fpr95_pos = np.abs(np.asarray(FPR) - 0.05).argmin()
    ccr_at_fpr05 = np.asarray(CCR)[fpr95_pos]

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n + 1] = 1.0
    FPR[n + 1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n + 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0

        OSCR = OSCR + h * w

    results["fpr"] = FPR
    results["ccr"] = CCR
    results["oscr"] = OSCR
    results["ccr@fpr05"] = ccr_at_fpr05
    results["fpr95_pos"] = fpr95_pos

    return results
