def modify_pred(pred):
    # TODO
    # 在 setting.cfg 增加濃度規範 (染料之上下限等)
    # 並以濃度規範調整濃度

    pred_ = pred.copy()
    pred_[(pred_ < 5e-3) & (pred_ != 0)] = 1e-2
    pred_[pred_ > 3] = 3
    return pred_
