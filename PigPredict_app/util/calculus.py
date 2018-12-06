import numpy as np 

def numerical_gradient(f, x, h=1e-4, idxs=None):
    
    # 通常我們希望對 input nparray 做 mutate 而不是創一個新的，所以 copy=False
    x = x.astype(np.float64, copy=False)
    
    grad = np.zeros_like(x)
    
    # flags 須加入 multi_index 才有多維index可用，op_flags則是指定我們對該array的操作權限
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite']) # a multi-dimensional iterator (to iterate over an array)
    while not it.finished:
        idx = it.multi_index 
        
        if idxs is not None and idx not in idxs:
            it.iternext()
            continue
            
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        x[idx] = float(tmp_val) - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
        it.iternext()   
        
    return grad
