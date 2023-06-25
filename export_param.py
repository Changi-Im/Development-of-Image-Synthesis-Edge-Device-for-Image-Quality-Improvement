import torch
from args_fusion import args
import numpy as np
from binary_fractions import Binary

if __name__ == "__main__":
    weights = torch.load(args.model_path_gray)

    n_weights = np.array([])
    str_weights = np.array([])
    idx = []
    w = ['conv1.depthwise.depthwise.weight', 'conv1.pointwise.pointwise.weight', 'DB1.denseblock.0.dense_conv.depthwise.depthwise.weight', 'DB1.denseblock.0.dense_conv.pointwise.pointwise.weight', 'DB1.denseblock.1.dense_conv.depthwise.depthwise.weight', 'DB1.denseblock.1.dense_conv.pointwise.pointwise.weight', 'DB1.denseblock.2.dense_conv.depthwise.depthwise.weight', 'DB1.denseblock.2.dense_conv.pointwise.pointwise.weight', 'conv1_1.depthwise.depthwise.weight', 'conv1_1.pointwise.pointwise.weight', 'DB2.denseblock.0.dense_conv.depthwise.depthwise.weight', 'DB2.denseblock.0.dense_conv.pointwise.pointwise.weight', 'DB2.denseblock.1.dense_conv.depthwise.depthwise.weight', 'DB2.denseblock.1.dense_conv.pointwise.pointwise.weight', 'DB2.denseblock.2.dense_conv.depthwise.depthwise.weight', 'DB2.denseblock.2.dense_conv.pointwise.pointwise.weight', 'conv2.depthwise.depthwise.weight', 'conv2.pointwise.pointwise.weight', 'conv3.depthwise.depthwise.weight', 'conv3.pointwise.pointwise.weight', 'conv4.depthwise.depthwise.weight', 'conv4.pointwise.pointwise.weight', 'conv5.depthwise.depthwise.weight', 'conv5.pointwise.pointwise.weight']
    
    for name in weights.keys():
       n_weights = np.append(n_weights,weights[name].numpy())
        
    np.set_printoptions(precision=32, suppress=True, threshold=np.inf, linewidth=np.inf)
    
    for i in range(len(n_weights)):
        a = Binary(n_weights[i])
        s = a.components()[0]
        integer = a.components()[1]
        fraction = a.components()[2]
        fraction = fraction + "000000000000"
        
        if s == 1:
            sign = '-'
        else:
            sign = ' '
            
        if integer == '1' or integer == '0':
            if sign == '-':
                str_weights = np.append(str_weights, "storage[{0}] = {1}14'b{2}_{3};".format(i, sign, '0'+integer, fraction[:11]))
            else:
                str_weights = np.append(str_weights, "storage[{0}] = 14'b{1}_{2};".format(i, '0'+integer, fraction[:11]))
        else:
            if sign == '-':
                str_weights = np.append(str_weights, "storage[{0}] = {1}14'b{2}_{3};".format(i, sign, integer, fraction[:11]))
            else:
                str_weights = np.append(str_weights, "storage[{0}] = 14'b{1}_{2};".format(i, integer, fraction[:11]))
            
    for name in weights.keys():
        s = 1
        for _ in range(4):
            s *= weights[name].shape[_]
        idx.append(s)
        
    s = 0
    for i in range(len(idx)):
        if i == 0:
            str_weights = np.insert(str_weights, 0, '//'+w[i])
        else:
            s += idx[i-1]
            str_weights = np.insert(str_weights, i + s, '//'+w[i])
            
    
    np.savetxt("./weight(14bit).txt", str_weights, fmt='%s')
    