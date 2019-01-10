import numpy as np
from nlib_dummydata import xor_data
from nlib_func import sig, dx_sig
from pprint import pprint as pprint
import pdb

class nLayer:
    def __init__(self, dim):
        self.values = np.zeros(shape=(dim,1))


class wLayer:
    def __init__(self, idim, odim):
        self.w = [[np.random.random() for p in range(odim)] for i in range(idim)]


nodes = [nLayer(2), nLayer(3), nLayer(1)]
weights = [wLayer(2,3), wLayer(3,1)]
# biases = [np.ones(shape=(2,)), np.ones(shape=(3,)), np.ones(shape=(1,))]

err_arr = []

def one(prt):
    item = xor_data()
    nodes[0].values = item['data']
    nodes[1].values = np.array([sig(j) for j in np.matmul(np.transpose(weights[0].w), nodes[0].values)])
    nodes[2].values = np.array([sig(n) for n in np.matmul(np.transpose(weights[1].w),  nodes[1].values)])
    if prt:
        print('my guess for {} is {}'.format(item['label'], nodes[2].values))
    out_err = item['label'] - nodes[2].values
    err_arr.append(out_err)
    out_grad = np.array([dx_sig(p) for p in nodes[2].values])
    out_grad = np.matmul(out_grad ,out_err * 0.3)
    h_err = weights[1].w * out_err
    h_grad = [dx_sig(p) for p in nodes[1].values]
    h_grad = np.matmul(h_grad, h_err * 0.3)
    out_dlt = np.matmul(out_grad, np.transpose(nodes[1].values))
    pdb.set_trace()
    h_dlt = np.matmul(h_grad*-1, [nodes[0].values])
    weights[0].w += h_dlt
    weights[1].w += out_dlt

i=0
while i < 1000:
    if i%100 == 4:
        one(True)
    else:
        one(False)
    i += 1


pprint(err_arr[0:10:1])
print('\n ------------')
pprint(err_arr[-10:-1:1])














stopit = 7
