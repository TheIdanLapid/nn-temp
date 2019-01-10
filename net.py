import numpy as np
from nlib_func import sig, dx_sig
from nlib_dummydata import xor_data
import pdb

class Rigid:
    def __init__(self):
        self.il = np.zeros(shape=(2,1))
        self.w_ih = np.array([[np.random.random() for i in range(3)] for j in range(2)])
        self.hl = np.zeros(shape=(3,1))
        self.w_ho = np.array([[np.random.random() for i in range(1)] for j in range(3)])
        self.ol = np.zeros(shape=(1,1))

# pdb.set_trace()

def shum(op=False):
    # 1 out of 4 possible items
    item = xor_data()
    # forward the item
    f.il = item['data']
    f.hl = [sig(x) for x in np.matmul(np.transpose(f.w_ih), f.il)]
    f.ol = [sig(x) for x in np.matmul(np.transpose(f.w_ho), f.hl)]
    if op:
        print('label:{} guess:{}'.format(item['label'], f.ol))
    # back propagate the error
    guess = f.ol
    out_err = np.array([abs(l-g) for l,g in zip(item['label'], guess)])
    out_grad = np.array([dx_sig(p) for p in f.ol])
    out_grad = np.matmul(out_grad, out_err*0.3)

    h_err = np.matmul(f.w_ho, out_err)
    h_grad = np.array([dx_sig(p) for p in f.hl])
    h_grad = np.matmul(h_grad, h_err*0.3)

    ho_dlt = [og*hlt for og, hlt in zip([out_grad], np.transpose(f.hl))]
    ih_dlt = [hg*ilt for hg, ilt in zip([h_grad], np.transpose(f.il))]
    # update weights
    f.w_ho += ho_dlt
    f.w_ih += ih_dlt


f = Rigid()
# execute 1000 times
d = 0
while d < 1000:
    if d%100 == 4:
        shum(True)
    elif d == 999:
        shum(True)
    else:
        shum()
    d += 1
