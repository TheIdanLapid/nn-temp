import numpy as np
from nlib_func import sig, dx_sig
from nlib_dummydata import xor_data
import pdb

class Rigid:
    def __init__(self):
        self.il = np.zeros(shape=(2,1))
        self.w_ih = np.array([[np.random.random() * 2 - 1 for i in range(3)] for j in range(2)])
        self.hl = np.zeros(shape=(3,1))
        self.w_ho = np.array([[np.random.random() * 2 - 1 for i in range(1)] for j in range(3)])
        self.ol = np.zeros(shape=(1,1))

# pdb.set_trace()

def shum(op=False, fullReport = False):
    # 1 out of 4 possible items
    item = xor_data()
    # forward the item
    f.il = item['data']
    f.hl = [sig(x) for x in np.matmul(np.transpose(f.w_ih), f.il)]
    f.ol = [sig(x) for x in np.matmul(np.transpose(f.w_ho), f.hl)]
    # back propagate the error
    guess = f.ol
    out_err = np.array([(l-g) for l,g in zip(item['label'], guess)])
    out_grad = np.array([dx_sig(p) for p in guess])
    out_grad = np.matmul(out_grad, out_err*0.01)

    h_err = np.matmul(f.w_ho, out_err)
    h_grad = np.array([dx_sig(p) for p in f.hl])*h_err*0.01


    ho_dlt = np.array([[og * hlt for og in [out_grad]] for hlt in f.hl])
    ih_dlt = np.array([[hg * ilt  for hg in h_grad] for ilt in f.il])

    if op:
        print('label:{} guess:{} err:{}'.format(item['label'], f.ol, out_err))
    if fullReport:
        print(' --- --- --- ---')
        print(f'out_err:{out_err}\nout_grad:{out_grad}\nh_err:{h_err}\nh_grad:{h_grad}\nho_dlt:{ho_dlt}\nih_dlt:{ih_dlt}')
        print(' --- --- --- ---')
        print(f'f.il{f.il}\nf.w_ih{f.w_ih}\nf.hl{f.hl}\nf.w_ho{f.w_ho}\nf.ol{f.ol}')
    # update weights
    f.w_ho += ho_dlt
    f.w_ih += ih_dlt


f = Rigid()
# execute 1000 times
d = 0
while d < 10000:
    if d%1000 == 4:
        shum(True)
    elif d == 9999:
        shum(True, True)
    else:
        shum()
    d += 1
