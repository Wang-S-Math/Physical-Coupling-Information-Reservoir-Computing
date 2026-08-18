import numpy as np
from numpy import zeros
from numpy import tanh, eye, std, var, mean, sqrt

rng = np.random.default_rng()


class rcnp:
    def __init__(self, can1=1, can2=0, can3=0):
        if isinstance(can1, int):
            if can2 == 0:
                pass
            else:
                self.chushi2(can1, can2)
        else:
            self.chushi1(can1, can2, can3)
        self.chshilist()

    @staticmethod
    def fwres(n, xshl=0.03, pbj=1.05):
        ##(0,1)
        wres = rng.random((n, n))
        wres[wres > xshl + 0.005] = 0
        wres[wres < 0.005] = 0
        wres = int(1 / xshl) * wres
        grgrou = np.max(np.abs(np.linalg.eig(wres)[0]))
        if grgrou < 0.1:
            print('原始特征值过小')
        elif grgrou < 1:
            print('特征值为:', grgrou, '未改变.')
        else:
            wres = pbj / grgrou * wres
        return wres

    def setwres(self, n, xshl=0.03, pbj=1.05):
        self.wres = self.fwres(n, xshl, pbj)

    @staticmethod
    def fwin3(n, inwei1, inw2=3, inw3=1, grsi=1, oo=0.5, leixin=0):
        ##分块输入矩阵，占比不同，等比例
        win = zeros((n, inwei1))
        ns = n // inwei1
        l1 = (inwei1 - inw2) // inw3
        win1 = rcnp.fwin(ns * inw2, inw2, grsi, oo)
        win[:ns * inw2, :inw2] = win1
        for ii in range(l1):
            win1 = rcnp.fwin(ns * inw3, inw3, grsi, oo)
            win[ns * inw2 + ii * (ns * inw3):ns * inw2 + (ii + 1) * (
                        ns * inw3), inw2 + ii * inw3:inw2 + ii * inw3 + inw3] = win1
        sl1 = n % inwei1
        if l1 == 0:
            sl2 = 0
        else:
            sl2 = (inwei1 - inw2) % l1
        if leixin == 0:
            if sl1 != 0:
                win[-sl1:] = 0.5 * grsi
            if sl2 != 0:
                win[:, -sl2:] = 0.5 * grsi
        else:
            if sl1 != 0:
                win[-sl1:, :inw2] = rcnp.fwin(sl1, inw2, grsi, oo)
        return win

    @staticmethod
    def fwin(n, inwei, grsi=1, oo=0.5):
        win = rng.random((n, inwei))
        if oo != 0:
            win = 2 * (win - oo)
        win = grsi * win
        return win

    def setwin(self, n, inwei, grsi=1, oo=0.5):
        self.win = self.fwin(n, inwei, grsi, oo)

    def chushi1(self, win, a, b):
        self.wres = np.array(a)
        self.win = np.array(win)
        self.bk = np.array(b)

    @staticmethod
    def fbk(nn, oo1=0, oo2=0):
        if oo1 == 0:
            bk = 0
        else:
            bk = oo1 * (rng.random((nn)) - oo2)
        return bk

    def chushi2(self, inwei, n):
        self.setwres(n)
        self.setwin(n, inwei)
        self.bk = self.fbk(n)

    def chshilist(self):
        self.out = None
        self.r0 = None
        self.rr = None
        self.RR = None
        self.grg = None
        self.UU = None
        self.wu = None
        self.xx = None
        self.yy = None

    def train(self, xx: np.ndarray, grg=0.7, r0=0, type=0):
        self.grg = grg
        if np.abs(np.max(self.win) * np.max(xx)) > 3:
            print("可能未调节win")
        self.RR = zeros((xx.shape[0], self.wres.shape[0]))
        if isinstance(r0, int):
            if isinstance(self.r0, np.ndarray):
                pass
            else:
                r0 = self.fxulex00(xx, r0, type)
                self.r0 = r0
        else:
            self.r0 = r0
        rr = self.r0.copy()
        if xx.ndim == 1:
            w_in1 = np.squeeze(self.win)
            for ii in range(xx.shape[0]):
                rr = (1 - grg) * rr + grg * tanh(w_in1 * xx[ii] + self.wres @ rr + self.bk)
                self.RR[ii] = rr
        elif xx.ndim == 2:
            for ii in range(xx.shape[0]):
                rr = (1 - grg) * rr + grg * tanh(self.win @ xx[ii] + self.wres @ rr + self.bk)
                self.RR[ii] = rr
        else:
            print('维度不对')
        self.rr = rr.copy()
        return self.RR

    def fxulex00(self, xx, n=1000, type=0):
        if type == 1:
            r0 = rng.random((self.wres.shape[0])) - 0.5
            if len(xx) < n:
                x_x = xx - xx[0]
                xx00 = xx[0] - x_x
            else:
                xx00 = 2 * xx[0] - xx[:n]
            xx00 = xx00[::-1]
            if xx.ndim == 1:
                w_in1 = np.squeeze(self.win)
                for ii in range(xx00.shape[0]):
                    r0 = (1 - self.grg) * r0 + self.grg * tanh(w_in1 * xx00 + self.wres @ r0 + self.bk)
            elif xx.ndim == 2:
                for ii in range(xx00.shape[0]):
                    r0 = (1 - self.grg) * r0 + self.grg * tanh(self.win @ xx00 + self.wres @ r0 + self.bk)
            else:
                print('维度不对')
        elif type == 2:
            r0 = rng.random((self.wres.shape[0])) - 0.5
            if xx.ndim == 1:
                w_in1 = np.squeeze(self.win)
                for ii in range(n):
                    r0 = (1 - self.grg) * r0 + self.grg * tanh(w_in1 * xx[0] + self.wres @ r0 + self.bk)
            elif xx.ndim == 2:
                for ii in range(n):
                    r0 = (1 - self.grg) * r0 + self.grg * tanh(self.win @ xx[0] + self.wres @ r0 + self.bk)
            else:
                print('维度不对')
        else:
            r0 = 2 * (rng.random((self.wres.shape[0])) - 0.5)
        return r0

    def frun(self, uu, rr):
        rr = (1 - self.grg) * rr + self.grg * tanh(self.win @ uu + self.wres @ rr + self.bk)
        return rr

    def getwoutshe(self, xx, she=300, grla=1e-8, dd=0):
        self.getwout(xx[she:], self.RR[she - 1:-1], grla)
        if dd == 1:
            self.bijiao(xx[she:], self.yy)

    def getwout(self, xx, RR, grla=1e-8):
        wout = xx.T @ RR @ np.linalg.pinv(RR.T @ RR + grla * eye(RR.shape[1]))
        wout1 = xx.T @ RR @ np.linalg.inv(RR.T @ RR + grla * eye(RR.shape[1]))
        self.out = wout
        self.out1 = wout1
        self.yy = RR @ wout.T
        self.wu = xx - self.yy
        return wout

    def bijiao(self, xx, yy):
        import matplotlib
        from matplotlib.pyplot import plot, figure, legend, rcParams
        matplotlib.rcParams['font.family'] = 'SimHei'
        rcParams['axes.unicode_minus'] = False
        for ii in range(xx.shape[1]):
            figure()
            plot(xx[:, ii], label="目标")
            plot(yy[:, ii], label="预测")
            legend()

    def yuce1(self, rr, n=2000):
        if self.out.ndim == 1:
            UU = zeros(n)
            w_in1 = np.squeeze(self.win)
            for ii in range(n):
                uu = self.out @ rr
                rr = (1 - self.grg) * rr + self.grg * tanh(self.wres @ rr + w_in1 * uu + self.bk)
                UU[ii] = uu
            self.UU = UU
            return UU
        UU = zeros((n, self.out.shape[0]))
        for ii in range(n):
            uu = self.out @ rr
            rr = (1 - self.grg) * rr + self.grg * tanh(self.wres @ rr + self.win @ uu + self.bk)
            UU[ii] = uu
        self.UU = UU
        return UU

    def fbijiao(self, x1, x2, bi: float = 0.5):
        '''论文高阶格兰杰,全局预测'''
        if x1.shape[0] < x1.shape[1]:
            x1 = x1.T
        if x2.shape[0] < x2.shape[1]:
            x2 = x2.T
        if x1.shape[0] < x2.shape[0]:
            nn = x1.shape[0]
        else:
            nn = x2.shape[0]
        stdx = std(x2, 0)
        ce = ((x1 - x2) / stdx) ** 2
        rmse = sqrt(mean(ce, axis=1))
        for ii in range(nn):
            jieguo = ii
            if rmse[ii] > bi:
                print(ii)
                break
        return (jieguo)