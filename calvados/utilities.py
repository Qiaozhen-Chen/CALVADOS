#这份代码基于 numpy 实现三个常用时序处理工具：滑动均值卷积平滑、自相关函数计算、双向窗口滑动平均（运行均值），常用于信号处理、分子动力学轨迹、时间序列降噪分析。

import numpy as np

def xconv(x,N=5):
    xf = np.convolve(x, np.ones(N)/N, mode='same')
    return xf

def autocorr(x,norm=True):
    y = x.copy()
    if norm:
        x = (x - np.mean(x)) / (np.std(x) * len(x))
        y = (y - np.mean(y)) / (np.std(y))
    c = np.correlate(x,y,mode='full')
    c = c[len(c)//2:]
    return c

def calc_runavg(xs,N=10):
    xs_ravg = []
    for idx, x in enumerate(range(len(xs))):
        # if x == np.nan:
        #     xs_ravg.append(np.nan)
        # else:
        x0 = max(0,idx-N)
        x1 = min(len(xs), idx+N+1)
        y = np.nanmean(xs[x0:x1])
        xs_ravg.append(y)
    xs_ravg = np.array(xs_ravg)
    return xs_ravg
