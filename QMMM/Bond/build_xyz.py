import numpy as np

roh0 = 0.950
theta0 = 104.8028
radian = 57.29577951308232088

theta = theta0/radian
cn = np.cos(0.5*theta)
sn = np.sin(0.5*theta)


zero = 0.0
dr = 0.005
for iconf in range(16):
    roh = iconf*dr + (roh0 - 0.01)
    hx = roh*sn
    hy = -roh*cn

    fname = 'wat_roh_'+str(roh)+'.xyz'
    fout = open(fname, 'w')
    print(' 3 ', file=fout)
    print(' roh =', roh, file=fout)
    print('%4s %10.6f %10.6f %10.6f ' % ('O', zero, zero, zero), file=fout)
    print('%4s %10.6f %10.6f %10.6f ' % ('H', hx, -hy, zero), file=fout)
    print('%4s %10.6f %10.6f %10.6f ' % ('H', -hx, -hy, zero), file=fout)
    fout.close()
