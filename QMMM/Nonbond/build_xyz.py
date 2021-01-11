import numpy as np

lines = open('geom.xyz', 'r').readlines()

wat1 = []
wat2 = []
for iat in range(3):
    words = lines[2+iat].split()
    wat1.append([float(words[1]), float(words[2]), float(words[3])])
    words = lines[5+iat].split()
    wat2.append([float(words[1]), float(words[2]), float(words[3])])

wat1 = np.array(wat1)
wat2 = np.array(wat2)
doo = wat2[0] - wat1[0]
roo = np.sqrt(np.einsum('i,i', doo, doo))

dr = 0.1
for iconf in range(12):
    dx = (iconf-4)*dr
    dpos = np.array([dx, 0.0, 0.0])
    fname = 'wat2_iconf'+str(iconf)+'.xyz'
    fout = open(fname, 'w')
    print(' 6 ', file=fout)
    print(' roo = ', roo+dx, file=fout)
    x, y, z = wat1[0]
    print('%4s %10.6f %10.6f %10.6f ' % ('O', x, y, z), file=fout)
    x, y, z = wat1[1]
    print('%4s %10.6f %10.6f %10.6f ' % ('H', x, y, z), file=fout)
    x, y, z = wat1[2]
    print('%4s %10.6f %10.6f %10.6f ' % ('H', x, y, z), file=fout)
    x, y, z = wat2[0] + dpos
    print('%4s %10.6f %10.6f %10.6f ' % ('O', x, y, z), file=fout)
    x, y, z = wat2[1] + dpos
    print('%4s %10.6f %10.6f %10.6f ' % ('H', x, y, z), file=fout)
    x, y, z = wat2[2] + dpos
    print('%4s %10.6f %10.6f %10.6f ' % ('H', x, y, z), file=fout)
    fout.close()
