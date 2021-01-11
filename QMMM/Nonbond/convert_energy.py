import numpy as np
from pyscf.data.nist import HARTREE2J, AVOGADRO

au2kcal_mol = HARTREE2J*AVOGADRO/4184.0

lines = open('ener_mon.dat', 'r').readlines()
en_mon1 = float(lines[0].split()[1])
en_mon2 = float(lines[1].split()[1])

lines = open('ener.dat', 'r').readlines()
fout = open('ener_nonbond.dat', 'w', 1)

for line in lines:
    words = line.split()
    r = float(words[0])
    en = float(words[1]) - en_mon1 - en_mon2

    print('%8.5f %8.4f' % (r, en*au2kcal_mol), file=fout)
