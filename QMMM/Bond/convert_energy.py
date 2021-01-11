import numpy as np
from pyscf.data.nist import HARTREE2J, AVOGADRO

au2kcal_mol = HARTREE2J*AVOGADRO/4184.0

lines = open('ener.dat', 'r').readlines()
en_opt = float(lines[0].split()[1])

fout = open('ener_bond.dat', 'w', 1)

for line in lines[1:]:
    words = line.split()
    r = float(words[0])
    en = float(words[1]) - en_opt

    print('%8.5f %8.4f' % (r, en*au2kcal_mol), file=fout)
