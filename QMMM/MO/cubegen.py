#!/usr/bin/env python

'''
Write orbitals, electron density, molecular electrostatic potential in
Gaussian cube file format.
'''

from pyscf import gto, scf
from pyscf.tools import cubegen

mol = gto.M(atom='''
O   0.0000000000000000        0.0000000000000000        0.000
H  -0.26312631683261728       0.92177640310981912       0.000
H   0.9645910938303689        4.0006988432649347E-002   0.000
''',
basis='aug-cc-pVDZ')
mf = scf.RHF(mol).run()

# electron density
cubegen.density(mol, 'h2o_den.cube', mf.make_rdm1())

# molecular electrostatic potential
cubegen.mep(mol, 'h2o_pot.cube', mf.make_rdm1())

# 1st MO
nhomo = mf.mol.nelec[0]
cubegen.orbital(mol, 'h2o_homo.cube', mf.mo_coeff[:,nhomo-1])

