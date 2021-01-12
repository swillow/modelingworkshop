#!/usr/bin/env python

'''
Write orbitals, electron density, molecular electrostatic potential in
Gaussian cube file format.
'''

from pyscf import gto, scf, qmmm
from pyscf.tools import cubegen
from pyscf.data.nist import BOHR
import numpy as np

ang2bohr = 1.0/BOHR

mol = gto.M(atom='''
O   0.0000000000000000        0.0000000000000000        0.000
H  -0.26312631683261728       0.92177640310981912       0.000
H   0.9645910938303689        4.0006988432649347E-002   0.000
''',
            basis='aug-cc-pVDZ')


mm_crd = np.array([[2.9125407227330018,   0.0000000000000000,    0.000],
                   [3.3354011353264896,  -0.41314678971687741,  -0.75710308103585677],
                   [3.3354011558614673,  -0.41314667699843621,   0.75710313896414105]])
mm_crd *= ang2bohr
mm_chgs = np.array([-0.68, 0.34, 0.34])

mf = qmmm.mm_charge(scf.RHF(mol), mm_crd, mm_chgs).run()
# electron density
cubegen.density(mol, 'h2o_den_qmmm.cube', mf.make_rdm1())

# molecular electrostatic potential
cubegen.mep(mol, 'h2o_pot_qmmm.cube', mf.make_rdm1())

# 1st MO
nhomo = mf.mol.nelec[0]
cubegen.orbital(mol, 'h2o_homo_qmmm.cube', mf.mo_coeff[:, nhomo-1])
