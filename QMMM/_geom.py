import sys
from berny import geomlib
from pyscf.data.nist import BOHR, HARTREE2J, AVOGADRO
#from simtk.openmm.openmm import AngstromsPerNm
import numpy as np

ang2bohr = 1.0/BOHR


def write_pdb(fname_pdb, fname_pdb_new, crds, box=None):
    '''
    crds : np.array ( (natom, 3))
    '''
    f_pdb = open(fname_pdb)
    l_pdb = f_pdb.read().split('\n')
    f_pdb.close()

    fout = open(fname_pdb_new, 'w', 1)

    iat = 0
    for line in l_pdb[:-1]:
        if line[:6] in ['ATOM  ', 'HETATM']:
            x, y, z = crds[iat]  # .value_in_unit(unit.angstrom)
            sx = ("%8.3f" % x)[:8]
            sy = ("%8.3f" % y)[:8]
            sz = ("%8.3f" % z)[:8]
            sline = line[:30]+sx+sy+sz+line[54:]
            print(sline, file=fout)
            iat += 1
        elif line[:6] in ['CRYST1']:
            if box is None:
                print(line, file=fout)
            else:
                x, y, z = box[0][0], box[1][1], box[2][2]
                sx = ("%9.3f" % x)[:9]
                sy = ("%9.3f" % y)[:9]
                sz = ("%9.3f" % z)[:9]
                sline = line[:6]+sx+sy+sz+line[33:]
                print(sline, file=fout)
        else:
            print(line, file=fout)
    fout.close()


def _load_geom(fname, qmResNames={}, l_protein=False):

    ext = fname.split('.')[-1]
    if ext != 'pdb':
        print("The pdb file is supported for the geometry")
        sys.exit()

    l_pdb = open(fname, 'r').readlines()

    species = []
    coords = []
    qm2mm_idx = {}  # index from QM atoms to MM atoms
    # for resName in qmResNames:
    #    qm_atoms[resName] = []

    index = 0

    for line in l_pdb:
        if line[:6] in ['ATOM  ', 'HETATM']:
            atomName = line[12:16].strip()
            resName = line[17:20].strip()
            resNumber = line[22:26].strip()
            sym = line[76:78].strip()

            words = line[30:].split()
            coords.append([float(x) for x in words[0:3]])
            species.append(sym)

            resId = resName+resNumber
            atmId = resId + ':' + atomName
            if resId in qmResNames:
                if l_protein:
                    if atomName not in ['O',  'H', 'OXT', 'H1', 'H2', 'H3']:
                        """ 
                            Peptide bonds, N-terminus, and C-terminus are not added into a QM residue 
                            CB--CA will be replaced with CB--H (H is a linked atom)
                            We add 'C', 'N', and 'HA' into the QM region since they are covalently bonded to 'CA'.
                            We will not estimate the Coulomb interaction and vdW interaction 
                            between them ('C', 'N', 'HA') and the QM particles.
                        """
                        qm2mm_idx[atmId] = index
                else:
                    qm2mm_idx[atmId] = index
            index += 1

    return geomlib.Geometry(species, coords), qm2mm_idx


def get_qm_geom(lig_geom, qm2mm_idx, qmatm_idx, prt_geom):
    '''
    Get the QM geom from ligand and protein geometry (QM regions are on both ligand and protein)
    lig_geom (geomlib.Geometry): ligand (sym, xyz)
    qm2mm_idx (dict): atmID = mm_idx
    qmatm_idx (dict): atmID = qm_idx
    prt_geom (geomlib.Geometry): protein (sym, xyz)
    '''
    #import copy

    qm_species = []  # copy.copy(lig_geom.species)
    qm_crds = []  # copy.copy(list(lig_geom.coords))
    pos_ca = {}
    pos_cb = {}

    for atmId in qm2mm_idx:
        resId, atomName = atmId.split(':')
        mm_idx = qm2mm_idx[atmId]
        qm_idx = qmatm_idx[atmId]

        if resId in ['LIG1']:
            x, y, z = lig_geom.coords[mm_idx]
            sym = lig_geom.species[mm_idx]
        else:
            x, y, z = prt_geom.coords[mm_idx]
            sym = prt_geom.species[mm_idx]
            if atomName == 'CA':
                pos_ca[resId] = np.array([x, y, z])
            if atomName == 'CB':
                pos_cb[resId] = np.array([x, y, z])

        qm_species.append(sym)
        qm_crds.append([x, y, z])

    for resId in pos_ca:
        pab = pos_ca[resId] - pos_cb[resId]
        rab = np.sqrt(np.einsum('i,i', pab, pab))
        x, y, z = pos_cb[resId] + pab/rab
        atmId = resId + ':' + 'CA'

        idx = qmatm_idx[atmId]
        qm_species[idx] = 'H'
        #print(idx, 'before', qm_crds[idx], 'after', x, y, z)
        qm_crds[idx] = [x, y, z]

    #print('qm_species', qm_species)
    #print('qm_crds', qm_crds)
    # sys.exit()

    return geomlib.Geometry(qm_species, qm_crds)
