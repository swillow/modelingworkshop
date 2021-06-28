import numpy as np
from pyscf import gto, scf, mp, qmmm, grad, lib
from pyscf.data.nist import BOHR, HARTREE2J, AVOGADRO
from esp import esp_atomic_charges, make_rdm1_with_orbital_response

ang2bohr = 1.0/BOHR
nm2bohr = 10.0/BOHR
bohr2nm = BOHR/10.0
bohr2ang = BOHR

au2kcal_mol = HARTREE2J*AVOGADRO/4184.0
au2kJ_mol = HARTREE2J*AVOGADRO/1000.0
kcal_mol2au = 1.0/au2kcal_mol
kJ_mol2au = 1.0/au2kJ_mol


def _pyscf_mol_build(qm_atm_list, basis_name, charge):
    """
    Generate the atom list in the QM region (pyscf.gto.Mole)
    Parameters
    ----------
    qm_atm_list : list of QM atoms with the following list
        [ ['sym1', (x1, y1, z1)], ['sym2', (x2,y2,z2)], ...]
        Note, the unit of position is in Bohr
    basis_name : string
    charge : int
        The total charge of the QM region
    Returns
    ------
    mol : pyscf.gto.Mole
    """
    from pyscf import gto

    mol = gto.Mole()
    mol.basis = basis_name
    mol.atom = qm_atm_list
    mol.charge = charge
    mol.unit = 'Bohr'
    mol.verbose = 0  # Turn off the print out
    mol.build()

    return mol


def _pyscf_qm(qm_atnm, qm_crds, qm_basis, qm_chg_tot,
              esp_opts={}):

    atm_list = []
    for ia, xyz in enumerate(qm_crds):
        sym = qm_atnm[ia]  # .split(':')[0]
        atm_list.append([sym, (xyz[0], xyz[1], xyz[2])])
        #print(ia, sym, xyz)

    #print('qm_chg_tot', qm_chg_tot)
    qm_mol = _pyscf_mol_build(atm_list, qm_basis, qm_chg_tot)
    mf = scf.HF(qm_mol).run()

    ener_QM = mf.e_tot
    grds_QM = mf.Gradients().kernel()

    dm = mf.make_rdm1()

    esp_chg = esp_atomic_charges(qm_mol, dm, esp_opts)

    return ener_QM, grds_QM, esp_chg


def _pyscf_mm_gradients(qm_mol, qm_dm, mm_mol):

    qm_crds = qm_mol.atom_coords()
    qm_chgs = qm_mol.atom_charges()
    mm_crds = mm_mol.atom_coords()
    mm_chgs = mm_mol.atom_charges()

    # (1) the interaction between QM atoms and MM particles
    dr = qm_crds[:, None, :] - mm_crds
    r = np.linalg.norm(dr, axis=2)
    g = np.einsum('r,R,rRx,rR->Rx', qm_chgs, mm_chgs, dr, r**-3)

    # (2) the interaction between electron density and MM particles
    for i, q in enumerate(mm_chgs):
        with qm_mol.with_rinv_origin(mm_crds[i]):
            v = qm_mol.intor('int1e_iprinv')
        f = (np.einsum('ij,xji->x', qm_dm, v) +
             np.einsum('ij,xij->x', qm_dm, v.conj())) * (-q)
        g[i] += f

    return g


def _pyscf_qm(qm_atnm, qm_crds, qm_basis, qm_chg_tot, l_mp2=False,
              l_esp=False, esp_opts={}):

    atm_list = []
    for ia, xyz in enumerate(qm_crds):
        sym = qm_atnm[ia]
        atm_list.append([sym, (xyz[0], xyz[1], xyz[2])])

    qm_mol = _pyscf_mol_build(atm_list, qm_basis, qm_chg_tot)
    mf = scf.RHF(qm_mol)
    mf.run()

    esp_chg = None
    if l_mp2:
        postmf = mp.MP2(mf).run()
        ener_QM = postmf.e_tot
        grds_QM = postmf.Gradients().kernel()

        if l_esp:
            dm = make_rdm1_with_orbital_response(postmf)
            esp_chg = esp_atomic_charges(qm_mol, dm, esp_opts)
    else:
        ener_QM = mf.e_tot
        grds_QM = mf.Gradients().kernel()
        if l_esp:
            dm = mf.make_rdm1()
            esp_chg = esp_atomic_charges(qm_mol, dm, esp_opts)

    return ener_QM, grds_QM, esp_chg


def _pyscf_qmmm(qm_atnm, qm_crds, qm_basis, qm_chg_tot,
                mm_crds, mm_q_list,
                l_esp=False,
                esp_opts={}):

    atm_list = []
    for ia, xyz in enumerate(qm_crds):
        sym = qm_atnm[ia]  # .split(':')[0]
        atm_list.append([sym, (xyz[0], xyz[1], xyz[2])])
        #print(ia, sym, xyz)

    #print('qm_chg_tot', qm_chg_tot)
    qm_mol = _pyscf_mol_build(atm_list, qm_basis, qm_chg_tot)
    mm_mol = qmmm.create_mm_mol(mm_crds, mm_q_list, unit='Bohr')
    mf = qmmm.qmmm_for_scf(scf.RHF(qm_mol), mm_mol)
    mf.run()

    ener_QMMM = mf.e_tot
    grds_QM = mf.Gradients().kernel()
    qm_dm = mf.make_rdm1()

    grds_MM = _pyscf_mm_gradients(qm_mol, qm_dm, mm_mol)

    esp_chg = None
    if l_esp:
        esp_chg = esp_atomic_charges(qm_mol, qm_dm, esp_opts)

    return ener_QMMM, grds_QM, grds_MM, esp_chg
