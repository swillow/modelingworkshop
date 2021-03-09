import numpy as np
import simtk.unit as unit
from simtk.openmm import app
import simtk.openmm as omm
from esp import esp_atomic_charges, make_rdm1_with_orbital_response

import sys
import time
from pyscf import gto, scf, mp, qmmm, grad, lib
from pyscf.data.nist import BOHR, HARTREE2J, AVOGADRO

from berny import Berny, geomlib


ang2bohr = 1.0/BOHR
nm2bohr = 10.0/BOHR
bohr2nm = BOHR/10.0
bohr2ang = BOHR

au2kcal_mol = HARTREE2J*AVOGADRO/4184.0
au2kJ_mol = HARTREE2J*AVOGADRO/1000.0
kcal_mol2au = 1.0/au2kcal_mol
kJ_mol2au = 1.0/au2kJ_mol


_nonbond_list = {}
_nonbond_list["C"] = {"epsilon": 0.359824*kJ_mol2au, "sigma": 0.34*nm2bohr}
_nonbond_list["H"] = {"epsilon": 0.0656888*kJ_mol2au, "sigma": 0.107*nm2bohr}
_nonbond_list["N"] = {"epsilon": 0.71128*kJ_mol2au, "sigma": 0.325*nm2bohr}
_nonbond_list["O"] = {"epsilon": 0.87864*kJ_mol2au, "sigma": 0.296*nm2bohr}
_nonbond_list["S"] = {"epsilon": 1.046*kJ_mol2au, "sigma": 0.35636*nm2bohr}

_r_CC = 1.526*ang2bohr
_k_CC = 0.5*259407.99999999994*kJ_mol2au/nm2bohr**2
_r_CCC = np.arccos(-0.5)
_k_CCC = 0.5*669.44*kJ_mol2au


def _gradient_on_mm_particles(mol, mm_mol, dm):
    # The interaction between QM atoms and MM particles
    # \sum_K d/dR (1/|r_K-R|) = \sum_K (r_K-R)/|r_K-R|^3
    # dm = mf_qmmm.make_rdm1()
    qm_coords = mol.atom_coords()
    qm_charges = mol.atom_charges()
    coords = mm_mol.atom_coords()
    charges = mm_mol.atom_charges()

    dr = qm_coords[:, None, :] - coords
    r = np.linalg.norm(dr, axis=2)
    g = np.einsum('r,R,rRx,rR->Rx', qm_charges, charges, dr, r**-3)

    # The interaction between electron density and MM particles
    # d/dR <i| (1/|r-R|) |j> = <i| d/dR (1/|r-R|) |j> = <i| -d/dr (1/|r-R|) |j>
    #   = <d/dr i| (1/|r-R|) |j> + <i| (1/|r-R|) |d/dr j>
    for i, q in enumerate(charges):
        with mol.with_rinv_origin(coords[i]):
            v = mol.intor('int1e_iprinv')
        f = (np.einsum('ij,xji->x', dm, v) +
             np.einsum('ij,xij->x', dm, v.conj())) * -q
        g[i] += f

    return g


def _load_geom_from_pdb(fname):

    l_pdb = open(fname).readlines()

    species = []
    coords = []
    for line in l_pdb:
        if line[:6] in ['ATOM  ', 'HETATM']:
            atnm = line[12:16].strip()
            rsnm = line[17:20].strip()
            words = line[30:].split()
            coords.append([float(x) for x in words[0:3]])

            if atnm != rsnm:
                species.append(atnm[0])
            elif atnm in ['CL', 'NA', 'MG', 'BE', 'LI', 'K', 'ZN']:
                species.append(atnm)
            else:
                print("Sorry, add atom name", atnm)
                sys.exit()

    return geomlib.Geometry(species, coords)


def _load_geom(fname):

    ext = fname.split('.')[-1]
    if ext == 'pdb':
        geom = _load_geom_from_pdb(fname)
    elif ext == 'xyz':
        geom = geomlib.readfile(fname)
    else:
        print("Not support load_geom ", fname)

    return geom


def _geom_to_crd(geom):
    # A -> Bohr
    return geom.coords*ang2bohr


def _geom_store(geom, cycle, fname):

    fout = open(fname, 'w', 1)
    natom = len(geom.coords)
    print(' %5d' % natom, file=fout)
    print(' %5d' % cycle, file=fout)
    for i in range(natom):
        x, y, z = geom.coords[i]
        print('%4s %10.6f %10.6f %10.6f' %
              (geom.species[i], x, y, z), file=fout)

    fout.close()


def _bond_pot_grad(pos_i, pos_j):
    dij = pos_i - pos_j
    rij = np.sqrt(np.einsum('i,i', dij, dij))
    delta = rij - _r_CC
    enr = _k_CC * delta*delta
    dedr = 2.0*_k_CC*delta
    de = dedr/rij
    gij = de*dij

    return enr, gij


def _angle_pot_grad(pos_i, pos_j, pos_k):
    dij = pos_i - pos_j
    dkj = pos_k - pos_j
    dpp = np.cross(dkj, dij)

    rij = np.sqrt(np.einsum('i,i', dij, dij))
    rkj = np.sqrt(np.einsum('i,i', dkj, dkj))
    rpp = np.sqrt(np.einsum('i,i', dpp, dpp))

    cs = np.einsum('i,i', dij, dkj)/(rij*rkj)
    cs = min(1.0, max(-1.0, cs))
    theta = np.arccos(cs)
    delta = theta - _r_CCC

    enr = _k_CCC * delta*delta
    dedt = 2.0*_k_CCC * delta
    termi = -dedt/(rij*rij*rpp)
    termk = dedt/(rkj*rkj*rpp)

    gi = termi*np.cross(dij, dpp)
    gk = termk*np.cross(dkj, dpp)

    return enr, gi, -(gi+gk), gk


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


def _pyscf_qm(atnm_list, qm_crds, qm_basis, qm_chg_tot,
              l_mp2=False, l_esp=False, esp_opts={}):

    atm_list = []
    for ia, xyz in enumerate(qm_crds):
        atm_list.append([atnm_list[ia], (xyz[0], xyz[1], xyz[2])])

    qm_mol = _pyscf_mol_build(atm_list, qm_basis, qm_chg_tot)
    mf = scf.HF(qm_mol).run()

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

    if l_esp:
        return ener_QM, grds_QM, esp_chg
    else:
        return ener_QM, grds_QM


def _pyscf_qmmm(atnm_list, qm_crds, qm_basis, qm_chg_tot,
                mm_crds, mm_chg, l_mp2=False, l_esp=False, esp_opts={}):

    atm_list = []
    for ia, xyz in enumerate(qm_crds):
        atm_list.append([atnm_list[ia], (xyz[0], xyz[1], xyz[2])])

    qm_mol = _pyscf_mol_build(atm_list, qm_basis, qm_chg_tot)
    mf = qmmm.mm_charge(scf.HF(qm_mol), mm_crds, mm_chg).run()

    if l_mp2:
        postmf = mp.MP2(mf).run()
        ener_QMMM = postmf.e_tot
        grds_QM = postmf.Gradients().kernel()
        dm = make_rdm1_with_orbital_response(postmf)
        grds_MM = _gradient_on_mm_particles(mf.mol, mf.mm_mol, dm)
    else:
        ener_QMMM = mf.e_tot
        grds_QM = mf.Gradients().kernel()
        dm = mf.make_rdm1()
        grds_MM = _gradient_on_mm_particles(mf.mol, mf.mm_mol, dm)

    if l_esp:
        esp_chg = esp_atomic_charges(qm_mol, dm, esp_opts)
        return ener_QMMM, grds_QM, grds_MM, esp_chg
    else:
        return ener_QMMM, grds_QM, grds_MM


def _openmm_energrads(simulation, mm_xyz):

    simulation.context.setPositions(mm_xyz)
    state = simulation.context.getState(
        getEnergy=True, getForces=True)
    ener_MM = state.getPotentialEnergy().value_in_unit(
        unit.kilocalorie_per_mole)*kcal_mol2au
    frc_MM = state.getForces(asNumpy=True).value_in_unit(
        unit.kilocalorie_per_mole/unit.angstrom)*(kcal_mol2au/ang2bohr)

    return ener_MM, -frc_MM


class QMMMSolver(object):

    def __init__(self,
                 opts):

        theory = "qm"
        if "theory" in opts:
            theory = opts["theory"]

        self._l_qm = False
        self._l_mm = False
        self._l_qmpol = False

        if theory == "qm":
            self._l_qm = True
        elif theory == "qmmm":
            self._l_qm = True
            self._l_mm = True
        elif theory == "qmmm-pol":
            self._l_qmpol = True
            self._l_mm = True
        elif theory == "mm":
            self._l_mm = True
            print('Not support')
            sys.exit(5)

        if self._l_qm or self._l_qmpol:
            self.qm_init(opts["qm"])
        else:
            print("No QM region")
            sys.exit()

        if self._l_mm:
            self.mm_init(opts["mm"])

        self._job = "ener"
        if "job" in opts:
            self._job = opts["job"]

        self._geomopt = None
        if self._job in ["geomopt", "opt", "gopt"]:
            self._geomopt = opts["geomopt"]

    def qm_init(self, qm_opts):

        self._l_mp2 = False
        if "method" in qm_opts:
            if qm_opts["method"] in ['mp2', 'MP2']:
                self._l_mp2 = True

        self._qm_basis = "sto-3g"
        if "basis" in qm_opts:
            self._qm_basis = qm_opts["basis"]

        self._qm_geom = None
        if "fname_geom" in qm_opts:
            self._qm_geom = _load_geom(qm_opts["fname_geom"])
        else:
            print("No QM geom")
            sys.exit()
        self._qm_natom = len(self._qm_geom)

        self._l_esp = False
        self._esp_opts = {}
        if "esp" in qm_opts:
            self._l_esp = qm_opts["esp"]
            self._esp_opts = qm_opts["esp_opts"]

        self._qm_linked_atoms = []
        self._qm_neigh_atoms = []
        if "linked_atom_list" in qm_opts:
            self._qm_linked_atoms = qm_opts["linked_atom_list"]
            if "linked_atom_neigh" in qm_opts:
                self._qm_neigh_atoms = qm_opts["linked_atom_neigh"]

        self._qm_chg_tot = 0
        if "charge" in qm_opts:
            self._qm_chg_tot = qm_opts["charge"]

        self._qm_constraints = []
        if "constraints_list" in qm_opts:
            if len(qm_opts["constraints_list"]) != 0:

                # Unit Conversion
                for ia, ja, kij0, rij0 in qm_opts["constraints_list"]:
                    kij = kij0*kcal_mol2au/(ang2bohr*ang2bohr)
                    rij = rij0*ang2bohr
                    self._qm_constraints.append([ia, ja, kij, rij])

        self._qm_nonbond = []

        self._qm_atnm = []
        const_2_6 = 2.0**(1.0/6.0)
        for elem in self._qm_geom.species:
            self._qm_atnm.append(elem)
            eps = 0.0
            sig = 1.0
            if elem in _nonbond_list:
                eps = _nonbond_list[elem]["epsilon"]
                sig = const_2_6*_nonbond_list[elem]["sigma"]
            self._qm_nonbond.append([sig, eps])
        self._qm_nonbond = np.array(self._qm_nonbond)

    def mm_init(self, mm_opts):

        self._mm_geom = None
        if "fname_geom" in mm_opts:
            self._mm_geom = _load_geom(mm_opts["fname_geom"])

        self._mm_prm = None
        if "fname_prmtop" in mm_opts:
            self._mm_prm = app.AmberPrmtopFile(mm_opts["fname_prmtop"])
        else:
            print("MM: No prmtop file")
            sys.exit()

        self._mm_chg = self._mm_prm._prmtop.getCharges()
        self._mm_natom = self._mm_prm._prmtop.getNumAtoms()
        self._mm_nonbond = np.array(self._mm_prm._prmtop.getNonbondTerms())

        self._mm_linked_atoms = []
        self._mm_linked_atom2 = {}

        if "linked_atom_list" in mm_opts:
            self._mm_linked_atoms = mm_opts["linked_atom_list"]

            for ia in self._mm_linked_atoms:
                self._mm_chg[ia-1] = 0.0
                self._mm_nonbond[ia-1, 1] = 0.0

            if "linked_atom_list2" in mm_opts:

                lnk2 = mm_opts["linked_atom_list2"]
                self._mm_linked_atom2 = lnk2

                for ia in self._mm_linked_atoms:
                    for ja in lnk2[str(ia)]:
                        self._mm_chg[ja-1] = 0.0
                        self._mm_nonbond[ja-1, 1] = 0.0

            if "linked_atom_list3" in mm_opts:
                lnk3 = mm_opts["linked_atom_list3"]
                for ia in self._mm_linked_atoms:
                    for ja in lnk3[str(ia)]:
                        self._mm_chg[ja-1] = 0.0
                        self._mm_nonbond[ja-1, 1] = 0.0

        self._qmmm_constraints = []
        if "qmmm_constraints_list" in mm_opts:
            if len(mm_opts["qmmm_constraints_list"]) != 0:

                # Unit Conversion
                for ia, ja, kij0, rij0 in mm_opts["qmmm_constraints_list"]:
                    kij = kij0*kcal_mol2au/(ang2bohr*ang2bohr)
                    rij = rij0*ang2bohr
                    self._qmmm_constraints.append([ia, ja, kij, rij])

        # Unit Conversion
        self._mm_nonbond[:, 0] *= nm2bohr  # rmin_2 (nm) -> Bohr
        # eps (kcal/mol) ->Hartree
        self._mm_nonbond[:, 1] *= kJ_mol2au

        mm_sys = self._mm_prm.createSystem(nonbondedMethod=app.NoCutoff,
                                           constraints=app.HBonds,
                                           implicitSolvent=None)
        time_step = 1.0  # fs
        integrator = omm.LangevinIntegrator(300.0*unit.kelvin,
                                            1.0/unit.picoseconds,
                                            time_step*unit.femtoseconds)
        platform = omm.Platform.getPlatformByName('Reference')
        properties = {}
        if "platform" in mm_opts:
            platform = omm.Platform.getPlatformByName(mm_opts["platform"])
            if mm_opts["platform"] == 'OpenCL':
                properties = {'OpenCLPrecision': 'mixed'}
            if mm_opts["platform"] == 'CUDA':
                properties = {'CudaPrecision': 'mixed'}

        self._mm_sim = app.Simulation(self._mm_prm.topology,
                                      mm_sys,
                                      integrator,
                                      platform, properties)

    def qmmm_Coul_vdw(self, qm_crds, mm_crds, esp_chg=None):

        grds_QM = np.zeros(qm_crds.shape, dtype=np.float)
        grds_MM = np.zeros(mm_crds.shape, dtype=np.float)
        if esp_chg is None:
            esp_chg = np.zeros(qm_crds.shape[0], dtype=np.float)

        ener_qmmm = 0.0

        for iatom in range(self._qm_natom):
            ri = qm_crds[iatom]
            rmin_2_i, eps_i = self._qm_nonbond[iatom]
            q_i = esp_chg[iatom]

            for jatom in range(self._mm_natom):
                rj = mm_crds[jatom]
                rmin_2_j, eps_j = self._mm_nonbond[jatom]
                q_j = self._mm_chg[jatom]

                rmin = rmin_2_i + rmin_2_j
                eps = np.sqrt(eps_i * eps_j)

                dij = ri - rj
                rij2 = np.einsum('i,i', dij, dij)
                rij = np.sqrt(rij2)

                # Coulomb
                enqq = q_i*q_j/rij
                deqq = -enqq/rij

                # VDW
                rtmp = rmin/rij
                rtmp6 = rtmp**6
                enlj = eps*rtmp6*(rtmp6 - 2.0)
                delj = eps*rtmp6*(rtmp6-1.0)*(-12.0/rij)

                ener_qmmm += (enlj+enqq)

                gij = (delj+deqq)*dij/rij

                grds_QM[iatom] += gij
                grds_MM[jatom] -= gij

        nqm_linked = len(self._qm_linked_atoms)
        for il in range(nqm_linked):
            ia = self._qm_linked_atoms[il] - 1
            ja = self._mm_linked_atoms[il] - 1

            enr_bnd, gij = _bond_pot_grad(qm_crds[ia], mm_crds[ja])

            ener_qmmm += enr_bnd
            grds_QM[ia] += gij
            grds_MM[ja] -= gij

            # angle
            for km2 in self._mm_linked_atom2[str(ja+1)]:
                ka = km2 - 1
                enr_ang, gi, gj, gk = _angle_pot_grad(
                    qm_crds[ia], mm_crds[ja], mm_crds[ka])

                ener_qmmm += enr_ang

                grds_QM[ia] += gi
                grds_MM[ja] += gj
                grds_MM[ka] += gk

            # angle (QM(ka)-QM(ia)-MM(ja))
            ka = self._qm_neigh_atoms[il] - 1
            enr_ang, gk, gi, gj = _angle_pot_grad(
                qm_crds[ka], qm_crds[ia], mm_crds[ja])

            ener_qmmm += enr_ang

            grds_QM[ia] += gi
            grds_MM[ja] += gj
            grds_QM[ka] += gk

        return ener_qmmm, grds_QM, grds_MM

    def send(self, qm_pos, mm_pos=None):
        """ Calculate the QM or QM/MM potential energy and gradients

        Args:
            qm_pos [geomlib.Geometry]: Coordinates (in A) of the QM region.
            mm_pos [geomlib.Geometry]: Coordinates (in A) of the MM region. Defaults to None.

        Returns:
            Energy values : in Hartree
            Gradient values : in Hartree/A (not Hartree/Bohr)
        """

        ener_QM = 0.0
        ener_MM = 0.0

        ener_const = 0.0
        ener_nbond = 0.0
        ener_bnd = 0.0
        ener_ang = 0.0

        if self._l_mm:
            # A -> nm (default length unit in OpenMM is nm)
            mm_xyz = mm_pos*0.1
            # energy unit: Hartree
            # gradient unit : Hartree/Bohr
            ener_MM, grds_MM = \
                _openmm_energrads(self._mm_sim, mm_xyz)

        # xyz in Bohr
        qm_crds = qm_pos*ang2bohr
        esp_chg = np.zeros(qm_crds.shape[0], dtype=np.float)
        if self._l_qm:

            # QM
            grds_QM = np.zeros(qm_crds.shape, dtype=np.float)

            if self._l_esp:
                ener_QM, grds_QM, esp_chg = \
                    _pyscf_qm(self._qm_atnm, qm_crds,
                              self._qm_basis, self._qm_chg_tot, self._l_mp2,
                              self._l_esp, self._esp_opts)
            else:
                ener_QM, grds_QM = \
                    _pyscf_qm(self._qm_atnm, qm_crds,
                              self._qm_basis, self._qm_chg_tot, self._l_mp2)

            if self._l_mm:
                ###
                mm_crds = mm_pos*ang2bohr
                ener_qmmm, grds_qmmm_QM, grds_qmmm_MM = \
                    self.qmmm_Coul_vdw(qm_crds, mm_crds, esp_chg)
                grds_QM += grds_qmmm_QM
                grds_MM += grds_qmmm_MM

                del grds_qmmm_QM
                del grds_qmmm_MM

        elif self._l_qmpol:
            mm_crds = mm_pos*ang2bohr
            if self._l_esp:
                ener_QM, grds_QM, grds_qmmm_Coul, esp_chg = \
                    _pyscf_qmmm(self._qm_atnm, qm_crds,
                                self._qm_basis, self._qm_chg_tot,
                                mm_crds, self._mm_chg, self._l_mp2,
                                self._l_esp, self._esp_opts)
            else:
                ener_QM, grds_QM, grds_qmmm_Coul = \
                    _pyscf_qmmm(self._qm_atnm, qm_crds,
                                self._qm_basis, self._qm_chg_tot,
                                mm_crds, self._mm_chg, self._l_mp2)

            grds_MM += grds_qmmm_Coul
            del grds_qmmm_Coul

            if self._l_mm:
                # Here, we don't estimate Coulomb interaction between QM and MM regions.
                ener_qmmm, grds_qmmm_QM, grds_qmmm_MM = \
                    self.qmmm_Coul_vdw(qm_crds, mm_crds)

                grds_QM += grds_qmmm_QM
                grds_MM += grds_qmmm_MM

                del grds_qmmm_QM
                del grds_qmmm_MM

        # Constraint the distance between two QM atoms.
        ener_const = 0.0
        for ia, ja, kij0, rij0 in self._qm_constraints:
            pij = qm_crds[ia-1] - qm_crds[ja-1]
            rij = np.sqrt(np.einsum('i,i', pij, pij))
            penalty = 0.5*kij0*(rij - rij0)**2
            ener_const += penalty
            grds_QM[ia-1] += kij0*(rij-rij0)*pij/rij
            grds_QM[ja-1] -= kij0*(rij-rij0)*pij/rij

        # Constraint the distance between QM and MM atoms
        for ia, ja, kij0, rij0 in self._qmmm_constraints:
            pij = qm_crds[ia-1] - mm_crds[ja-1]
            rij = np.sqrt(np.einsum('i,i', pij, pij))
            penalty = 0.5*kij0*(rij - rij0)**2
            ener_const += penalty
            grds_QM[ia-1] += kij0*(rij-rij0)*pij/rij
            grds_MM[ja-1] -= kij0*(rij-rij0)*pij/rij

        # Since the length unit in GeomOpt is A (angstrom),
        # Length unit is converted into A
        grds_QM /= bohr2ang

        if self._l_mm:
            grds_MM /= bohr2ang

        if self._l_mm:
            return ener_QM, ener_const, grds_QM, ener_qmmm, \
                ener_MM, grds_MM, esp_chg
        else:
            return ener_QM, ener_const, grds_QM, esp_chg

    def optimize(self):

        fout_xyz = open(self._geomopt["fname_gopt_xyz"], 'w', 1)
        fout_log = open(self._geomopt["fname_gopt_log"], 'w', 1)
        qm_natom = self._qm_natom

        t0, w0 = time.clock(), time.time()

        optimizer = Berny(self._qm_geom)

        t1, w1 = t0, w0

        e_last = 0.0
        qm_grd_norm_last = 0.0

        crds_old = self._qm_geom.coords
        if self._l_mm:
            mm_crds = self._mm_geom.coords

        def step_func(x):
            if x < -0.5:
                x = -0.5
            elif x > 0.5:
                x = 0.5
            return x

        for cycle, qm_geom in enumerate(optimizer):

            # Some QM atoms are fixed
            crds_new = qm_geom.coords

            dx = crds_new - crds_old

            if self._l_mm:
                # Some MM atoms are fixed

                ener_QM, ener_const, grds_QM, ener_qmmm, ener_MM, grds_MM, esp_chg = \
                    self.send(crds_new, mm_crds)
                ener = ener_QM+ener_const+ener_qmmm+ener_MM
                qm_grd_norm = np.linalg.norm(grds_QM)
                grds_MM = np.array(
                    [[step_func(x), step_func(y), step_func(z)] for x, y, z in grds_MM])

                print(' %5d' % qm_natom, file=fout_xyz)
                print('cycle %d: E = %.12g %.8g %.4g %.6g  %.6g' %
                      (cycle+1, ener, ener_QM, ener_const,
                       ener_qmmm, ener_MM),
                      file=fout_xyz)
                print('cycle %d: E = %.12g %.8g %.4g %.6g  %.6g dE = %g  norm(grad) = %g' %
                      (cycle+1, ener, ener_QM, ener_const,
                       ener_qmmm, ener_MM, ener - e_last,
                       qm_grd_norm),
                      file=fout_log)
                for i in range(qm_natom):
                    print('%4s %10.6f %10.6f %10.6f' %
                          (self._qm_atnm[i],
                           crds_new[i, 0], crds_new[i, 1], crds_new[i, 2]),
                          file=fout_xyz)
                    print('%5d %4s %8.4f %8.4f %8.4f  %8.4f %8.4f %8.4f' %
                          ((i+1), self._qm_atnm[i],
                           dx[i, 0], dx[i, 1], dx[i, 2],
                           grds_QM[i, 0], grds_QM[i, 1], grds_QM[i, 2]),
                          file=fout_log)

            else:
                ener_QM, ener_const, grds_QM, esp_chg = \
                    self.send(crds_new)
                ener = ener_QM+ener_const
                qm_grd_norm = np.linalg.norm(grds_QM)

                print(' %5d' % qm_natom, file=fout_xyz)
                print('cycle %d: E = %.12g %.8g %.4g' %
                      (cycle+1, ener, ener_QM, ener_const),
                      file=fout_xyz)
                print('cycle %d: E = %.12g %.8g %.4g dE = %g  norm(grad) = %g' %
                      (cycle+1, ener, ener_QM, ener_const, ener - e_last,
                       qm_grd_norm),
                      file=fout_log)
                for i in range(qm_natom):
                    print('%4s %10.6f %10.6f %10.6f' %
                          (self._qm_atnm[i],
                           crds_new[i, 0], crds_new[i, 1], crds_new[i, 2]),
                          file=fout_xyz)
                    print('%5d %4s %8.4f %8.4f %8.4f  %8.4f %8.4f %8.4f' %
                          ((i+1), self._qm_atnm[i],
                           dx[i, 0], dx[i, 1], dx[i, 2],
                           grds_QM[i, 0], grds_QM[i, 1], grds_QM[i, 2]),
                          file=fout_log)

            grds_QM = np.array(
                [[step_func(x), step_func(y), step_func(z)] for x, y, z in grds_QM])

            if (cycle+1) % 5 == 0:
                qm_geom.write(self._geomopt["fname_qm_xyz"])
                if self._l_mm:
                    self._mm_geom.coords = mm_crds
                    self._mm_geom.write(self._geomopt["fname_mm_xyz"])

            dE = ener - e_last
            if abs(dE)/qm_natom < 1.0e-8:
                break

            dG = qm_grd_norm - qm_grd_norm_last
            if abs(dG)/qm_natom < 1.0e-8:
                break

            e_last = ener
            qm_grd_norm_last = qm_grd_norm

            # steepest decent method is used for the MM coordinates
            if self._l_mm:
                mm_crds -= 0.01*grds_MM
            optimizer.send((ener, grds_QM))

            crds_old = crds_new

        t1, w1 = time.clock(), time.time()
        print('geometry optimization done ',
              'CPU time %9.2f sec, wall time %9.2f sec' % ((t1-t0), (w1-w0)))
        self._qm_geom.coords = crds_old
        self._qm_geom.write('qm_optimized.xyz')
        if self._l_mm:
            self._mm_geom.coords = mm_crds
            self._mm_geom.write('mm_optimized.xyz')

        fout_log.close()
        fout_xyz.close()

    def run(self):

        if self._job in ["geomopt", "opt", "gopt"]:
            self.optimize()
        elif self._job in ["ener", "grad", "energrad"]:
            print('ENER Start')
            qm_crds = self._qm_geom.coords
            if self._l_mm:
                mm_crds = self._mm_geom.coords
                ener_QM, ener_const, grds_QM, ener_qmmm, ener_MM, grds_MM, esp_chg = \
                    self.send(qm_crds, mm_crds)
            else:
                ener_QM, ener_const, grds_QM, esp_chg = self.send(qm_crds)

            print('E(QM)[Hartree] = %.12g  E(const) = %.8g' %
                  (ener_QM, ener_const))
            print('Grads[Hartree/A]:')
            for i in range(self._qm_natom):
                print('%4s %10.6f %10.6f %10.6f' %
                      (self._qm_atnm[i],
                       grds_QM[i, 0], grds_QM[i, 1], grds_QM[i, 2]))
            if self._l_esp:
                print('(R)ESP', esp_chg)


