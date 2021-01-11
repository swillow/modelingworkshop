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
au2kJ_mol = HARTREE2J*AVOGADRO
kcal_mol2au = 1.0/au2kcal_mol
kJ_mol2au = 1.0/au2kJ_mol


_nonbond_list = {}
_nonbond_list["C"] = {"epsilon": 0.359824*kJ_mol2au, "sigma": 0.34*nm2bohr}
_nonbond_list["H"] = {"epsilon": 0.0656888*kJ_mol2au, "sigma": 0.107*nm2bohr}
_nonbond_list["N"] = {"epsilon": 0.71128*kJ_mol2au, "sigma": 0.325*nm2bohr}
_nonbond_list["O"] = {"epsilon": 0.87864*kJ_mol2au, "sigma": 0.296*nm2bohr}
_nonbond_list["S"] = {"epsilon": 1.046*kJ_mol2au, "sigma": 0.35636*nm2bohr}


def _gradient_on_mm_particles(mol, mm_mol, dm):
    # The interaction between QM atoms and MM particles
    # \sum_K d/dR (1/|r_K-R|) = \sum_K (r_K-R)/|r_K-R|^3
    #dm = mf_qmmm.make_rdm1()
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
        with mf_qmmm.mol.with_rinv_origin(coords[i]):
            v = mf_qmmm.mol.intor('int1e_iprinv')
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
            words = line[30:].split()
            coords.append([float(x) for x in words[0:3]])
            species.append(words[-1])

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
                mm_crds, mm_chg, l_mp2):

    atm_list = []
    for ia, xyz in enumerate(qm_crds):
        atm_list.append([atnm_list[ia], (xyz[0], xyz[1], xyz[2])])

    qm_mol = _pyscf_mol_build(atm_list, qm_basis, qm_chg_tot)
    mf = qmmm.mm_charge(scf.HF(qm_mol), mm_crds, mm_chg).run()

    if l_mp2:
        postmf = mp.MP2(mf).run()
        ener_QM = postmf.e_tot
        grds_QM = postmf.Gradients().kernel()
        dm = make_rdm1_with_orbital_response(postmf)
        grds_MM = _gradient_on_mm_particles(mf.mol, mf.mm_mol, dm)
    else:
        ener_QM = mf.e_tot
        grds_QM = mf.Gradients().kernel()
        dm = mf.make_rdm1()
        grds_MM = _gradient_on_mm_particles(mf.mol, mf.mm_mol, dm)

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
        elif theory == "qm-pol":
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

        self._qm_fixed_atoms = []
        if "fixed_atom_list" in qm_opts:
            self._qm_fixed_atoms = qm_opts["fixed_atom_list"]

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

        self._mm_fixed_atoms = []
        if "fixed_atoms" in mm_opts:
            self._mm_fixed_atoms = mm_opts["fixed_atoms"]

        self._mm_prm = None
        if "fname_prmtop" in mm_opts:
            self._mm_prm = app.AmberPrmtopFile(mm_opts["fname_prmtop"])
        else:
            print("MM: No prmtop file")
            sys.exit()

        self._mm_chg = self._mm_prm._prmtop.getCharges()

        self._mm_natom = self._mm_prm._prmtop.getNumAtoms()

        self._mm_nonbond = np.array(self._mm_prm._prmtop.getNonbondTerms())

        # Unit Conversion
        self._mm_nonbond[0, :] *= nm2bohr  # rmin_2 (nm) -> Bohr
        # eps (kcal/mol) ->Hartree
        self._mm_nonbond[1, :] *= kJ_mol2au

        mm_sys = self._mm_prm.createSystem(nonbondedMethod=app.NoCutoff,
                                           constraints=app.HBonds,
                                           implicitSolvent=None)
        time_step = 1.0  # fs
        integrator = omm.LangevinIntegrator(300.0*unit.kelvin,
                                            1.0/unit.picoseconds,
                                            time_step*unit.femtoseconds)
        platform = omm.Platform.getPlatformByName('Reference')
        self._mm_sim = app.Simulation(self._mm_prm.topology,
                                      mm_sys,
                                      integrator,
                                      platform)

    def send(self, qm_pos, mm_pos=None):

        ener_QM = 0.0
        ener_MM = 0.0

        ener_const = 0.0
        ener_nbond = 0.0

        if self._l_mm:
            # A -> nm (default length unit in OpenMM is nm)
            mm_xyz = mm_pos*0.1
            # energy unit: Hartree
            # gradient unit : Hartree/Bohr
            ener_MM, grds_MM = \
                _openmm_energrads(self._mm_sim, mm_xyz)

        # xyz in Bohr
        qm_crds = qm_pos*ang2bohr
        if self._l_qm:

            # QM
            esp_chg = np.zeros(qm_crds.shape[0], dtype=np.float)
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
                mm_crds = mm_pos*ang2bohr
                ener_nbond = 0.0

                for iatom in range(self._qm_natom):
                    ri = qm_crds[iatom]

                    rmin_2_i, eps_i = self._qm_nonbond[iatom]
                    q_i = esp_chg[iatom]

                    for jatom in range(self._mm_natom):
                        rj = mm_crds[jatom]
                        q_j = self._mm_chg[jatom]

                        rmin_2_j, eps_j = self._mm_nonbond[jatom]

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

                        ener_nbond += (enlj+enqq)

                        gij = (delj+deqq)*dij/rij

                        grds_QM[iatom] += gij
                        grds_MM[iatom] -= gij

        elif self._l_qmpol:
            mm_crds = mm_pos*ang2bohr
            ener_QM, grds_QM, grds_QMM = \
                _pyscf_qmmm(self._qm_atnm, qm_crds,
                            self._qm_basis, self._qm_chg_tot,
                            mm_crds, self._mm_chg, self._l_mp2)

            grds_MM += grds_QMM

            ener_nbond = 0.0

            for iatom in range(self._qm_natom):
                ri = qm_crds[iatom]
                rmin_2_i, eps_i = self._qm_nonbond[iatom]

                for jatom in range(self._mm_natom):
                    rj = mm_crds[jatom]
                    rmin_2_j, eps_j = self._mm_nonbond[jatom]

                    rmin = rmin_2_i + rmin_2_j
                    eps = np.sqrt(eps_i * eps_j)

                    dij = ri - rj

                    rij2 = np.einsum('i,i', dij, dij)
                    rij = np.sqrt(rij2)

                    # VDW
                    rtmp = rmin/rij
                    rtmp6 = rtmp**6
                    enlj = eps*rtmp6*(rtmp6 - 2.0)
                    delj = eps*rtmp6*(rtmp6-1.0)*(-12.0/rij)

                    ener_nbond += enlj

                    gij = delj*dij/rij

                    grds_QM[iatom] += gij
                    grds_MM[iatom] -= gij

        # Constraint the distance between two QM atoms.
        ener_const = 0.0
        for ia, ja, kij0, rij0 in self._qm_constraints:
            pij = qm_crds[ia-1] - qm_crds[ja-1]
            rij = np.sqrt(np.einsum('i,i', pij, pij))
            penalty = 0.5*kij0*(rij - rij0)**2
            ener_const += penalty
            grds_QM[ia-1] += kij0*(rij-rij0)*pij/rij
            grds_QM[ja-1] -= kij0*(rij-rij0)*pij/rij

        # Gradients of fixed atoms are zero
        for atm_id in self._qm_fixed_atoms:
            grds_QM[atm_id-1] = [0.0, 0.0, 0.0]

        # Since the length unit in GeomOpt is A (angstrom),
        # Length unit is converted into A
        grds_QM /= bohr2ang

        if self._l_mm:
            for atm_id in self._mm_fixed_atoms:
                grds_MM[atm_id-1] = [0.0, 0.0, 0.0]

            grds_MM /= bohr2ang

        if self._l_mm:
            return ener_QM, ener_const, grds_QM, ener_nbond, ener_MM, grds_MM, esp_chg
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
        crds_old = self._qm_geom.coords
        if self._l_mm:
            mm_crds = self._mm_geom.coords

        def step_func(x):
            if x < -0.1:
                x = -0.1
            elif x > 0.1:
                x = 0.1
            return x

        for cycle, qm_geom in enumerate(optimizer):

            # Some QM atoms are fixed
            crds_new = qm_geom.coords
            for atm_id in self._qm_fixed_atoms:
                crds_new[atm_id-1] = crds_old[atm_id-1]

            dx = crds_new - crds_old

            if self._l_mm:
                # Some MM atoms are fixed
                for atm_id in self._mm_fixed_atoms:
                    mm_crds[atm_id-1] = self._mm_geom.coords[atm_id-1]

                ener_QM, ener_const, grds_QM, ener_nbond, ener_MM, grds_MM, esp_chg = \
                    self.send(crds_new, mm_crds)
                ener = ener_QM+ener_const+ener_nbond+ener_MM

                grds_MM = np.array(
                    [[step_func(x), step_func(y), step_func(z)] for x, y, z in grds_MM])

                print(' %5d' % qm_natom, file=fout_xyz)
                print('cycle %d: E = %.12g %.8g %.4g %.6g  %.6g' %
                      (cycle+1, ener, ener_QM, ener_const,
                       ener_nbond, ener_MM),
                      file=fout_xyz)
                print('cycle %d: E = %.12g %.8g %.4g %.6g  %.6g dE = %g  norm(grad) = %g' %
                      (cycle+1, ener, ener_QM, ener_const,
                       ener_nbond, ener_MM, ener - e_last,
                       np.linalg.norm(grds_QM)),
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

                print(' %5d' % qm_natom, file=fout_xyz)
                print('cycle %d: E = %.12g %.8g %.4g' %
                      (cycle+1, ener, ener_QM, ener_const),
                      file=fout_xyz)
                print('cycle %d: E = %.12g %.8g %.4g dE = %g  norm(grad) = %g' %
                      (cycle+1, ener, ener_QM, ener_const, ener - e_last,
                       np.linalg.norm(grds_QM)),
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

            e_last = ener

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
            qm_crds = self._qm_geom.coords
            if self._l_mm:
                mm_crds = self._mm_geom.coords
                ener_QM, ener_const, grds_QM, ener_nbond, ener_MM, grds_MM, esp_chg = \
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
