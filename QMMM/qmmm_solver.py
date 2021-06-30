import numpy as np
import simtk.unit as unit
from simtk.openmm import app
import simtk.openmm as omm
# from esp import esp_atomic_charges, make_rdm1_with_orbital_response

import sys
import time
# from pyscf import gto, scf, mp, qmmm, grad, lib
from pyscf.data.nist import BOHR, HARTREE2J, AVOGADRO
from simtk.unit import constants
from berny import Berny, geomlib

import _geom
import _pyscf

ang2bohr = 1.0/BOHR
nm2bohr = 10.0/BOHR
bohr2nm = BOHR/10.0
bohr2ang = BOHR

au2kcal_mol = HARTREE2J*AVOGADRO/4184.0
au2kJ_mol = HARTREE2J*AVOGADRO/1000.0
kcal_mol2au = 1.0/au2kcal_mol
kJ_mol2au = 1.0/au2kJ_mol


_nonbond_list = {}
_const_2_6 = 2.0**(1.0/6.0)
_nonbond_list["C"] = {"epsilon": 0.359824 * kcal_mol2au,
                      "rmin_2": 0.34*nm2bohr*_const_2_6}
_nonbond_list["H"] = {"epsilon": 0.0656888*kJ_mol2au,
                      "rmin_2": 0.107*nm2bohr*_const_2_6}
_nonbond_list["N"] = {"epsilon": 0.71128*kJ_mol2au,
                      "rmin_2": 0.325*nm2bohr*_const_2_6}
_nonbond_list["O"] = {"epsilon": 0.87864*kJ_mol2au,
                      "rmin_2": 0.296*nm2bohr*_const_2_6}
_nonbond_list["S"] = {"epsilon": 1.046*kJ_mol2au,
                      "rmin_2": 0.35636*nm2bohr*_const_2_6}


def _openmm_init(fname_prmtop, platformName=None):

    prmtop = app.AmberPrmtopFile(fname_prmtop)
    system = prmtop.createSystem(nonbondedMethod=app.NoCutoff,
                                 constraints=app.HBonds,
                                 implicitSolvent=None)
    time_step = 1.0  # fs
    integrator = omm.LangevinIntegrator(300.0*unit.kelvin,
                                        1.0/unit.picoseconds,
                                        time_step*unit.femtoseconds)

    platform = omm.Platform.getPlatformByName('Reference')
    properties = {}
    if platformName == 'OpenCL':
        platform = omm.Platform.getPlatformByName('OpenCL')
        properties = {'OpenCLPrecision': 'mixed'}
    if platformName == 'CUDA':
        platform = omm.Platform.getPlatformByName('CUDA')
        properties = {'CudaPrecision': 'mixed'}

    simulation = app.Simulation(prmtop.topology,
                                system,
                                integrator,
                                platform, properties)

    charge_list = prmtop._prmtop.getCharges()
    natom = prmtop._prmtop.getNumAtoms()
    nonbond = np.array(prmtop._prmtop.getNonbondTerms())

    return simulation, natom, charge_list, nonbond


def _openmm_energrads(simulation, mm_xyz):
    """Estimate the potential energy and gradient using OpenMM

    Args:
        simulation (app.Simulation): 
        mm_xyz (np.array[natom, 3]): coordinates in the length unit of nm
    """

    simulation.context.setPositions(mm_xyz)
    state = simulation.context.getState(
        getEnergy=True, getForces=True)
    ener_MM = state.getPotentialEnergy().value_in_unit(
        unit.kilocalorie_per_mole)*kcal_mol2au
    frc_MM = state.getForces(asNumpy=True).value_in_unit(
        unit.kilocalorie_per_mole/unit.angstrom)*(kcal_mol2au/ang2bohr)

    return ener_MM, -frc_MM


class QMMMSolver(object):
    """ QM and QM/MM Scheme:
    Here, the interactions between QM and MM regions are estimated via AMBER force fields.

    QM: Ligand
    QM_Residues: Residues in the protein are involved in the chemical reactions
    MM_Residues: MM atoms except QM_residues
    We have MM Force Field (AMBER) for the protein, but not for Ligand.
    Hence, we have no MM force field between ligand and MM.
    This code is developed using the following equation:
    E_sys = E_MM(MM_Residues)
          + E_QM(QM_Residue+QM_Lig)
          + E_MM([QM_Lig+QM_Lig]--MM_Residue) // Add a missing interaction between QM_Lig and MM_Residues

    """

    def __init__(self, gbl_opts):

        theory = "qm"
        if "theory" in gbl_opts:
            theory = gbl_opts["theory"]

        self._l_mm = False

        if theory == "qmmm":
            self._l_mm = True
        elif theory == "mm":
            print('Not MM support')
            sys.exit(5)

        self.geom_init(gbl_opts['geom'])
        self.qm_init(gbl_opts["qm"])

        self._job = "ener"
        if "job" in gbl_opts:
            self._job = gbl_opts["job"]

        if self._l_mm:
            self.mm_init(gbl_opts["mm"])

        self._geomopt = None
        if self._job in ["geomopt", "opt", "gopt"]:
            self._geomopt = gbl_opts["geomopt"]

    def geom_init(self, geom_opts):
        qmResNames = None
        self._qm_chg = 0
        if 'qm_residues' in geom_opts:
            qmResNames = geom_opts['qm_residues']
            for res_ID in qmResNames:
                self._qm_chg += qmResNames[res_ID]

        self._qm2mm_index = {}
        self._lig_geom = None
        self._fname_ligand_pdb = None

        if 'ligand' in geom_opts:
            self._fname_ligand_pdb = geom_opts['ligand']
            self._lig_geom, qm2mm_idx = _geom._load_geom(
                geom_opts['ligand'], qmResNames, l_protein=False)
            for atmId in qm2mm_idx:
                self._qm2mm_index[atmId] = qm2mm_idx[atmId]
        else:
            print('No Lig Geom')
            sys.exit()

        self._prt_geom = None
        self._fname_protein_pdb = None
        self._prt_mm_idx = []
        if self._l_mm:

            if 'protein' in geom_opts:
                self._fname_protein_pdb = geom_opts['protein']
                self._prt_geom, qm2mm_idx = _geom._load_geom(
                    geom_opts['protein'], qmResNames, l_protein=True)
                prt_qm_idx = []
                for atmId in qm2mm_idx:
                    resID, atomName = atmId.split(':')
                    if atomName not in ['C', 'N', 'HA']:
                        self._qm2mm_index[atmId] = qm2mm_idx[atmId]
                    prt_qm_idx.append(qm2mm_idx[atmId])
                # print(atmId, qm2mm_idx[atmId])

                prt_natom = len(self._prt_geom.coords)
                self._prt_mm_idx = [iat for iat in range(
                    prt_natom) if iat not in prt_qm_idx]
            else:
                print('No Protein Geom')
                sys.exit()

        self._qmatm_index = {}
        for qm_iat, atmId in enumerate(self._qm2mm_index):
            self._qmatm_index[atmId] = qm_iat

        self._qm_geom = \
            _geom.get_qm_geom(self._lig_geom,
                              self._qm2mm_index,
                              self._qmatm_index,
                              self._prt_geom)

        self._qm_constraints = []
        if 'constraints' in geom_opts:
            for atmId1, atmId2, kb, r0 in geom_opts['constraints']:
                self._qm_constraints.append(
                    [atmId1, atmId2, kb*kcal_mol2au/(ang2bohr*ang2bohr), r0*ang2bohr])

    def qm_init(self, qm_opts):

        self._l_mp2 = False
        if "method" in qm_opts:
            if qm_opts["method"] in ['mp2', 'MP2']:
                self._l_mp2 = True

        self._qm_basis = "sto-3g"
        if "basis" in qm_opts:
            self._qm_basis = qm_opts["basis"]

        self._l_esp = False
        self._esp_opts = {}
        if "esp" in qm_opts:
            self._l_esp = qm_opts["esp"]
            self._esp_opts = qm_opts["esp_opts"]

    def mm_init(self, mm_opts):

        platformName = 'Reference'
        if 'platform' in mm_opts:
            platformName = mm_opts['platform']

        if "fname_prmtop" in mm_opts:
            self._prt_sim, self._prt_natom, prt_chg, self._prt_nonbond = \
                _openmm_init(mm_opts['fname_prmtop'], platformName)
            self._prt_mm_chg = [prt_chg[iat] for iat in self._prt_mm_idx]
        else:
            print("MM: No prmtop file")
            sys.exit()

        # rmin_2 (nm -> Bohr)
        self._prt_nonbond[:, 0] *= nm2bohr
        # energy: eps (kJ/mol -> Hartree)
        self._prt_nonbond[:, 1] *= kJ_mol2au

    def qmmm_Coul_vdw(self, qm_crds, prt_crds, esp_chg=None):

        grds_qm = np.zeros(qm_crds.shape, dtype=np.float)
        grds_prt = np.zeros(prt_crds.shape, dtype=np.float)
        #natom_qm = qm_crds.shape[0]

        if esp_chg is None:
            esp_chg = np.zeros(qm_crds.shape[0], dtype=np.float)

        ener_qmmm = 0.0
        enr_vdw = 0.0
        enr_coul = 0.0

        #for iatom in range(natom_qm):
        for iat, atmId in self._qm2mm_index:
            resId, atomName = atmId.split(':')
            
            if resId not in ['LIG1'] and atomName in ['CA', 'CB']:
                continue

            ri = qm_crds[iatom]
            sym = self._qm_geom.species[iatom]
            eps_i = _nonbond_list[sym]['epsilon']
            rmin_2_i = _nonbond_list[sym]['rmin_2']

            q_i = esp_chg[iatom]

            for jmm, jatom in enumerate(self._prt_mm_idx):
                rj = prt_crds[jatom]
                rmin_2_j, eps_j = self._prt_nonbond[jatom]
                q_j = self._prt_mm_chg[jmm]

                rmin = rmin_2_i + rmin_2_j
                eps = np.sqrt(eps_i * eps_j)

                dij = ri - rj
                rij2 = np.einsum('i,i', dij, dij)
                rij = np.sqrt(rij2)

                # Coulomb
                enqq = q_i*q_j/rij
                deqq = -enqq/rij
                enr_coul += enqq

                # VDW
                rtmp = rmin/rij
                rtmp6 = rtmp**6
                enlj = eps*rtmp6*(rtmp6 - 2.0)
                delj = eps*rtmp6*(rtmp6-1.0)*(-12.0/rij)

                enr_vdw += enlj

                gij = (delj+deqq)*dij/rij

                grds_qm[iatom] += gij
                grds_prt[jatom] -= gij

        ener_qmmm = enr_vdw + enr_coul

        return ener_qmmm, grds_qm, grds_prt

    def pot_grad(self, qm_geom, prt_crds=None):
        """ Calculate the QM or QM/MM potential energy and gradients

        Args:
            qm_geom [geomlib.Geometry]: Coordinates (in A) of the QM region.
            prt_crds [geomlib.Geometry]: Coordinates (in A) of the PRT (QM+MM) region. Defaults to None.

        Returns:
            Energy values : in Hartree
            Gradient values : in Hartree/A (not Hartree/Bohr)
        """

        ener_QM = 0.0
        ener_PRT = 0.0
        ener_qmmm = 0.0  # interaction between QM and MM
        ener_const = 0.0

        grds_PRT = None

        if self._l_mm:
            # A -> nm (default length unit in OpenMM is nm)
            omm_xyz = prt_crds*0.1
            # energy unit: Hartree
            # gradient unit : Hartree/Bohr
            ener_PRT, grds_PRT = \
                _openmm_energrads(self._prt_sim, omm_xyz)
            del omm_xyz

        # xyz in Bohr
        qm_crds = qm_geom.coords*ang2bohr
        qm_atnm = qm_geom.species

        esp_chg = np.zeros(qm_crds.shape[0], dtype=np.float)

        # QM
        grds_QM = np.zeros(qm_crds.shape, dtype=np.float)

        ener_QM, grds_QM, esp_chg = \
            _pyscf._pyscf_qm(qm_atnm, qm_crds,
                             self._qm_basis, self._qm_chg, self._l_mp2,
                             self._l_esp, self._esp_opts)

        if self._l_mm:
            ###
            ener_qmmm, grds_qmmm_QM, grds_qmmm_PRT = \
                self.qmmm_Coul_vdw(qm_crds, prt_crds*ang2bohr, esp_chg)
            grds_QM += grds_qmmm_QM
            grds_PRT += grds_qmmm_PRT

            del grds_qmmm_QM
            del grds_qmmm_PRT

        # Constraint the distance between two QM atoms.
        ener_const = 0.0
        for atmId1, atmId2, kij0, rij0 in self._qm_constraints:
            iat = self._qmatm_index[atmId1]
            jat = self._qmatm_index[atmId2]
            pij = qm_crds[iat-1] - qm_crds[jat-1]
            rij = np.sqrt(np.einsum('i,i', pij, pij))
            penalty = 0.5*kij0*(rij - rij0)**2
            ener_const += penalty
            grds_QM[iat-1] += kij0*(rij-rij0)*pij/rij
            grds_QM[jat-1] -= kij0*(rij-rij0)*pij/rij

        # Since the length unit in GeomOpt is A (angstrom),
        # Length unit is converted into A
        grds_QM /= bohr2ang

        if self._l_mm:
            grds_PRT /= bohr2ang

        return ener_QM, ener_const, grds_QM, esp_chg, \
            ener_qmmm, ener_PRT, grds_PRT

    def save_coordinates(self, qm_crds, prt_crds):

        if self._l_mm:
            self._prt_geom.coords = prt_crds
            _geom.write_pdb(self._fname_protein_pdb,
                            self._geomopt['fname_prt_pdb'],
                            self._prt_geom.coords)

        for atmId in self._qm2mm_index:
            resId, atomName = atmId.split(':')
            if resId in ['LIG1']:
                mm_idx = self._qm2mm_index[atmId]
                qm_idx = self._qmatm_index[atmId]
                self._lig_geom.coords[mm_idx] = qm_crds[qm_idx]

        _geom.write_pdb(self._fname_ligand_pdb,
                        self._geomopt['fname_lig_pdb'],
                        self._lig_geom.coords)

    def optimize(self):

        fout_xyz = open(self._geomopt["fname_gopt_xyz"], 'w', 1)
        fout_log = open(self._geomopt["fname_gopt_log"], 'w', 1)
        qm_natom = len(self._qm_geom)
        t0, w0 = time.clock(), time.time()

        optimizer = Berny(self._qm_geom)

        def step_func(x):
            if x < -0.5:
                x = -0.5
            elif x > 0.5:
                x = 0.5
            return x

        t1, w1 = t0, w0

        qm_crds_old = self._qm_geom.coords
        if self._l_mm:
            prt_crds = self._prt_geom.coords
        ener_last = 0.0
        qm_grd_norm_last = 0.0
        ener_QM0 = 0.0

        for cycle, qm_geom in enumerate(optimizer):

            # Some QM atoms are fixed
            qm_crds_new = qm_geom.coords

            dx = qm_crds_new - qm_crds_old

            for atmId in self._qm2mm_index:
                resId, atomName = atmId.split(':')
                if resId not in ['LIG1']:
                    qm_idx = self._qmatm_index[atmId]
                    mm_idx = self._qm2mm_index[atmId]
                    if atomName not in ['CA']:
                        prt_crds[mm_idx] = qm_crds_new[qm_idx]

            ener_QM, ener_const, grds_QM, esp_chg, ener_qmmm, ener_PRT, grds_PRT = \
                self.pot_grad(qm_geom, prt_crds)
            if cycle == 0:
                ener_QM0 = ener_QM

            ener = ener_QM+ener_const+ener_qmmm+ener_PRT
            qm_grd_norm = np.linalg.norm(grds_QM)

            print(' %5d' % qm_natom, file=fout_xyz)
            print('cycle %3d: E = %12.6f %12.6f %10.4f %10.4f %10.4f  %12.6f' %
                  (cycle+1, ener, ener_QM, ener_const,
                   ener_qmmm, ener_PRT, ener_QM0),
                  file=fout_xyz)
            print('cycle %3d: E = %12.6f %12.6f %10.4f %10.4f %10.4f dE = %.6g  norm(grad) = %g' %
                  (cycle+1, ener, ener_QM, ener_const,
                   ener_qmmm, ener_PRT, ener - ener_last,
                   qm_grd_norm),
                  file=fout_log)

            for i in range(qm_natom):
                print('%4s %10.6f %10.6f %10.6f' %
                      (self._qm_geom.species[i],
                       qm_crds_new[i, 0], qm_crds_new[i, 1], qm_crds_new[i, 2]),
                      file=fout_xyz)
                print('%5d %4s %8.4f %8.4f %8.4f  %8.4f %8.4f %8.4f' %
                      ((i+1), self._qm_geom.species[i],
                       dx[i, 0], dx[i, 1], dx[i, 2],
                       grds_QM[i, 0], grds_QM[i, 1], grds_QM[i, 2]),
                      file=fout_log)

            if (cycle + 1) % 2 == 0:
                self.save_coordinates(qm_crds_new, prt_crds)

            dE = ener - ener_last
            if abs(dE)/qm_natom < 1.0e-8:
                break

            dG = qm_grd_norm - qm_grd_norm_last
            if abs(dG)/qm_natom < 1.0e-8:
                break

            ener_last = ener
            qm_grd_norm_last = qm_grd_norm
            qm_crds_old = qm_crds_new

            for atmId in self._qm2mm_index:
                resId, atomName = atmId.split(':')
                if resId[:3] not in ['LIG'] and atomName not in ['CA']:
                    '''
                    'CA' belongs to MM
                    Except 'CA', bond, angle, and torsional gradients, which are estimated within MM, are copied to QM particles.
                    '''
                    mm_idx = self._qm2mm_index[atmId]
                    qm_idx = self._qmatm_index[atmId]
                    grds_QM[qm_idx] += grds_PRT[mm_idx]
                    grds_PRT[mm_idx, 0] = 0
                    grds_PRT[mm_idx, 1] = 0
                    grds_PRT[mm_idx, 2] = 0

            # steepest decent method is used for the MM coordinates
            if self._l_mm:
                grds_PRT = np.array(
                    [[step_func(x), step_func(y), step_func(z)]
                     for x, y, z in grds_PRT])
                prt_crds -= 0.01*grds_PRT

            optimizer.send((ener, grds_QM))

        self.save_coordinates(qm_crds_old, prt_crds)

        t1, w1 = time.clock(), time.time()
        print('geometry optimization done ',
              'CPU time %9.2f sec, wall time %9.2f sec' % ((t1-t0), (w1-w0)))

        fout_log.close()
        fout_xyz.close()

    def run(self):

        if self._job in ["geomopt", "opt", "gopt"]:
            self.optimize()
        elif self._job in ["ener", "grad", "energrad"]:
            print('ENER Start')

            prt_crds = None
            if self._l_mm:
                prt_crds = self._prt_geom.coords
            ener_QM, ener_const, grds_QM, esp_chg, ener_qmmm, ener_PRT, grds_PRT = \
                self.pot_grad(self._qm_geom, prt_crds)

            print('E(QM)[Hartree] = %.12g  E(const) = %.8g' %
                  (ener_QM, ener_const))
            print('Grads[Hartree/A]:')
            qm_natom = len(self._qm_geom)
            for i in range(qm_natom):
                print('%4s %10.6f %10.6f %10.6f' %
                      (self._qm_geom.species[i],
                       grds_QM[i, 0], grds_QM[i, 1], grds_QM[i, 2]))
            if self._l_esp:
                print('(R)ESP', esp_chg)


if __name__ == "__main__":

    import getopt
    import json

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "hi:",
                               ["help=", "input="])

    if len(opts) == 0:
        print('python run_qmmm.py -i <input json file> ')
        sys.exit(2)

    fname_json = 'input.json'
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('python run_pyscf.py -i <input json file> ')
            sys.exit(1)
        if opt in ("-i", "--input"):
            fname_json = arg

    with open(fname_json) as f:
        data = json.load(f)

        # print(data["mm"])
        # sys.exit()
        solver = QMMMSolver(data)

        solver.run()
