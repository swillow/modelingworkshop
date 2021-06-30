import numpy as np
import simtk.unit as unit
from simtk.openmm import app
import simtk.openmm as omm
# from esp import esp_atomic_charges, make_rdm1_with_orbital_response

import sys
import xml.etree.ElementTree as etree

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
_nonbond_list["C"] = {"epsilon": 0.359824*kJ_mol2au,
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



class QMMMPol (object):
    """ Polarizable QM/MM Scheme: 
    *** Charges in the MM region give an effect on the electron density of the QM region.
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

        self._job = "ener"
        if 'job' in gbl_opts:
            self._job = gbl_opts['job']

        self.geom_init(gbl_opts["geom"])
        self.qm_init(gbl_opts["qm"])

        if 'mm' in gbl_opts:
            self.mm_init(gbl_opts['mm'])

        self._geomopt = {}
        if 'geomopt' in gbl_opts:
            self._geomopt = gbl_opts['geomopt']
        ###

    def geom_init(self, geom_opts):

        #fname_bonds = 'residue_bonds.xml'
        #res_bonds = loadBondDefinitions(fname_bonds)

        qmResNames = None
        self._qm_chg = 0
        if 'qm_residues' in geom_opts:
            qmResNames = geom_opts['qm_residues']
            for res_ID in qmResNames:
                self._qm_chg += qmResNames[res_ID]
        #print('qm_chg', self._qm_chg)

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
                #print(atmId, qm2mm_idx[atmId])

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
        '''
        qmatm_linked = []

        for atmId in self._qm2mm_index:
            resId, atomName = atmId.split(':')
            if resId not in ['LIG1'] and atomName in ['CA', 'CB']:
                qmatm_linked.append(self._qmatm_index[atmId])

            if resId[:3] in res_bonds:
                for atnm1, atnm2 in res_bonds[resId[:3]]:
                    if atnm1 in ['CB']:
                        atmJd = resId + ':' + atnm2
                        if atmJd in self._qmatm_index:
                            qmatm_linked.append(self._qmatm_index[atmJd])

        self._qmatm_linked = list(set(qmatm_linked))
        # print(self._qmatm_linked)
        # sys.exit()
        '''

        # constraints

        self._qm_constraints = []
        if 'constraints' in geom_opts:
            for atmId1, atmId2, kb, r0 in geom_opts['constraints']:
                self._qm_constraints.append(
                    [atmId1, atmId2, kb*kcal_mol2au/(ang2bohr*ang2bohr), r0*ang2bohr])

    def qm_init(self, qm_opts):

        self._qm_basis = '6-31g'
        if 'basis' in qm_opts:
            self._qm_basis = qm_opts['basis']

        self._esp_opts = {}
        if 'esp_opts' in qm_opts:
            self._esp_opts = qm_opts['esp_opts']

    def mm_init(self, mm_opts):

        platformName = 'Reference'
        if 'platform' in mm_opts:
            platformName = mm_opts['platform']
        if 'fname_prmtop' in mm_opts:
            self._prt_sim, self._prt_natom, prt_chg, self._prt_nonbond = \
                _openmm_init(mm_opts["fname_prmtop"],
                                     platformName)
            self._prt_mm_chg = [prt_chg[iat] for iat in self._prt_mm_idx]
        else:
            print("MM: No prmtop file")
            sys.exit()

        # Unit Conversion
        # length: rmin_2 (nm -> Bohr)
        self._prt_nonbond[:, 0] *= nm2bohr
        # energy: eps (kJ/mol->Hartree)
        self._prt_nonbond[:, 1] *= kJ_mol2au

    def qmmm_vdw(self, qm_crds, prt_crds):

        grds_qm = np.zeros(qm_crds.shape, dtype=np.float)
        grds_prt = np.zeros(prt_crds.shape, dtype=np.float)
        #natom_qm = qm_crds.shape[0]
        #natom_prt = prt_crds.shape[0]

        enr_vdw = 0.0
        #enr_coul = 0.0

        for iat, atmId in self._qm2mm_index:
            resId, atomName = atmId.split(':')
            
            if resId not in ['LIG1'] and atomName in ['CA', 'CB']:
                continue

            #for iat in range(natom_qm):
            ri = qm_crds[iat]

            sym = self._qm_geom.species[iat]  # qm_atnm[iat].split(':')[0]

            eps_i = _nonbond_list[sym]['epsilon']
            rmin_2_i = _nonbond_list[sym]['rmin_2']

            for jat in self._prt_mm_idx:
                rj = prt_crds[jat]
                rmin_2_j, eps_j = self._prt_nonbond[jat]

                rmin = rmin_2_i + rmin_2_j
                eps = np.sqrt(eps_i * eps_j)

                dij = ri - rj
                rij2 = np.einsum('i,i', dij, dij)
                rij = np.sqrt(rij2)

                # vdW
                rtmp = rmin / rij
                rtmp6 = rtmp**6
                enlj = eps * rtmp6 * (rtmp6 - 2.0)
                delj = eps*rtmp6*(rtmp6-1.0)*(-12.0/rij)

                enr_vdw += enlj

                gij = (delj)*dij/rij

                grds_qm[iat] += gij
                grds_prt[jat] -= gij
            #print('qmmm_coul/vdw', iat, enr_coul, enr_vdw)

        return enr_vdw, grds_qm, grds_prt

    def pot_grad(self, qm_geom, prt_crds):
        """Calculate the QM/MM potential energy and gradients

        Args:
            qm_geom [geomlib.Geometry]: Coordinates (in A) of the QM region.
            prt_geom (np.array): Coordinates (in A) of the PRT (QM+MM) region. Defaults to None.

        Returns:
            Energy values : in Hartree
            Gradient values : in Hartree/A (not Hartree/Bohr)
        """

        ener_PRT = 0.0
        ener_QMMM = 0.0
        ener_QMMM_vdw = 0.0

        # A->nm (default length unit in OpenMM is nm)
        omm_xyz = prt_crds*0.1
        # energy unit : Hartree
        # gradient unit : Hartree/Bohr
        ener_PRT, grds_PRT = \
            _openmm_energrads(self._prt_sim, omm_xyz)
        del omm_xyz

        # xyz in Bohr
        qm_crds = qm_geom.coords*ang2bohr
        qm_atnm = qm_geom.species
        # (QM/MM)
        mm_crds = []
        for iat in self._prt_mm_idx:
            mm_crds.append(prt_crds[iat])
        mm_crds = np.array(mm_crds)*ang2bohr

        l_esp = False
        ener_QMMM, grds_QM, grds_MM, esp_QM = _pyscf._pyscf_qmmm(qm_atnm,
                                                                 qm_crds,
                                                                 self._qm_basis,
                                                                 self._qm_chg,
                                                                 mm_crds,
                                                                 self._prt_mm_chg,
                                                                 l_esp,
                                                                 self._esp_opts)
        for ii, iat in enumerate(self._prt_mm_idx):
            grds_PRT[iat] += grds_MM[ii]

        del grds_MM

        ener_QMMM_vdw, qm_grds, prt_grds = self.qmmm_vdw(qm_crds,
                                                         prt_crds*ang2bohr)

        grds_QM += qm_grds
        grds_PRT += prt_grds

        del qm_grds
        del prt_grds

        # constraints
        ener_const = 0.0
        for atmId1, atmId2, kij0, rij0 in self._qm_constraints:
            iat = self._qmatm_index[atmId1]
            jat = self._qmatm_index[atmId2]
            pij = qm_crds[iat] - qm_crds[jat]
            rij = np.sqrt(np.einsum('i,i', pij, pij))
            penalty = 0.5*kij0*(rij - rij0)**2
            ener_const += penalty
            grds_QM[iat] += kij0*(rij - rij0)*pij/rij
            grds_QM[jat] -= kij0*(rij - rij0)*pij/rij

        # the Length Unit in GeomOpt is A (angstrom)
        grds_QM /= bohr2ang
        grds_PRT /= bohr2ang

        return ener_QMMM, ener_const, grds_QM, esp_QM, \
            ener_QMMM_vdw, ener_PRT, grds_PRT


    def save_coordinates (self, qm_crds, prt_crds):

        self._prt_geom.coords = prt_crds

        for atmId in self._qm2mm_index:
            resId, atomName = atmId.split(':')
            if resId in ['LIG1']:
                mm_idx = self._qm2mm_index[atmId]
                qm_idx = self._qmatm_index[atmId]
                self._lig_geom.coords[mm_idx] = qm_crds[qm_idx]

        _geom.write_pdb(self._fname_protein_pdb,
                        self._geomopt['fname_prt_pdb'],
                        self._prt_geom.coords)
        _geom.write_pdb(self._fname_ligand_pdb,
                        self._geomopt['fname_lig_pdb'],
                        self._lig_geom.coords)

    def optimize(self):
        fout_xyz = open(self._geomopt['fname_gopt_xyz'], 'w', 1)
        fout_log = open(self._geomopt['fname_gopt_log'], 'w', 1)

        optimizer = Berny(self._qm_geom)
        qm_natom = len(self._qm_geom)

        def step_func(x):
            if x < -0.5:
                x = -0.5
            elif x > 0.5:
                x = 0.5
            return x

        qm_crds_old = self._qm_geom.coords
        prt_crds = self._prt_geom.coords
        ener_last = 0.0
        qm_grd_norm_last = 0.0
        ener_QM0 = 0.0
        for cycle, qm_geom in enumerate(optimizer):

            qm_crds_new = qm_geom.coords
            d_crds = qm_crds_new - qm_crds_old

            for atmId in self._qm2mm_index:
                resId, atomName = atmId.split(':')
                if resId not in ['LIG1']:
                    mm_idx = self._qm2mm_index[atmId]
                    qm_idx = self._qmatm_index[atmId]
                    if atomName not in ['CA']:
                        prt_crds[mm_idx] = qm_crds_new[qm_idx]

            ener_QMMM, ener_const, grds_QM, esp_QM, \
                ener_QMMM_vdw, ener_PRT, grds_PRT = self.pot_grad(
                    qm_geom, prt_crds)
            if cycle == 0:
                ener_QM0 = ener_QMMM
            ener_QMMM -= ener_QM0
            ener = ener_QMMM + ener_const + ener_QMMM_vdw + ener_PRT
            qm_grd_norm = np.linalg.norm(grds_QM)

            print(' %5d' % qm_natom, file=fout_xyz)
            print('cycle %3d: E = %12.6f %12.6f %10.4f %10.4f %10.4f  %12.6f' %
                  (cycle+1, ener, ener_QMMM, ener_const,
                   ener_QMMM_vdw, ener_PRT, ener_QM0),
                  file=fout_xyz)
            print('cycle %3d: E = %12.6f %12.6f %10.4f %10.4f %10.4f dE = %.6g  norm(grad) = %g' %
                  (cycle+1, ener, ener_QMMM, ener_const,
                   ener_QMMM_vdw, ener_PRT, ener - ener_last,
                   qm_grd_norm),
                  file=fout_log)

            for i in range(qm_natom):
                print('%4s %10.6f %10.6f %10.6f' %
                      (self._qm_geom.species[i],
                       qm_crds_new[i, 0], qm_crds_new[i, 1], qm_crds_new[i, 2]),
                      file=fout_xyz)
                print('%5d %4s %8.4f %8.4f %8.4f  %8.4f %8.4f %8.4f' %
                      ((i+1), self._qm_geom.species[i],
                       d_crds[i, 0], d_crds[i, 1], d_crds[i, 2],
                       grds_QM[i, 0], grds_QM[i, 1], grds_QM[i, 2]),
                      file=fout_log)

            if (cycle+1) % 2 == 0:    
                self.save_coordinates (qm_crds_new, prt_crds)

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

            
            grds_PRT = np.array(
                    [[step_func(x), step_func(y), step_func(z)]
                     for x, y, z in grds_PRT])
            prt_crds -= 0.01*grds_PRT

            grds_QM = np.array(
                [[step_func(x), step_func(y), step_func(z)]
                 for x, y, z in grds_QM])

            optimizer.send((ener, grds_QM))

        
        self.save_coordinates(qm_crds_old, prt_crds)


    def run(self):

        if self._job in ["geomopt", "opt", "gopt"]:
            self.optimize()
        elif self._job in ["ener", "grad", "energrad"]:
            print('ENER Start')
            
            prt_crds = self._prt_geom.coords
            ener_QMMM, ener_const, grds_QM, esp_QM, \
                ener_QMMM_vdw, ener_PRT, grds_PRT = \
                self.pot_grad(self._qm_geom, prt_crds)

            print('E(QM/MM)[Hartree] = %.12g  E(const) = %.8g E(MM) = %.8g' %
                  (ener_QMMM+ener_QMMM_vdw, ener_const, ener_PRT))
            print('Grads[Hartree/A]:')
            qm_natom = len(self._qm_geom)
            for i in range(qm_natom):
                print('%4s %10.6f %10.6f %10.6f' %
                      (self._qm_geom.species[i],
                       grds_QM[i, 0], grds_QM[i, 1], grds_QM[i, 2]))
            # if self._l_esp:
            #    print('(R)ESP', esp_chg)


if __name__ == "__main__":

    import getopt
    import json

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "hi:",
                               ["help=", "input="])

    if len(opts) == 0:
        print('python qmmm_pol.py -i <input json file> ')
        sys.exit(2)

    fname_json = 'input.json'
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('python qmmm_pol.py -i <input json file> ')
            sys.exit(1)
        if opt in ("-i", "--input"):
            fname_json = arg

    with open(fname_json) as f:
        data = json.load(f)

        solver = QMMMPol(data)
        solver.run()
        
