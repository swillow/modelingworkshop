from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
import numpy as np


def write_pdb(fname_pdb, fname_pdb_new, crds, box=None):
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
        elif len(line) == 0:
            continue
        elif len(line) > 0 and line[0] == ' ':
            continue
        else:
            print(line, file=fout)
    fout.close()


def get_positions_from_pdb(fname_pdb):
    nameMembrane = ['DPP', 'POP']

    f_pdb = open(fname_pdb)
    l_pdb = f_pdb.read().split('\n')
    f_pdb.close()

    coords = []
    prt_heavy_atoms = []
    mem_heavy_atoms = []
    iatom = 0
    for line in l_pdb[:-1]:
        if line[:6] in ['ATOM  ', 'HETATM']:
            words = line[30:].split()
            x = float(words[0])
            y = float(words[1])
            z = float(words[2])

            coords.append(Vec3(x, y, z))

            if line[17:20] in nameMembrane and words[-1] != 'H':
                mem_heavy_atoms.append(iatom)
            elif line[:6] in ['ATOM  '] and words[-1] != 'H':
                prt_heavy_atoms.append(iatom)
            
            iatom += 1

    return np.array(coords), prt_heavy_atoms, mem_heavy_atoms


def run_min(options):

    prmtop = AmberPrmtopFile(options['prmtop'])
    crds_A, prt_heavy_atoms, mem_heavy_atoms = get_positions_from_pdb(
        options['pdb'])
    # inpcrd = AmberInpcrdFile('step5.inpcrd')

    system = prmtop.createSystem(nonbondedMethod=PME,
                                 nonbondedCutoff=1.2*nanometer,
                                 constraints=HBonds,
                                 rigidWater=True,
                                 ewaldErrorTolerance=0.0005)
    # Restraints
    # fc_pos = 1000.0  # kJ/mol/nm^2
    if 'fc_pos' in options:
        fc_pos = options["fc_pos"]  # kJ/mol/nm^2
        prt_rest = CustomExternalForce('fc_pos*periodicdistance(x,y,z,x0,y0,z0)^2')

        prt_rest.addGlobalParameter('fc_pos', fc_pos)
        prt_rest.addPerParticleParameter('x0')
        prt_rest.addPerParticleParameter('y0')
        prt_rest.addPerParticleParameter('z0')

        for iatom in prt_heavy_atoms:
            x, y, z = 0.1*crds_A[iatom]
            prt_rest.addParticle(iatom, [x, y, z])
        system.addForce(prt_rest)

        mem_rest = CustomExternalForce('fc_pos*periodicdistance(x,y,z,x,y,z0)^2')
        
        mem_rest.addGlobalParameter('fc_pos', fc_pos)
        mem_rest.addPerParticleParameter('z0')
        for iatom in mem_heavy_atoms:
            x, y, z = 0.1*crds_A[iatom]
            mem_rest.addParticle(iatom, [z])
        system.addForce(mem_rest)

    integrator = LangevinIntegrator(
        300*kelvin, 1.0/picosecond, 1.0*femtoseconds)

    if "Platform" in options:
        if options["Platform"] == 'OpenCL':
            platform = Platform.getPlatformByName('OpenCL')
            properties = {'OpenCLPrecision': 'mixed'}
            simulation = Simulation(prmtop.topology, system,
                                    integrator, platform, properties)
        elif options["Platform"] == 'CUDA':
            platform = Platform.getPlatformByName('CUDA')
            properties = {'CudaPrecision': 'mixed'}
            simulation = Simulation(prmtop.topology, system,
                                    integrator, platform, properties)
        else:
            simulation = Simulation(prmtop.topology, system, integrator)
    else:  # CPU
        simulation = Simulation(prmtop.topology, system, integrator)
    simulation.context.setPositions(crds_A*0.1)  # A->nm

    print("\nInitial System Energy")
    print(simulation.context.getState(
        getEnergy=True).getPotentialEnergy().value_in_unit(kilocalorie_per_mole), " kcal/mol")

    mini_nstep = 5000
    mini_Tol = 100.0
    if "maxIteraction" in options:
        mini_nstep = options['maxIteraction']
    if "tolerance" in options:
        mini_Tol = options['tolerance']

    print("\n Energy minimization : %s steps" % mini_nstep)
    simulation.minimizeEnergy(tolerance=mini_Tol * kilojoule_per_mole,
                              maxIterations=mini_nstep)

    state = simulation.context.getState(getEnergy=True,
                                        getPositions=True,
                                        enforcePeriodicBox=True)
    ener = state.getPotentialEnergy().value_in_unit(kilocalorie_per_mole)
    print('Energy ', ener, " kcal/mol")
    crds_A = state.getPositions(asNumpy=True).value_in_unit(angstrom)
    write_pdb(options['pdb'], options['save_pdb'], crds_A)


def run_nvt(options):

    prmtop = AmberPrmtopFile(options['prmtop'])
    crds_A, prt_heavy_atoms, mem_heavy_atoms = get_positions_from_pdb(
        options['pdb'])

    system = prmtop.createSystem(nonbondedMethod=PME,
                                 nonbondedCutoff=1.2*nanometer,
                                 constraints=HBonds,
                                 rigidWater=True,
                                 ewaldErrorTolerance=0.0005)

    if "fc_pos" in options:
        # Restraints
        fc_pos = options['fc_pos']  # kJ/mol/nm^2
        prt_rest = CustomExternalForce('fc_pos*periodicdistance(x,y,z,x0,y0,z0)^2')
        
        prt_rest.addGlobalParameter('fc_pos', fc_pos)
        prt_rest.addPerParticleParameter('x0')
        prt_rest.addPerParticleParameter('y0')
        prt_rest.addPerParticleParameter('z0')

        for iatom in prt_heavy_atoms:
            x, y, z = crds_A[iatom]/10
            prt_rest.addParticle(iatom, [x, y, z])
        system.addForce(prt_rest)

        mem_rest = CustomExternalForce('fc_pos*periodicdistance(x,y,z,x,y,z0)^2')
        
        mem_rest.addGlobalParameter('fc_pos', fc_pos)
        mem_rest.addPerParticleParameter('z0')
        for iatom in mem_heavy_atoms:
            x, y, z = crds_A[iatom]/10
            mem_rest.addParticle(iatom, [z])
        system.addForce(mem_rest)

    dt = 1.0
    if "dt_fs" in options:
        dt = options['dt_fs']

    temp = 300.0
    if 'Temperature' in options:
        temp = options['Temperature']

    integrator = LangevinIntegrator(
        temp*kelvin, 1.0/picosecond, dt*femtoseconds)

    if "Platform" in options:
        if options["Platform"] == 'OpenCL':
            platform = Platform.getPlatformByName('OpenCL')
            properties = {'OpenCLPrecision': 'mixed'}
            simulation = Simulation(prmtop.topology, system,
                                    integrator, platform, properties)
        elif options["Platform"] == 'CUDA':
            platform = Platform.getPlatformByName('CUDA')
            properties = {'CudaPrecision': 'mixed'}
            simulation = Simulation(prmtop.topology, system,
                                    integrator, platform, properties)
        else:
            simulation = Simulation(prmtop.topology, system, integrator)
    else:  # CPU
        simulation = Simulation(prmtop.topology, system, integrator)

    simulation.context.setPositions(crds_A*0.1)  # A->nm

    simulation.context.setVelocitiesToTemperature(temp)

    nstep = 125000
    nstdout = 1000
    nstdcd = 5000
    if 'totalSteps' in options:
        nstep = options['totalSteps']
    if 'nstdout' in options:
        nstdout = options['nstdout']
    if 'ndcd' in options:
        nstdcd = options['ndcd']

    simulation.reporters.append(app.StateDataReporter(options['stdout'],
                                                      nstdout,
                                                      step=True, time=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      progress=False,
                                                      remainingTime=True,
                                                      speed=False,
                                                      volume=False,
                                                      totalSteps=nstep,
                                                      separator='     '))
    if 'dcd' in options:
        simulation.reporters.append(
            app.DCDReporter(options['dcd'], nstdcd))

    simulation.step(nstep)

    state = simulation.context.getState(getPositions=True,
                                        enforcePeriodicBox=True)
    crds_A = state.getPositions(asNumpy=True).value_in_unit(angstrom)
    write_pdb(options['pdb'], options['save_pdb'], crds_A)

    if "save_state" in options:
        simulation.saveState(options['save_state'])


def run_npt(options):

    prmtop = AmberPrmtopFile(options['prmtop'])
    crds_A, prt_heavy_atoms, mem_heavy_atoms = get_positions_from_pdb(
        options['pdb'])

    system = prmtop.createSystem(nonbondedMethod=PME,
                                 nonbondedCutoff=1.2*nanometer,
                                 constraints=HBonds,
                                 rigidWater=True,
                                 ewaldErrorTolerance=0.0005)

    press = 1.0
    if "Pressure" in options:
        press = options["Pressure"]

    temp = 300.0
    if 'Temperature' in options:
        temp = options['Temperature']

    if options['Barostat'] == "MonteCarloMembraneBarostat":
        system.addForce(MonteCarloMembraneBarostat(press*bar, 300*bar*nanometer, temp*kelvin,
                                                   MonteCarloMembraneBarostat.XYIsotropic, MonteCarloMembraneBarostat.ZFree, 100))
    else:
        system.addForce(MonteCarloBarostat(press*bar, temp*kelvin, 100))

    dt = 1.0
    if "dt_fs" in options:
        dt = options['dt_fs']

    integrator = LangevinIntegrator(
        temp*kelvin, 1.0/picosecond, dt*femtoseconds)

    if "Platform" in options:
        if options["Platform"] == 'OpenCL':
            platform = Platform.getPlatformByName('OpenCL')
            properties = {'OpenCLPrecision': 'mixed'}
            simulation = Simulation(prmtop.topology, system,
                                    integrator, platform, properties)
        elif options["Platform"] == 'CUDA':
            platform = Platform.getPlatformByName('CUDA')
            properties = {'CudaPrecision': 'mixed'}
            simulation = Simulation(prmtop.topology, system,
                                    integrator, platform, properties)
        else:
            simulation = Simulation(prmtop.topology, system, integrator)
    else:  # CPU
        simulation = Simulation(prmtop.topology, system, integrator)

    simulation.context.setPositions(crds_A*0.1)  # A->nm

    if 'load_state' in options:
        simulation.loadState(options['load_state'])
    else:
        simulation.context.setVelocitiesToTemperature(temp)

    nstep = 125000
    nstdout = 1000
    nstdcd = 5000
    ncycle = 40
    if 'ncycle' in options:
        ncycle = options['ncycle']
    if 'nstep_cycle' in options:
        nstep = options['nstep_cycle']
    if 'nstdout' in options:
        nstdout = options['nstdout']
    if 'ndcd' in options:
        nstdcd = options['ndcd']

    simulation.reporters.append(app.StateDataReporter(options['stdout'],
                                                      nstdout,
                                                      step=True, time=True,
                                                      potentialEnergy=True,
                                                      temperature=True,
                                                      progress=False,
                                                      remainingTime=True,
                                                      speed=False,
                                                      volume=True,
                                                      totalSteps=nstep*ncycle,
                                                      separator='     '))
    if 'dcd' in options:
        simulation.reporters.append(
            app.DCDReporter(options['dcd'], nstdcd))

    for i in range(ncycle):
        simulation.step(nstep)

        state = simulation.context.getState(getPositions=True,
                                            enforcePeriodicBox=True)
        crds_A = state.getPositions(asNumpy=True).value_in_unit(angstrom)
        box_A = state.getPeriodicBoxVectors(
            asNumpy=True).value_in_unit(angstrom)

        if 'save_pdb' in options:
            fname = options['save_pdb'].split('.')[0]
            fname_pdb_out = fname + '_' + str(i+1) + '.pdb'
            write_pdb(options['pdb'], fname_pdb_out, crds_A, box_A)
        if 'save_state' in options:
            fname = options['save_state'].split('.')[0]
            fname_rst_out = fname + '_' + str(i+1) + '.rst'
            simulation.saveState(fname_rst_out)


if __name__ == "__main__":
    import getopt
    import json

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "hi:",
                               ["help=", "input="])

    if len(opts) == 0:
        print('python AmberOpenMM.py -i <input json file> ')
        sys.exit(2)

    fname_json = 'input.json'
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('python AmberOpenMM.py -i <input json file> ')
            sys.exit(1)
        if opt in ("-i", "--ifile"):
            fname_json = arg

    with open(fname_json) as f:
        data = json.load(f)
        job = data['job']

        if job == 'min':
            run_min(data[job])
        elif job == 'nvt':
            run_nvt(data[job])
        elif job == 'npt':
            run_npt(data[job])
