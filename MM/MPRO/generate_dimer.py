import numpy as np
import sys
import MDAnalysis as mda


def generate_dimer(fname_pqr, fname_mon1, fname_mon2, R, T):
    #prt = mda.Universe('mpro_6lu7.pdb')
    prt = mda.Universe(fname_pqr)

    prt.atoms.write(fname_mon1)

    prt.atoms.rotate(R)
    prt.atoms.translate(T)
    prt.atoms.write(fname_mon2)


if __name__ == "__main__":
    import json
    import getopt

    argv = sys.argv[1:]

    opts, args = getopt.getopt(
        argv, "hi:", ["help=", "input="])

    if (len(opts) == 0):
        print("python generate_dimer.py -i <input_file.json>")
        sys.exit(2)

    fname_json = 'input.json'
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("python generate_dimer.py -i <input_file.json>")
            sys.exit(1)
        elif opt in ("-i", "--input"):
            fname_json = arg

    with open(fname_json) as f:
        data = json.load(f)

        fname_pqr = data["fname_monomer_pqr"]
        fname_mon1 = data["fname_monomer1_pdb"]
        fname_mon2 = data["fname_monomer2_pdb"]
        R = np.array(data["R"])
        T = np.array(data["T"])

        generate_dimer(fname_pqr, fname_mon1, fname_mon2, R, T)
