import getopt
import json
import sys
import qmmm_solver


if __name__ == "__main__":

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

        solver = qmmm_solver.QMMMSolver(data)

        solver.run()
