'''
Split ligand, protein from xtal structure
'''
import os
import numpy as np

excluded_residues = ['HOH']
ions_list = ['ZN', 'MN', 'NA', 'CA', 'MG', 'K', 'CL', 'FE', 'HG', 'Mn']


def separate_pdb(fname_xtal, fname_prt, fname_lig):
    lines = open(fname_xtal, 'r').readlines()

    fout = open(fname_prt, 'w')
    l_prt = True
    for line in lines:
        if line[:4] in ['ATOM', 'TER '] and l_prt:
            print(line[:-1], file=fout)
        if line[:6] in ['HETATM']:
            l_prt = False
            atnm = line[17:20].split()[0]

            if atnm in ions_list:
                print(line[:-1], file=fout)

    fout.close()

    fout = open(fname_lig, 'w')
    l_ligand = False

    for line in lines:
        if line[:4] in ['ATOM'] and l_ligand:
            print(line[:-1], file=fout)
        if line[:6] in ['HETATM']:
            l_ligand = True
            if line[17:20] in ['HOH', 'WAT']:
                continue

            print(line[:-1], file=fout)
    fout.close()


#-----------------------#
if __name__ == "__main__":
    import json
    import getopt
    import sys

    argv = sys.argv[1:]
    opts, args = getopt.getopt(
        argv, "hi:", ["help=", "input="])

    if (len(opts) == 0):
        print("python separate_complex.py -i <input_file.json>")
        sys.exit(2)

    fname_json = 'input.json'
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("python separate_complex.py -i <input_file.json>")
            sys.exit(1)
        elif opt in ("-i", "--input"):
            fname_json = arg

    with open(fname_json) as f:
        data = json.load(f)

        fname_xtal = data['fname_xtal']
        fname_prt = data['fname_protein']
        fname_lig = data['fname_ligand']

        separate_pdb(fname_xtal, fname_prt, fname_lig)
