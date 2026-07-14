import re
import copy
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import ChiralType

from localtemplate.template_utils import *

chiral_type_map = {ChiralType.CHI_UNSPECIFIED : 0, ChiralType.CHI_TETRAHEDRAL_CW: 1, ChiralType.CHI_TETRAHEDRAL_CCW: -1}
bond_type_map = {'SINGLE': '-', 'DOUBLE': '=', 'TRIPLE': '#', 'AROMATIC': '@'}
electronegativity = {
    "H": 1, "B": 3.1, "C": 2.55, "N": 3.2, "O": 3.44, "F": 3.98,
    "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16, "Se": 2.55, "Br": 2.96, "I": 2.66
}

def sort_bond_by_electneg(atom_maps, bond):
    atom_map1, atom_map2 = bond
    atom1, atom2 = atom_maps[atom_map1][1], atom_maps[atom_map2][1]
    symbol1, symbol2 = atom1.GetSymbol(), atom2.GetSymbol()
    if symbol1 == 'C' and symbol2 != 'C':
        return (atom_map1, atom_map2)
    elif symbol2 == 'C' and symbol1 != 'C':
        return (atom_map2, atom_map1)
    en1, en2 = electronegativity.get(symbol1, 1), electronegativity.get(symbol2, 1)
    if en1 < en2:
        return (atom_map1, atom_map2)
    elif en1 > en2:
        return (atom_map2, atom_map1)
    else:
#         nb_en1 = sum([electronegativity.get(n.GetSymbol(), 1) for n in atom1.GetNeighbors() if n.GetAtomMapNum() == 0])
#         nb_en2 = sum([electronegativity.get(n.GetSymbol(), 1) for n in atom2.GetNeighbors() if n.GetAtomMapNum() == 0])
        nb_en1 = sum([electronegativity.get(n.GetSymbol(), 1) for n in atom1.GetNeighbors()])
        nb_en2 = sum([electronegativity.get(n.GetSymbol(), 1) for n in atom2.GetNeighbors()])
        if nb_en1 < nb_en2:
            return (atom_map2, atom_map1)
        else:
            return (atom_map1, atom_map2)
        
def get_template_bond(temp_order, bond_smarts):
    bond_match = {}
    for n, _ in enumerate(temp_order):
        bond_match[(temp_order[n], temp_order[n-1])] = bond_smarts[n-1]
        bond_match[(temp_order[n-1], temp_order[n])] = bond_smarts[n-1]
    return bond_match   

def check_bond_break(bond1, bond2):
    if bond1 == None and bond2 != None:
        return False
    elif bond1 != None and bond2 == None:
        return True
    else:
        return False

def check_bond_formed(bond1, bond2):
    if bond1 != None and bond2 == None:
        return False
    elif bond1 == None and bond2 != None:
        return True
    else:
        return False
    
def check_bond_change(pbond, rbond):
    if pbond == None or rbond == None:
        return False
    elif bond_to_smarts(pbond) != bond_to_smarts(rbond):
        return True
    else:
        return False
    
def atom_neighbors(atom):
    neighbor = []
    for n in atom.GetNeighbors():
        neighbor.append(n.GetAtomMapNum())
    return sorted(neighbor)

def extend_changed_atoms(changed_atom_tags, reactants, max_map):
    for reactant in reactants:
        extend_idx = []
        for atom in reactant.GetAtoms():
            if str(atom.GetAtomMapNum()) in changed_atom_tags:
                for n in atom.GetNeighbors():
                    if n.GetAtomMapNum() == 0:
                        extend_idx.append(n.GetIdx())
        for idx in extend_idx:
            reactant.GetAtomWithIdx(idx).SetAtomMapNum(max_map)
    
def label_retro_edit_site(products, reactants, edit_num, use_remote=False):
    edit_num = [int(num) for num in edit_num]
    pmol = Chem.MolFromSmiles(products)
    rmol = Chem.MolFromSmiles(reactants)
    patom_maps = {atom.GetAtomMapNum(): (atom.GetIdx(), atom) for atom in pmol.GetAtoms()}
    ratom_maps = {atom.GetAtomMapNum(): (atom.GetIdx(), atom) for atom in rmol.GetAtoms()}
    used_atom = set()
    grow_atoms = []
    broken_bonds = []
    changed_bonds = []

    # cut bond
    for a in edit_num:
        for b in edit_num:
            if a >= b:
                continue
            patom_idx1, patom_idx2 = patom_maps[a][0], patom_maps[b][0]
            ratom_idx1, ratom_idx2 = ratom_maps[a][0], ratom_maps[b][0]
            pbond = pmol.GetBondBetweenAtoms(patom_idx1, patom_idx2)
            rbond = rmol.GetBondBetweenAtoms(ratom_idx1, ratom_idx2)
            if check_bond_break(pbond, rbond): # cut bond
                bond = sort_bond_by_electneg(ratom_maps, (a, b))
                broken_bonds.append(bond)
                used_atom.update([a, b])
    
    # Add LG
    for a in edit_num:
        if a in used_atom:
            continue
        patom = patom_maps[a][1]
        ratom = ratom_maps[a][1]
        if atom_neighbors(patom) != atom_neighbors(ratom):
            used_atom.update([a])
            grow_atoms.append(a)
            
    # change bond type   
    for a in edit_num:
        for b in edit_num:
            if a >= b:
                continue
            pbond = pmol.GetBondBetweenAtoms(patom_maps[a][0], patom_maps[b][0])
            rbond = rmol.GetBondBetweenAtoms(ratom_maps[a][0], ratom_maps[b][0])
            if check_bond_change(pbond, rbond):
                if a not in used_atom and b not in used_atom:
                    changed_bonds.append((a, b))
                    changed_bonds.append((b, a))
                    
    if use_remote:
        used_atoms = set(grow_atoms + [atom for bond in broken_bonds+changed_bonds for atom in bond])
        remote_atoms = [atom for atom in edit_num if atom not in used_atoms]
        remote_atoms_ = []
        for a in remote_atoms:
            atom = rmol.GetAtomWithIdx(ratom_map[a])
            neighbors_map = [n.GetAtomMapNum() for n in atom.GetNeighbors()]
            connected_neighbors = [b for b in used_atoms if b in neighbors_map]
            if len(connected_neighbors) > 0:
                pass
            else:
                for n in neighbors_map:
                    remote_atoms_.append(a)
    else:
        remote_atoms_ = None
    return grow_atoms, broken_bonds, changed_bonds, remote_atoms_

def label_foward_edit_site(reactants, products, edit_num):
    edit_num = [int(num) for num in edit_num]
    rmol = Chem.MolFromSmiles(reactants)
    pmol = Chem.MolFromSmiles(products)
    ratom_map = {atom.GetAtomMapNum():atom.GetIdx() for atom in rmol.GetAtoms()}
    patom_map = {atom.GetAtomMapNum():atom.GetIdx() for atom in pmol.GetAtoms()}
    atom_symbols = {atom.GetAtomMapNum():atom.GetSymbol() for atom in rmol.GetAtoms()}

    formed_bonds = []
    broken_bonds = []
    changed_bonds = []
    acceptors1 = set()
    acceptors2 = set()
    donors = set()
    form_bond = False
    break_bond = False
    change_bond = False
       
    # cut bond
    for a in edit_num:
        for b in edit_num:
            if a >= b:
                continue
            try:
                pbond = pmol.GetBondBetweenAtoms(patom_map[a], patom_map[b])
            except:
                pbond = None
            rbond = rmol.GetBondBetweenAtoms(ratom_map[a], ratom_map[b])
            if check_bond_break(rbond, pbond):
                if a in patom_map:
                    broken_bonds.append((a, b))
                    acceptors1.add(a)
                if b in patom_map:
                    broken_bonds.append((b, a))
                    acceptors1.add(b)
                break_bond = True
                
    # change bond
    for a in edit_num:
        for b in edit_num:
            if a >= b:
                continue
            try:
                pbond = pmol.GetBondBetweenAtoms(patom_map[a], patom_map[b])
            except:
                pbond = None
            rbond = rmol.GetBondBetweenAtoms(ratom_map[a], ratom_map[b])
            if check_bond_change(rbond, pbond):
                changed_bonds.append((a, b))
                changed_bonds.append((b, a))
                change_bond = True
                acceptors2.update([a, b])
                
    symmetric = True
    # form bond
    for a in edit_num:
        for b in edit_num:
            if a >= b:
                continue
            try:
                pbond = pmol.GetBondBetweenAtoms(patom_map[a], patom_map[b])
            except:
                pbond = None
            rbond = rmol.GetBondBetweenAtoms(ratom_map[a], ratom_map[b])
            if check_bond_formed(rbond, pbond): # cut bond
                form_bond = True
                if a not in acceptors1 and b not in acceptors1 and a not in acceptors2 and b not in acceptors2 :
                    formed_bonds.append((a, b))
                    formed_bonds.append((b, a))
                elif a in acceptors1 and b in acceptors1:
                    symmetric = False
                    formed_bonds.append((a, b))
                    formed_bonds.append((b, a))
                else:
                    symmetric = False
                    if a in acceptors1:
                        formed_bonds.append((b, a))
                    elif a in acceptors2 and b not in acceptors1:
                        formed_bonds.append((b, a))
                    if b in acceptors1:
                        formed_bonds.append((a, b))
                    elif b in acceptors2 and a not in acceptors1:
                        formed_bonds.append((a, b))

    if not symmetric:
        new_changed_bonds = []
        # electron acceptor propagation
        acceptors = set([bond[1] for bond in formed_bonds]).union(acceptors1)
        for atom in acceptors:
            for bond in changed_bonds:
                if bond[0] == atom:
                    new_changed_bonds.append(bond)
        donors = set([bond[0] for bond in formed_bonds])
        for atom in donors:
            for bond in changed_bonds:
                if bond[1] == atom:
                    new_changed_bonds.append(bond)
        changed_bonds = list(set(new_changed_bonds))
        
    used_atoms = set([atom for bond in formed_bonds+broken_bonds+changed_bonds for atom in bond])
    remote_atoms = [atom for atom in edit_num if atom not in used_atoms]
    remote_bonds = []
    for a in remote_atoms:
        atom = rmol.GetAtomWithIdx(ratom_map[a])
        neighbors_map = [n.GetAtomMapNum() for n in atom.GetNeighbors()]
        connected_neighbors = [b for b in used_atoms if b in neighbors_map]
        if len(connected_neighbors) > 0:
            pass
        else:
            for n in neighbors_map:
                remote_bonds.append((a, n))
    return formed_bonds, broken_bonds, changed_bonds, remote_bonds

def label_CHS_change(atom_map_dicts, edit_num, replacement_dict, use_stereo):
    H_dict = defaultdict(list)
    C_dict = defaultdict(list)
    S_dict = defaultdict(list)
    common_maps = set([v for vs in atom_map_dicts.values() for v in vs])
    for atom_map in edit_num:
        atom_map = int(atom_map)
        for mol, atom_map_dict in atom_map_dicts.items():
            if atom_map not in atom_map_dict:
                break
            atom = mol.GetAtomWithIdx(atom_map_dict[atom_map])
            H_dict[atom_map].append(atom.GetNumExplicitHs())
            C_dict[atom_map].append(int(atom.GetFormalCharge()))
            S_dict[atom_map].append(chiral_type_map[atom.GetChiralTag()])
            
    H_change = {replacement_dict[atom_map]: H[1]-H[0] for atom_map, H in H_dict.items() if len(H) == 2}
    C_change = {replacement_dict[atom_map]: C[1]-C[0] for atom_map, C in C_dict.items() if len(C) == 2}
    S_change = {replacement_dict[atom_map]: abs(S[1]-S[0]) for atom_map, S in S_dict.items()  if len(S) == 2}
    
    if not use_stereo:
        S_change = {atom_idx: 0 for atom_idx, S in S_change.items()}
            
    return H_change, C_change, S_change

def atommap2idx(atom_maps, idx_dict, temp_dict):
    atom_idxs = [idx_dict[atom_map] for atom_map in atom_maps]
    atom_temps = [temp_dict[atom_map] for atom_map in atom_maps]
    return (atom_idxs, atom_maps, atom_temps)

def bondmap2idx(bond_maps, idx_dict, temp_dict, sort = False, remote = False):
    bond_idxs = [(idx_dict[bond_map[0]], idx_dict[bond_map[1]]) for bond_map in bond_maps]
    if remote:
        bond_temps = list(set([(temp_dict[bond_map[0]], -1) for bond_map in bond_maps]))
        return (bond_idxs, bond_maps, bond_temps)
    else:
        bond_temps = [(temp_dict[bond_map[0]], temp_dict[bond_map[1]]) for bond_map in bond_maps]
    if not sort:
        return (bond_idxs, bond_maps, bond_temps)
    else:
        sort_bond_idxs = []
        sort_bond_maps = []
        sort_bond_temps = []
        for bond1, bond2, bond3 in zip(bond_idxs, bond_maps, bond_temps):
            if bond3[0] < bond3[1]:
                sort_bond_idxs.append(bond1)
                sort_bond_maps.append(bond2)
                sort_bond_temps.append(bond3)
            else:
                sort_bond_idxs.append(tuple(bond1[::-1]))
                sort_bond_maps.append(tuple(bond2[::-1]))
                sort_bond_temps.append(tuple(bond3[::-1]))
        return (sort_bond_idxs, sort_bond_maps, sort_bond_temps)
