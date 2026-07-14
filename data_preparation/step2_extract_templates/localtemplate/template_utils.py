from rdkit import Chem

bond_dict = {'SINGLE':'-', 'DOUBLE':'=', 'TRIPLE':'#', 'AROMATIC':':', 'DATIVE':'->'}
# bond_dict = {'SINGLE':'-', 'DOUBLE':'=', 'TRIPLE':'#', 'AROMATIC':'-'} # do not distinguish aromatic and single bond

def get_mapped_atoms(mol):
    mapped_atoms = {atom: atom.GetAtomMapNum() for atom in mol.GetAtoms()}
    return mapped_atoms
    
def clean_atommap(mol, keep_maps=[]):
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms() if atom.GetAtomMapNum() not in keep_maps]
    return mol

def clean_map_and_sort(smiles_list, product_maps=[], return_mols=False):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    [clean_atommap(mol, product_maps) for mol in mols]
    mols.sort(key=lambda m: m.GetNumAtoms(), reverse=True)
    if return_mols:
        return mols  
    else:
        return [Chem.MolToSmiles(m) for m in mols]

def sanitize_mols(mols):
    for mol in mols:
        Chem.SanitizeMol(mol)
        mol.UpdatePropertyCache()
    return

def atom_neighbors(atom):
    neighbors = [n.GetAtomMapNum() for n in atom.GetNeighbors()]
    return sorted(neighbors)

def bond_to_smarts(bond):
    '''This function takes an RDKit bond and creates a label describing
    the most important attributes'''
    begin_atom = bond.GetBeginAtom()
    end_atom = bond.GetEndAtom()
    begin_label = '%s%s' % (begin_atom.GetSymbol(), begin_atom.GetAtomMapNum())
    end_label = '%s%s' % (end_atom.GetSymbol(), end_atom.GetAtomMapNum())
    bond_smarts = bond_dict[str(bond.GetBondType())]
    bond_label = bond_smarts.join(sorted([begin_label, end_label]))
    return bond_label

def atoms_are_different(atom1, atom2): 
    '''Compares two RDKit atoms based on basic properties'''

#     if atom1.GetFormalCharge() != atom2.GetFormalCharge(): 
#         return True
#     if atom1.GetTotalNumHs() != atom2.GetTotalNumHs(): 
#         return True
    if atom1.GetNumRadicalElectrons() != atom2.GetNumRadicalElectrons():
        return True
    if atom_neighbors(atom1) != atom_neighbors(atom2): 
        return True 
    
    # change bonds
    bonds1 = sorted([bond_to_smarts(bond) for bond in atom1.GetBonds()]) 
    bonds2 = sorted([bond_to_smarts(bond) for bond in atom2.GetBonds()]) 
    if bonds1 != bonds2: 
        return True

    return False

def extend_atom_maps(reactants, product_maps, neighbor_only=False):
    max_num = max(product_maps)
    remapped_reactants = []
    if neighbor_only:
        for reactant in reactants:
            untagged_neighbors = []
            # map the unmapped neighbors  
            for atom in reactant.GetAtoms():
                if atom.GetAtomMapNum() == 0:
                    continue
                for n_atom in atom.GetNeighbors():
                    if n_atom.GetAtomMapNum() == 0:
                        untagged_neighbors.append(n_atom.GetIdx())

            for idx in untagged_neighbors:
                max_num += 1
                atom = reactant.GetAtomWithIdx(idx)
                atom.SetAtomMapNum(max_num)
            remapped_reactants.append(reactant)
    else:
        for reactant in reactants:
            for atom in reactant.GetAtoms():
                if atom.GetAtomMapNum() == 0:
                    max_num += 1
                    atom.SetAtomMapNum(max_num)
            remapped_reactants.append(reactant)
    return remapped_reactants

