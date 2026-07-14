import re
from collections import defaultdict
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import AllChem

# from template_extract_utils import *
# from template_extention import *
from localtemplate.template_utils import *
from localtemplate.template_extention import *
from localtemplate.template_canonicalize import *
from localtemplate.template_labeling import *


halides = ['Cl', 'Br', 'I']

class TemplateExtractor:
    def __init__(self, 
                 # basic settings
                 retro = False,
                 remote = False,
                 stereo = False, 
                 major_product_only = True,
                 verbose = False,
                 
                 # symbol settings
                 simplify_symbol = False,
                 simplify_aromatic = False,
                 simplify_halide = False,
                 show_charge = False,
                 show_degree = False,

                 # extension settings
                 radius = 0,
                 map_closest_LGs = False,
                 include_LGs = False,
                 map_all_atoms = False,
                 include_special_groups = None, # ['conjugated', 'functional']
                 
                 # other settings
                 
                ):
        
        """Initialize TemplateExtractor with configurable settings."""
        
        
        self.remote = remote
        self.retro = retro
        self.stereo = stereo
        self.major_product_only = major_product_only
        self.verbose = verbose
        
        self.simplify_symbol = simplify_symbol
        self.simplify_aromatic = simplify_aromatic
        self.simplify_halide = simplify_halide
        self.show_charge = show_charge
        self.show_degree = show_degree
        
        self.radius = radius
        self.map_closest_LGs = map_closest_LGs
        self.map_all_atoms = map_all_atoms
        self.include_LGs = include_LGs
        self.include_special_groups = include_special_groups
        
        

    def split_reagents(self, reaction):
        reactants, reagents, products = reaction.split('>')
        reactants = reactants.split('.')
        reagents = reagents.split('.')
        products = products.split('.')
        if len(products) == 0:  # no product
            return [], [], []
        
        splitted_reactants = []
        splitted_reagents = []
        splitted_products = []
        product_maps = []
        major_product_n = 0
        for smiles in products:
            mol = Chem.MolFromSmiles(smiles)
            mapped_atoms = get_mapped_atoms(mol)
            atom_maps = [v for v in mapped_atoms.values() if v != 0]
            if sum(atom_maps) > 0:
                if self.major_product_only and len(mapped_atoms) > major_product_n:
                    major_product_n = len(mapped_atoms)
                    product_maps = atom_maps
                    splitted_products = [smiles]
                else:
                    product_maps += atom_maps
                    splitted_products.append(smiles)
            elif smiles in reactants:
                splitted_reagents.append(smiles)

        for smiles in reactants:
            mol = Chem.MolFromSmiles(smiles)
            mol = clean_atommap(mol, product_maps)
            smiles = Chem.MolToSmiles(mol)
            mapped_atoms = get_mapped_atoms(mol)
            if smiles in splitted_reagents:
                continue
            if sum(mapped_atoms.values()) == 0:
                splitted_reagents.append(smiles)
            else:
                splitted_reactants.append(smiles)

        return splitted_reactants, reagents+splitted_reagents, splitted_products, product_maps
    
    def get_changed_atoms(self, reactants, products):
        """Determines which atoms changed in a reaction."""
        react_atoms, prod_atoms = {}, {}
        for mol in reactants:
            react_atoms.update(get_mapped_atoms(mol))
        for mol in products:
            prod_atoms.update(get_mapped_atoms(mol))
        
        # changed atoms compared to products
        changed_atoms, changed_maps = [], []
        react_atoms_inv = {v: k for k, v in react_atoms.items()}
        for p_atom, p_map in prod_atoms.items():
            r_atom = react_atoms_inv[p_map]
            if atoms_are_different(p_atom, r_atom):
                changed_atoms.append(r_atom)
                changed_maps.append(p_map)
                
        # Reactant atoms that do not appear in product (mapped leaving groups)
        for r_atom, r_map in react_atoms.items():
            if r_map not in [0]+changed_maps+list(prod_atoms.values()):
                changed_atoms.append(r_atom)
                changed_maps.append(r_map)
                
        return changed_atoms, changed_maps
    
    def get_smarts_for_atom(self, atom, show_symbol=False):
        '''
        For an RDkit atom object, generate a SMARTS pattern that
        matches the atom as strictly as possible
        '''     
        symbol = '[%s:%s]' % (atom.GetSymbol(), atom.GetAtomMapNum())
        if atom.GetIsAromatic():
            if self.simplify_aromatic:
                symbol = '[a:%s]' % atom.GetAtomMapNum()
            else:
                symbol = symbol.lower()

        # Explicit formal charge
        if self.show_charge:
            charge = atom.GetFormalCharge()
            charge_symbol = '+' if (charge >= 0) else '-'
            charge_symbol += '{}'.format(abs(charge))
            if ':' in symbol: 
                symbol = symbol.replace(':', '{}:'.format(charge_symbol))
            else:
                symbol = symbol.replace(']', '{}]'.format(charge_symbol))
                    
        # Explicit degree
        if self.show_degree:
            if ':' in symbol:
                symbol = symbol.replace(':', ';D{}:'.format(atom.GetDegree()))
            else:
                symbol = symbol.replace(']', ';D{}]'.format(atom.GetDegree()))
    
        
            
        if show_symbol:
            return symbol
        
        if self.simplify_symbol:
            symbol = '[A:%s]' % atom.GetAtomMapNum()
                
        elif atom.GetSymbol() in halides and self.simplify_halide :
            symbol = '[Br:%s]' % atom.GetAtomMapNum() # use Br to represent all halides

        return symbol
    
    def expand_atoms_by_neighbors(self, mol, atoms_to_use, symbol_replacements):
        '''Given an RDKit molecule and a list of AtomIdX which should be included
        in the reaction, this function expands the list of AtomIdXs to include one 
        nearest neighbor with special consideration of (a) unimportant neighbors and
        (b) important functional groupings'''

        # Copy
        expanded_atoms = atoms_to_use[:]
        
        # Look for all atoms in the current list of atoms to use
        for atom in mol.GetAtoms():
            if atom.GetIdx() not in atoms_to_use:
                continue
            for n_atom in atom.GetNeighbors():
                atom_idx = n_atom.GetIdx()
                if atom_idx in expanded_atoms:
                    continue
                expanded_atoms.append(atom_idx)
                symbol = self.get_smarts_for_atom(n_atom, show_symbol=True)
                symbol_replacements.append((atom_idx, symbol))
        return expanded_atoms, symbol_replacements
        
    def expand_atoms_by_group(self, mol, atoms_to_use, groups, symbol_replacements):
        '''Given an RDKit molecule and a list of AtomIdX which should be included
        in the reaction, this function expands the list of AtomIdXs to include one 
        nearest neighbor with special consideration of (a) unimportant neighbors and
        (b) important functional groupings'''

        # Copy
        expanded_atoms = atoms_to_use[:]
        special_num = 200
        special_mappings = {}
        
        # Look for all atoms in the current list of atoms to use
        for atom in mol.GetAtoms():
            if atom.GetIdx() not in atoms_to_use:
                continue
            # Ensure membership of changed atom is checked against group
            for group in groups:
                if int(atom.GetIdx()) in group[0]:
                    for idx in group[1]:
                        if idx not in expanded_atoms:
                            n_atom = mol.GetAtomWithIdx(idx)
                            if n_atom.GetAtomMapNum() == 0:
                                special_num += 1
                                n_atom.SetAtomMapNum(special_num)
                                special_mappings[idx] = special_num
                            expanded_atoms.append(idx)
                            symbol = self.get_smarts_for_atom(n_atom, show_symbol=True)
        return expanded_atoms, symbol_replacements, mol

    def get_fragments_for_changed_atoms(self, mols, changed_maps, category = 'reactant', special_atoms = []):
        fragments = ''
        mols_changed = []
        mols_special = []
        for mol in mols:
            symbol_replacements = []
            atoms_to_use = []
            mapped_atoms = get_mapped_atoms(mol)
            for atom, atom_map in mapped_atoms.items():
                if atom_map in changed_maps+special_atoms:
                    atoms_to_use.append(atom.GetIdx())
                    symbol = self.get_smarts_for_atom(atom)
                    symbol_replacements.append((atom.GetIdx(), symbol))
                elif self.include_LGs and atom_map == 0:
                    atoms_to_use.append(atom.GetIdx())

            for k in range(self.radius):
                atoms_to_use, symbol_replacements = self.expand_atoms_by_neighbors(mol, atoms_to_use, 
                    symbol_replacements=symbol_replacements)
            
            if category == 'reactant':
                if self.include_special_groups and self.include_special_groups:
                    groups = get_special_groups(mol, self.include_special_groups)
                    atoms_to_use, symbol_replacements, mol = self.expand_atoms_by_group(mol, atoms_to_use, 
                        groups=groups, symbol_replacements=symbol_replacements)
                    for atom_idx in atoms_to_use:
                        atom = mol.GetAtomWithIdx(atom_idx)
                        atom_map = atom.GetAtomMapNum()
                        if atom_map not in changed_maps:
                            special_atoms.append(atom_map)
                            symbol = self.get_smarts_for_atom(atom)
                            symbol_replacements.append((atom.GetIdx(), symbol))

            else:
                groups = []
            
            mols_special.append(mol)
            # Define new symbols based on symbol_replacements
            symbols = [atom.GetSmarts() for atom in mol.GetAtoms()]
            for (i, symbol) in symbol_replacements:
                symbols[i] = symbol
            if not atoms_to_use: 
                continue
            
            mol_copy = deepcopy(mol)
            clean_atommap(mol_copy)
            this_fragment = AllChem.MolFragmentToSmiles(mol_copy, atoms_to_use, 
                atomSymbols=symbols, allHsExplicit=True, 
                isomericSmiles=self.stereo)
            fragments += '(' + this_fragment + ').'
            mols_changed.append(Chem.MolToSmiles(clean_atommap(Chem.MolFromSmiles(Chem.MolToSmiles(mol, True))), True))

        return fragments[:-1], mols_special, special_atoms

    def canonicalize_transform(self, react_fragments, product_fragments):
        '''This function takes an atom-mapped SMARTS transform and
        converts it to a canonical form by, if nececssary, rearranging
        the order of reactant and product templates and reassigning
        atom maps.'''
        
        raw_template = '%s>>%s' % (product_fragments, react_fragments)
        raw_template = raw_template.split('>>')[0][1:-1].replace(').(', '.') + \
                      '>>' + raw_template.split('>>')[1][1:-1].replace(').(', '.')
        template_reordered = canonicalize_atom_mapped_reaction(raw_template)
        if self.retro:
            template_reordered = '>>'.join(template_reordered.split('>>')[::-1])
            
        if self.simplify_symbol:
            template_reordered = template_reordered.replace('*', 'A')
            
        canonical_template, replacement_dict = reassign_atom_mapping(template_reordered)
        
        return canonical_template, replacement_dict

    def match_label(self, reactant_smi, product_smi, replacement_dict, changed_maps):
        if self.retro:
            smiles1, smiles2 = product_smi, reactant_smi
        else:
            smiles1, smiles2 = reactant_smi, product_smi 
        mol1, mol2 = Chem.MolFromSmiles(smiles1), Chem.MolFromSmiles(smiles2)
        
        replacement_dict = {int(k): int(v) for k, v in replacement_dict.items()}
        atom_map_dict = {mol: {atom.GetAtomMapNum():atom.GetIdx() for atom in mol.GetAtoms()} for mol in [mol1, mol2]}
        
        H_change, Charge_change, Chiral_change = label_CHS_change(atom_map_dict, changed_maps, replacement_dict, self.stereo)

        if self.retro:
            ALG_atoms, broken_bonds, changed_bonds, remote_atoms = label_retro_edit_site(smiles1, smiles2, changed_maps)
            edits = {'A': atommap2idx(ALG_atoms, atom_map_dict[mol1], replacement_dict), # add leaving group
                     'B': bondmap2idx(broken_bonds, atom_map_dict[mol1], replacement_dict, True), # bread bond
                     'C': bondmap2idx(changed_bonds, atom_map_dict[mol1], replacement_dict)} # change bond
            if len(edits['B'][0]) != 0:
                edits['A'] = ([], [], [])
                edits['C'] = ([], [], [])
            if self.remote:
                edits['R'] = atommap2idx(remote_atoms, atom_map_dict[mol1], replacement_dict) # remote
        else:
            formed_bonds, broken_bonds, changed_bonds, remote_bonds = label_foward_edit_site(smiles1, smiles2, changed_maps)
            edits = {'A': bondmap2idx(formed_bonds, atom_map_dict[mol1], replacement_dict), # attack
                     'B': bondmap2idx(broken_bonds, atom_map_dict[mol1], replacement_dict), # break bond
                     'C': bondmap2idx(changed_bonds, atom_map_dict[mol1], replacement_dict)} # change bond
            if self.remote:
                edits['R'] = bondmap2idx(remote_bonds, atom_map_dict[mol1], replacement_dict, False, True) # remote
        return edits, H_change, Charge_change, Chiral_change


    def __call__(self, reaction, return_template_only=False, return_changed_atom_only=False):
        """Extracts a reaction template based on the defined settings."""
        reactants_list, reagents_list, products_list, product_maps = self.split_reagents(reaction)
        if not products_list:
            return None
        
        products = clean_map_and_sort(products_list, product_maps, return_mols=True)
        reactants = clean_map_and_sort(reactants_list, product_maps, return_mols=True)
        
        if self.map_closest_LGs:
            reactants = extend_atom_maps(reactants, product_maps, neighbor_only=True)
        if self.map_all_atoms:
            reactants = extend_atom_maps(reactants, product_maps, neighbor_only=False)
            
        try:
            sanitize_mols(reactants + products)
        except Exception as e:
            if self.verbose:
                print(e, 'Could not sanitize molecules.')
            return None
        
        changed_atoms, changed_maps = self.get_changed_atoms(reactants, products)
        if return_changed_atom_only:
            return changed_maps
        if self.verbose:
            print ('Changed atoms:', changed_maps)
            
        if not changed_maps:
            if self.verbose:
                print ('No changes in the reaction.')
            return None
        
#         special_atoms = []
        react_fragments, reactants, special_atoms = self.get_fragments_for_changed_atoms(reactants, changed_maps, 'reactant', [])
        product_fragments, _, _ = self.get_fragments_for_changed_atoms(products, changed_maps, 'product', special_atoms)
        
        if self.verbose:
            print ('Reagetant fragments:', react_fragments)
            print ('Product fragments:', product_fragments)
        canonical_template, replacement_dict = self.canonicalize_transform(react_fragments, product_fragments)

        rxn = AllChem.ReactionFromSmarts(canonical_template)
        if rxn.Validate()[1] != 0 and self.verbose: 
            print('Could not validate reaction successfully')
            print('canonical_template:', canonical_template)
            return 
        
        if return_template_only:
            return canonical_template
        else:
            reactants_smiles = '.'.join([Chem.MolToSmiles(r) for r in reactants])
            products_smiles = '.'.join([Chem.MolToSmiles(p) for p in products])
            edits, H_change, C_change, S_change = self.match_label(reactants_smiles, products_smiles, replacement_dict, changed_maps)    
            results = {
                'reaction_template': canonical_template,
                'reaction': reactants_smiles + '>>' + products_smiles,
                'necessary_reagent': clean_map_and_sort(reagents_list),
                'replacement_dict': replacement_dict,
                'change_maps': changed_maps,
                'edits': edits,
                'Hydrogen_change': H_change,
                'Charge_change': C_change,
                'Chirality_change': S_change
                }
            return results
