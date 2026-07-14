# canonicalize reaction template

import re
from rdkit import Chem
from rdkit.Chem import AllChem

def fragment_scorer(smarts):
    LG_scores = {
        ']-F': 1, 'F-[': 1,
        ']=O': 2, 'O=[': 2,
        ']-Cl': 3, 'Cl-[': 3,
        ']-O': 4, 'O-[': 4,
        ']-Br': 5, 'Br-[': 5,
        ']-O-S(-C)(=O)=O': 6,
        ']-O-S(=O)(=O)-C(-F)(-F)-F': 7,
        ']-I': 8, 'I-[': 8,
    }
    LG_score = 0
    for lg, score in LG_scores.items():
        if lg in smarts:
            LG_score += score
    if LG_score != 0:
        return LG_score*5
    
    atom_scores = {
            'N': 0.01,
            'P': 0.015,
            'o': 0.016,
            'O': 0.02, 
            'c': 0.025,
            'C': 0.03,
         }
    atoms = re.findall(r'\[([A-Za-z])(:\d+)?\]', smarts)
    atom_score = 0
    for atom, _ in atoms:
        atom_score += atom_scores.get(atom, 0)
    return len(atoms) + atom_score

def atoms_counter(smarts):
    counts = 0
    is_letter = False
    for token in smarts:
        if token.isalpha():
            if not is_letter:  # This condition ensures we count the atom when a new one starts
                counts += 1
                is_letter = True
        else:
            is_letter = False
    return counts

def remove_Hs(template):
    # Match atoms like [CH3:1], [cH:2], [nH:3], [OH:4], etc.
    pattern = r'\[([A-Za-z]{1,2})[H\d]*:(\d+)\]'

    def repl(match):
        atom = match.group(1)  # e.g., CH, cH, c, etc.
        index = match.group(2)
        # Remove explicit H (and trailing digits) only when preceded by another letter
        # e.g. CH3->C, cH->c, nH->n, OH->O  but NOT Hg->g or Hs->s
        atom_clean = re.sub(r'(?<=[a-zA-Z])H\d*', '', atom)
        if not atom_clean:  # atom is pure H (e.g. [H:69]) — stripping would give invalid [:69]
            return match.group(0)
        return f'[{atom_clean}:{index}]'

    cleaned_template = re.sub(pattern, repl, template)
    return cleaned_template

def canonicalize_atom_mapped_reaction(reaction_smarts, remove_map = False):
    """
    Canonicalizes an atom-mapped reaction SMARTS string.
    
    The approach:
    1. Split the reaction into reactants, agents, and products
    2. Parse each component into molecules
    3. Extract atom mapping information
    4. Generate canonical SMILES for each molecule while preserving mappings
    5. Reorder components based on canonical representation
    6. Reconstruct the canonical reaction SMARTS
    
    Args:
        reaction_smarts (str): Atom-mapped reaction SMARTS
        
    Returns:
        str: Canonicalized atom-mapped reaction SMARTS
    """
    # Split reaction into components (use negative lookbehind to avoid splitting on '->' dative bonds)
    components = re.split(r'(?<!-)>', reaction_smarts)

    if len(components) != 3:
        raise ValueError("Invalid reaction SMARTS format. Expected 'reactants>agents>products'")
    
    reactants_str, agents_str, products_str = components
    
    # Process reactants
    reactants = reactants_str.split(".") if reactants_str else []
    agents = agents_str.split(".") if agents_str else []
    products = products_str.split(".") if products_str else []
    
    # Function to canonicalize a single molecule while preserving atom maps
    def canonicalize_mol_with_maps(smarts):
        if not smarts:
            return ""
        
        # Parse the molecule
        mol = Chem.MolFromSmarts(smarts)
        if not mol:
            return smarts  # Return original if parsing fails
        
        # Extract atom maps
        atom_maps = {}
        for atom in mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num:
                atom_maps[atom.GetIdx()] = map_num
        
        # Clear atom maps to generate canonical SMILES
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        
        # Get canonical SMILES (without atom maps)
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, allBondsExplicit=True)
        if remove_map:
            return canonical_smiles
        # Recreate molecule from canonical SMILES
        canon_mol = Chem.MolFromSmiles(canonical_smiles, sanitize=False)
        if not canon_mol:
            return smarts  # Return original if recreation fails
        
        # Create mapping from original to canonical atom indices
        match = mol.GetSubstructMatch(canon_mol)
        if not match:  # If matching fails, try the reverse
            match = canon_mol.GetSubstructMatch(mol)
            if not match or len(match) != mol.GetNumAtoms():
                return smarts  # Return original if matching fails
        
        # Restore atom maps to canonical molecule
        for new_idx, old_idx in enumerate(match):
#         for old_idx, new_idx in enumerate(match):
            if old_idx in atom_maps:
                canon_mol.GetAtomWithIdx(new_idx).SetAtomMapNum(atom_maps[old_idx])
        
        # Convert back to SMARTS with atom maps
        smiles = Chem.MolToSmiles(canon_mol, canonical=False, allBondsExplicit=True)
        return remove_Hs(smiles)

    # Canonicalize each molecule in each component
    canon_reactants = sorted([canonicalize_mol_with_maps(r) for r in reactants if r], key=lambda x: fragment_scorer(x))
    canon_agents = sorted([canonicalize_mol_with_maps(a) for a in agents if a], key=lambda x: fragment_scorer(x))
    canon_products = sorted([canonicalize_mol_with_maps(p) for p in products if p], key=lambda x: fragment_scorer(x))

    # Reconstruct the canonical reaction SMARTS
    canon_reactants_str = ".".join(canon_reactants)
    canon_agents_str = ".".join(canon_agents)
    canon_products_str = ".".join(canon_products)
    
    return f"{canon_reactants_str}>{canon_agents_str}>{canon_products_str}"

def permutations(template):
    mols = [Chem.MolFromSmarts(smarts) for smarts in template.split('>>')]
    if any(m is None for m in mols):
        raise ValueError(f'Could not parse template SMARTS: {template[:120]}')
    n_atoms = sum(m.GetNumAtoms() for m in mols)
    labels = re.findall(r'(\[\d*[a-zA-Z@][^\]]*:\d+\])', template)
    if len(labels) == 1 or '(' in template or n_atoms > len(labels): # include leaving group
        return [labels]
    charges = re.findall('\;(.+?[0-9]+)\:', template)
    bonds = re.findall('\]([-=#:])\[', template)
    if ''.join(bonds) != ''.join(bonds[::-1]) or ''.join(charges) != ''.join(charges[::-1]):
        return [labels]
    return [labels,  labels[::-1]]
        
def enumerate_mapping(transform):
    for i, templates in enumerate(transform.split('>>')):
        grow_template = None
        for template in templates.split('.'):
            pert_template = permutations(template)
            if grow_template == None:
                grow_template = pert_template
            else:
                growed_template = []
                for t in grow_template:
                    for p in pert_template:
                        growed_template.append(t+p)
                grow_template = growed_template
        if i == 0:
            r_permutes = grow_template
        else:
            p_permutes = grow_template
    t_permutes = []
    for r in r_permutes:
        for p in p_permutes:
            t_permutes.append(r+p)
    return t_permutes

def reassign_atom_mapping(transform):
    p_labels = enumerate_mapping(transform)
    templates = set()
    templates_sort = {}
    replacement_dicts = {}
    for all_labels in p_labels:
    # Define list of replacements which matches all_labels *IN ORDER*
        replacements = []
        replacement_dict_symbol = {}
        replacement_dict = {}
        counter = 1
        for label in all_labels: # keep in order! this is important
            atom_map = label.split(':')[1].split(']')[0]
            if atom_map not in replacement_dict:
                replacement_dict_symbol[label] = '%s:%s]' % (label.split(':')[0], counter)
                replacement_dict[atom_map] = str(counter)
                counter += 1
            else:
                replacement_dict_symbol[label] = '%s:%s]' % (label.split(':')[0], replacement_dict[atom_map])
            replacements.append(replacement_dict_symbol[label])
            
        # Perform replacements in order
        transform_newmaps = re.sub(r'\[\d*[a-zA-Z@][^\]]*:\d+\]', lambda match: (replacements.pop(0)), transform)
        transform_newmaps = '>>'.join(transform_newmaps.split('>>')[::-1])

        templates.add(transform_newmaps)
        templates_sort[transform_newmaps] = ''.join(re.findall(r'\[\d*[a-zA-Z@][^\]]*:\d+\]', transform_newmaps))
        replacement_dicts[transform_newmaps] = replacement_dict
    
    transform_newmaps = sorted(list(templates), key = lambda t:templates_sort[t])[0]
    replacement_dict = replacement_dicts[transform_newmaps]
    return transform_newmaps, replacement_dict


def canonicalize_template(template):
    '''This function takes one-half of a template SMARTS string 
    (i.e., reactants or products) and re-orders them based on
    an equivalent string without atom mapping.'''

    # Strip labels to get sort orders
    template_nolabels = re.sub('\:[0-9]+\]', ']', template)

    # Split into separate molecules *WITHOUT wrapper parentheses*
    template_nolabels_mols = template_nolabels[1:-1].split(').(')
    template_mols          = template[1:-1].split(').(')

    # Split into fragments within those molecules
    for i in range(len(template_mols)):
        nolabel_mol_frags = template_nolabels_mols[i].split('.')
        mol_frags         = template_mols[i].split('.')

        # Get sort order within molecule, defined WITHOUT labels
        sortorder = [j[0] for j in sorted(enumerate(nolabel_mol_frags), key = lambda x:x[1])]

        # Apply sorting and merge list back into overall mol fragment
        template_nolabels_mols[i] = '.'.join([nolabel_mol_frags[j] for j in sortorder])
        template_mols[i]          = '.'.join([mol_frags[j] for j in sortorder])

    # Get sort order between molecules, defined WITHOUT labels
    sortorder = [j[0] for j in sorted(enumerate(template_nolabels_mols), key = lambda x:x[1])]

    # Apply sorting and merge list back into overall transform
    template = '(' + ').('.join([template_mols[i] for i in sortorder]) + ')'

    return template