#!/usr/bin/env python3

import os
import json
import csv

root_dir = os.environ.get("PISTACHIO_DIR", "")
out_path = "Raw_data_actiondetails_LIST.csv"

def norm_str(x):
    return (x or "").replace("\r\n", " ").replace("\n", " ").replace("\t", " ").strip()

def is_list(x): return isinstance(x, list)
def is_dict(x): return isinstance(x, dict)

def extract_temp_and_time(parameters):
    temp_obj, time_val = "", ""
    if not is_list(parameters): return temp_obj, time_val
    for p in parameters:
        if not is_dict(p): continue
        ptype = (p.get("type", "") or "").lower()
        if ptype in ("temperature", "temp"):
            v  = p.get("value", None)
            tx = norm_str(p.get("text", ""))
            temp_obj = {"value": v, "text": ""} if v is not None else ({"value": "", "text": tx} if tx else "")
        elif ptype == "time":
            if p.get("value", None) is not None:
                time_val = p["value"]
    return temp_obj, time_val

def extract_components(components):
    out = []
    if not is_list(components): return out
    for comp in components:
        if not is_dict(comp): continue
        out.append({
            "role": comp.get("role", ""),
            "name": norm_str(comp.get("name", "")),
            "smiles": comp.get("smiles", ""),
            "quantities": comp.get("quantities", []) or []
        })
    return out

def extract_yield_for_action(act_type, components):
    if (act_type or "").lower() != "yield": return ""
    if not is_list(components): return ""
    for comp in components:
        if not is_dict(comp): continue
        for q in comp.get("quantities", []) or []:
            if (q.get("type","") or "").lower() == "yield" and q.get("value", None) is not None:
                return q["value"]
    return ""

fieldnames = [
    "row_id",
    "Patent Title","Date","Quality","Reaction Class","Reaction Name","SMILES",
    "Reactants","Products","Catalysts","Agents","Solvents",
    "Action Order","Actions"
]

written = skipped_quality = parse_fail = 0
next_row_id = 1

with open(out_path, "w", newline="", encoding="utf-8-sig") as fout:
    writer = csv.DictWriter(
        fout,
        fieldnames=fieldnames,
        delimiter=",",
        quoting=csv.QUOTE_ALL,
        quotechar='"',
        doublequote=True,
        lineterminator="\n",
    )
    writer.writeheader()

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if not file.endswith(".json"):
                continue
            filepath = os.path.join(subdir, file)
            with open(filepath, encoding="utf-8") as fin:
                for line in fin:
                    try:
                        rxndata = json.loads(line)
                    except json.JSONDecodeError:
                        parse_fail += 1
                        continue

                    try:
                        data_value = rxndata.get("data", {}) or {}
                        quality = norm_str(data_value.get("qualityRank", "")).upper()
                        if quality not in ("S","A"):
                            skipped_quality += 1
                            continue

                        row = {
                            "row_id": next_row_id,
                            "Patent Title": norm_str(rxndata.get("title", "")),
                            "Date": data_value.get("date", ""),
                            "Quality": data_value.get("qualityRank", ""),
                            "Reaction Class": data_value.get("namerxn", ""),
                            "Reaction Name": data_value.get("namerxndef", ""),
                            "SMILES": data_value.get("smiles", ""),
                        }

                        role_smiles = {"Product": [], "Reactant": [], "Catalyst": [], "Agent": [], "Solvent": []}
                        component_ls = rxndata.get("components", []) or []
                        if is_list(component_ls):
                            for comp in component_ls:
                                if not is_dict(comp): continue
                                role = comp.get("role", "")
                                smi  = comp.get("smiles", "")
                                if role in role_smiles and smi:
                                    role_smiles[role].append(smi)
                        row["Products"]  = ";".join(role_smiles["Product"])
                        row["Reactants"] = ";".join(role_smiles["Reactant"])
                        row["Catalysts"] = ";".join(role_smiles["Catalyst"])
                        row["Agents"]    = ";".join(role_smiles["Agent"])
                        row["Solvents"]  = ";".join(role_smiles["Solvent"])

                        actions_out, order_seq = [], []
                        action_ls = rxndata.get("actions", []) or []
                        if is_list(action_ls):
                            for act in action_ls:
                                if not is_dict(act): continue
                                t = norm_str(act.get("type", ""))
                                temp_obj, time_val = extract_temp_and_time(act.get("parameters", []) or [])
                                comps_list = extract_components(act.get("components", []) or [])
                                yld = extract_yield_for_action(t, act.get("components", []) or [])

                                actions_out.append({
                                    "type": t,
                                    "Temperature": temp_obj,
                                    "Time": time_val,
                                    "Component": comps_list,
                                    "Text": norm_str(act.get("text", "")),
                                    "Yield": yld
                                })
                                order_seq.append(t)

                        row["Action Order"] = ";".join(order_seq)
                        row["Actions"] = json.dumps(actions_out, ensure_ascii=False)

                        writer.writerow(row)
                        written += 1
                        next_row_id += 1
                    except Exception:
                        parse_fail += 1
                        continue


import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import AllChem

pd.set_option('display.max_colwidth', None)

def clean_smiles_text(smiles):
    if not isinstance(smiles, str):
        return ''
    smiles = re.sub(r'\|[^\|]+\|', '', smiles)
    smiles = smiles.strip().replace(" ", "")
    return smiles

def clean_reaction(rxn):
    if not isinstance(rxn, str):
        return rxn
    if '>>' in rxn:
        return rxn
    parts = rxn.split('>')
    if len(parts) == 3:
        return parts[0] + '>>' + parts[2]
    elif len(parts) == 1:
        return rxn.replace('>', '>>')
    return rxn

def validate_atommapped_reaction(rxn):
    if not isinstance(rxn, str): return False
    try:
        parts = rxn.split('>>')
        if len(parts) != 2: return False
        
        react_mol = Chem.MolFromSmiles(parts[0])
        prod_mol = Chem.MolFromSmiles(parts[1])
        
        if react_mol is None or prod_mol is None:
            return False
        
        if react_mol == prod_mol:
            return False
            
        Chem.SanitizeMol(react_mol)
        Chem.SanitizeMol(prod_mol)
        
        if react_mol.GetNumAtoms() == 0 or prod_mol.GetNumAtoms() == 0:
            return False
        
        def get_canonical_molecules(molecule_str):
            canonical_set = set()
            for mol_str in molecule_str.split('.'):
                mol_str = mol_str.strip()
                if not mol_str:
                    continue
                try:
                    mol = Chem.MolFromSmiles(mol_str)
                    if mol is not None:
                        for atom in mol.GetAtoms():
                            atom.SetAtomMapNum(0)
                        canonical_smi = Chem.MolToSmiles(mol, canonical=True)
                        canonical_set.add(canonical_smi)
                except:
                    pass
            return canonical_set
        
        react_canonical = get_canonical_molecules(parts[0])
        prod_canonical = get_canonical_molecules(parts[1])
        
        if len(react_canonical & prod_canonical) > 0:
            return False
            
        return True
    except:
        return False
    

from rdkit import RDLogger
import gc
RDLogger.DisableLog('rdApp.*')

def demap_multi(smis: str):
    if not isinstance(smis, str) or not smis.strip():
        return True, smis
    
    clean_list = []
    for smi in smis.split(";"):
        s = smi.strip()
        if not s:
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return False, smis, f"RDKit failed to parse SMILES: {s}"
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        clean_list.append(Chem.MolToSmiles(mol, canonical=True))
    
    return True, ";".join(sorted(set(clean_list)))


valid_file = "Raw_rxn_R_P_R_Canon.csv"

cols = ["SMILES", "Reaction Class", "Reaction Name", "Reactants", "Products"]

COLS_TO_PROCESS = ["Reactants", "Products", "Catalysts", "Solvents", "Agents"]

CHUNK_SIZE = 1000000


first_chunk = True
total_raw = 0
total_valid_cols = 0
total_removed_cols = 0

total_valid_canon = 0
total_invalid_canon = 0

for chunk_num, df_chunk in enumerate(pd.read_csv("Raw_data_actiondetails_LIST.csv", chunksize=CHUNK_SIZE), 1):
    total_raw += len(df_chunk)
    
    df_nan_removed = df_chunk[df_chunk[cols].isna().any(axis=1)]
    
    df_nonan = df_chunk.dropna(subset=cols)
    df_blank_removed = df_nonan[(df_nonan[cols] == "").any(axis=1)]
    
    df_nonan = df_nonan[~((df_nonan["Reaction Class"] == 0) | (df_nonan["Reaction Class"] == "0"))]
    
    df_nonan = df_nonan[df_nonan["Reaction Name"] != "Unrecognized"]
    
    df_S = df_nonan[~(df_nonan[cols] == "").any(axis=1)].copy()
    
    df_removed = pd.concat([df_nan_removed, df_blank_removed]).drop_duplicates()

    total_valid_cols += len(df_S)
    total_removed_cols  += len(df_removed)

    invalid_rows = []
    valid_indices = []
    
    for idx, row in df_S.iterrows():
        try:
            smiles_str = str(row['SMILES'])
            smiles_cleaned = clean_smiles_text(smiles_str)
            smiles_cleaned = clean_reaction(smiles_cleaned)
            
            if not validate_atommapped_reaction(smiles_cleaned):
                raise ValueError("Invalid SMILES")
            
            df_S.at[idx, 'SMILES'] = smiles_cleaned

            col_updates = {}
            for col in COLS_TO_PROCESS:
                if col not in df_S.columns:
                    continue
                ok, new_val = demap_multi(row[col])
                if not ok:
                    raise ValueError(f"{col} error")
                col_updates[col] = new_val
            
            for col, val in col_updates.items():
                df_S.at[idx, col] = val
            valid_indices.append(idx)
            
        except:
            invalid_rows.append(row.to_dict())
        

    df_valid = df_S.loc[valid_indices].reset_index(drop=True)
    df_invalid = pd.DataFrame(invalid_rows)

    mode = 'w' if first_chunk else 'a'
    header = first_chunk

    df_valid.to_csv(valid_file, mode=mode, header=header, index=False)

    total_valid_canon += len(df_valid)
    total_invalid_canon += len(df_invalid)


    first_chunk = False
    del df_chunk, df_nan_removed, df_nonan, df_blank_removed, df_S, df_removed, df_valid, df_invalid, invalid_rows, valid_indices
    gc.collect()


RDLogger.DisableLog('rdApp.*')

def all_temperature_empty_dict(actions_json):
    try:
        actions = json.loads(actions_json) if isinstance(actions_json, str) else actions_json
    except Exception:
        return True
    if not isinstance(actions, list):
        return True
    for a in actions:
        if isinstance(a, dict):
            temp = a.get("Temperature", "")
            if isinstance(temp, dict):
                return False
            elif str(temp).strip() != "":
                return False
    return True

CHUNK_SIZE = 5000000


input_file = "Raw_rxn_R_P_R_Canon.csv"
output_file_with_temp = "Raw_rxn_R_P_R_Canon_withtemp.csv"
output_file_without_temp = "Raw_rxn_R_P_R_Canon_withouttemp.csv"

first_chunk = True
total_processed = 0
total_with_temp = 0
total_without_temp = 0

for chunk_num, df in enumerate(pd.read_csv(input_file, chunksize=CHUNK_SIZE, encoding="utf-8"), 1):
    
    with_temp_indices = []
    without_temp_indices = []
    
    for idx, row in df.iterrows():
        if not all_temperature_empty_dict(row['Actions']):
            with_temp_indices.append(idx)
        else:
            without_temp_indices.append(idx)
    
    df_with_temp = df.loc[with_temp_indices].reset_index(drop=True)
    df_without_temp = df.loc[without_temp_indices].reset_index(drop=True)
    
    mode = 'w' if first_chunk else 'a'
    header = first_chunk
    
    if not df_with_temp.empty:
        df_with_temp.to_csv(output_file_with_temp, mode=mode, header=header, index=False, encoding='utf-8')
    
    if not df_without_temp.empty:
        df_without_temp.to_csv(output_file_without_temp, mode=mode, header=header, index=False, encoding='utf-8')
    
    first_chunk = False
    
    total_processed += len(df)
    total_with_temp += len(df_with_temp)
    total_without_temp += len(df_without_temp)

    del df, df_with_temp, df_without_temp, with_temp_indices, without_temp_indices
    gc.collect()


from collections import defaultdict

INPUT_FILE = "Raw_rxn_R_P_R_Canon_withtemp.csv"
OUTPUT = "Raw_rxn_R_P_R_Canon_withtemp_tagged.csv"
CHUNK_SIZE = 3500000

def parse_actions(actions_json):
    if actions_json is None:
        return []
    if isinstance(actions_json, list):
        return actions_json
    s = str(actions_json).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return json.loads(s.replace("'", '"'))

def tag_actions(actions):
    counts = defaultdict(int)
    tagged = []

    for a in actions:
        if not isinstance(a, dict):
            tagged.append(a)
            continue
        t = (a.get("type", "") or "").strip()
        counts[t] += 1
        k = counts[t]
        t_tag = f"{t}[{k}]"
        b = dict(a)
        b["type"] = t_tag
        tagged.append(b)

    order_tagged = ";".join(x.get("type", "") for x in tagged)
    return tagged, order_tagged

first_chunk = True
total_processed = 0


for chunk_num, df_chunk in enumerate(pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE), 1):
    
    tagged_actions_str = []
    tagged_orders_str = []

    for _, row in df_chunk.iterrows():
        actions = parse_actions(row.get("Actions", "[]"))
        actions_tagged, order_tagged = tag_actions(actions)
        tagged_actions_str.append(json.dumps(actions_tagged, ensure_ascii=False))
        tagged_orders_str.append(order_tagged)

    df_chunk["Actions"] = tagged_actions_str
    df_chunk["Action Order"] = tagged_orders_str

    mode = 'w' if first_chunk else 'a'
    header = first_chunk
    
    df_chunk.to_csv(OUTPUT, mode=mode, header=header, index=False)
    
    total_processed += len(df_chunk)
    first_chunk = False
    
    del df_chunk, tagged_actions_str, tagged_orders_str
    gc.collect()


import re, json, ast
from typing import Any, List, Dict, Optional

WORKUP_BASES = {
    "Extract[1]", "Purify[1]", "Recover[1]", "Filter[1]", "Quench[1]",
    "Concentrate[1]", "Partition[1]", "Dry[1]", "Wash[1]", "PhChange[1]"
}
NON_REACTANT_ROLES = {"unknown", "catalyst", "solvent", "agent"}

def strip_index(tok: str) -> str:
    return re.sub(r"\[\d+\]$", "", tok.strip()) if isinstance(tok, str) else ""

def get_index(tok: str) -> int:
    m = re.search(r"\[(\d+)\]$", tok.strip()) if isinstance(tok, str) else None
    return int(m.group(1)) if m else 1

def parse_actions(s):
    if isinstance(s, list): return s
    if s is None or (isinstance(s, float) and pd.isna(s)): return []
    st = str(s).strip()
    if not st: return []
    for parser in (json.loads, lambda x: json.loads(x.replace("'", '"')), ast.literal_eval):
        try:
            v = parser(st)
            if isinstance(v, list): return v
            if isinstance(v, dict): return [v]
        except Exception:
            pass
    return []

def norm_components(obj) -> List[Dict[str, Any]]:
    if obj is None: return []
    if isinstance(obj, dict): return [obj]
    if isinstance(obj, list): return [c for c in obj if isinstance(c, dict)]
    return []

def get_action_by_type_nth(actions, base: str, nth: int) -> Optional[Dict[str, Any]]:
    cnt = 0
    for a in actions:
        if not isinstance(a, dict): 
            continue
        t = str(a.get("type", "")).strip()
        if strip_index(t).lower() == base.lower():
            cnt += 1
            if cnt == nth:
                return a
    return None

def has_role(a: Dict[str, Any], roles_lower: set) -> bool:
    comps = norm_components(a.get("Component", [])) if isinstance(a, dict) else []
    for c in comps:
        role = str(c.get("role") or c.get("Role") or "").strip().lower()
        if role in roles_lower:
            return True
    return False

def has_any_component(a: Dict[str, Any]) -> bool:
    comps = norm_components(a.get("Component", [])) if isinstance(a, dict) else []
    return len(comps) > 0

def add_run_start(tokens: List[str], pos_before: int) -> int:
    j = pos_before
    while j >= 0 and strip_index(tokens[j]) == "Add":
        j -= 1
    return j + 1

def transform_action_order_with_before(order: str, actions_raw):
    if not isinstance(order, str) or not order.strip():
        return pd.Series({"Before_Main[1]": "", "Action Order_Main[1]": order})

    tokens = [t.strip() for t in order.split(";") if t.strip()]
    if not tokens:
        return pd.Series({"Before_Main[1]": "", "Action Order_Main[1]": ""})

    k = next((i for i, t in enumerate(tokens) if t in WORKUP_BASES), -1)

    if k == -1:
        before_main = ";".join(tokens)
        return pd.Series({"Before_Main[1]": before_main, "Action Order_Main[1]": ("Main[1]" if before_main else "")})

    first_workup_base = strip_index(tokens[k])
    actions = parse_actions(actions_raw)

    if first_workup_base == "Quench":
        q_nth = get_index(tokens[k])
        quench_act = get_action_by_type_nth(actions, "Quench", q_nth)
        if quench_act and has_any_component(quench_act):
            before_main = ";".join(tokens[:k]) if k > 0 else ""
            out = []
            if before_main:
                out.append("Main[1]")
            out.extend(tokens[k:])
            new_order = ";".join(out)
            return pd.Series({"Before_Main[1]": before_main, "Action Order_Main[1]": new_order})


    if k > 0:
        prev_tok = tokens[k-1]
        prev_base = strip_index(prev_tok)

        if prev_base == "Stir":
            nth = get_index(prev_tok)
            stir_act = get_action_by_type_nth(actions, "Stir", nth)

            if stir_act and has_role(stir_act, {"reactant"}):
                start_exclude = k
            else:
                if stir_act and has_role(stir_act, NON_REACTANT_ROLES):
                    add_start = add_run_start(tokens, k-2)
                    start_exclude = add_start if add_start < k-1 else k-1
                else:
                    if k-2 >= 0 and strip_index(tokens[k-2]) == "Add":
                        start_exclude = _apply_rule2_add_block(tokens, actions, k, nearest_add_pos=k-2)
                    else:
                        start_exclude = k

            before_main = ";".join(tokens[:start_exclude]) if start_exclude > 0 else ""
            out = []
            if before_main:
                out.append("Main[1]")
            out.extend(tokens[start_exclude:])
            new_order = ";".join(out)
            return pd.Series({"Before_Main[1]": before_main, "Action Order_Main[1]": new_order})

        if prev_base == "Add":
            start_exclude = _apply_rule2_add_block(tokens, actions, k, nearest_add_pos=k-1)
            before_main = ";".join(tokens[:start_exclude]) if start_exclude > 0 else ""
            out = []
            if before_main:
                out.append("Main[1]")
            out.extend(tokens[start_exclude:])
            new_order = ";".join(out)
            return pd.Series({"Before_Main[1]": before_main, "Action Order_Main[1]": new_order})

    before_main = ";".join(tokens[:k]) if k > 0 else ""
    out = []
    if before_main:
        out.append("Main[1]")
    out.extend(tokens[k:])
    new_order = ";".join(out)
    return pd.Series({"Before_Main[1]": before_main, "Action Order_Main[1]": new_order})

def _apply_rule2_add_block(tokens: List[str], actions, k: int, nearest_add_pos: int) -> int:
    add_start = nearest_add_pos
    while add_start >= 0 and strip_index(tokens[add_start]) == "Add":
        add_start -= 1
    add_start += 1

    include_from_pos = None
    j = nearest_add_pos
    while j >= add_start and strip_index(tokens[j]) == "Add":
        nth = get_index(tokens[j])
        add_act = get_action_by_type_nth(actions, "Add", nth)
        if add_act and has_role(add_act, {"reactant"}):
            include_from_pos = j
            break
        j -= 1

    if include_from_pos is None:
        return add_start

    cut = k
    for tpos in range(include_from_pos + 1, k):
        base = strip_index(tokens[tpos])
        nth = get_index(tokens[tpos])
        act = get_action_by_type_nth(actions, base, nth)
        if act and has_any_component(act):
            cut = tpos
            break

    return cut


CHUNK_SIZE = 2500000

input_file = "Raw_rxn_R_P_R_Canon_withtemp_tagged.csv"
output_file = "Raw_rxn_R_P_R_Canon_withtemp_tagged_main.csv"

first_chunk = True
total_processed = 0

for chunk_num, df in enumerate(pd.read_csv(input_file, chunksize=CHUNK_SIZE, encoding="utf-8"), 1):

    df[["Before_Main[1]", "Action Order_Main[1]"]] = df.apply(
        lambda r: transform_action_order_with_before(r["Action Order"], r["Actions"]), 
        axis=1
    )

    mode = 'w' if first_chunk else 'a'
    header = first_chunk
    
    df.to_csv(output_file, mode=mode, header=header, index=False, encoding='utf-8-sig')
    
    total_processed += len(df)
    first_chunk = False
    
    
    del df
    gc.collect()


def safe_load(x):
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return None
        try:
            return json.loads(x)
        except Exception:
            return None
    return x

def keep_before_main(row):
    before = row["Before_Main[1]"]
    if pd.isna(before) or before == "":
        k = 0
    else:
        before_types = [t for t in before.split(";") if t]
        k = len(before_types)

    steps = json.loads(row["Actions"])

    before_steps = steps[:k]

    reduced = []
    for s in before_steps:
        reduced.append({
            "type": s.get("type"),
            "Temperature": s.get("Temperature", ""),
            "Component": s.get("Component",[]) or s.get("Components", [])
        })

    return json.dumps(reduced, ensure_ascii=False)

def parse_set(s):
    if not isinstance(s, str) or not s.strip():
        return set()
    return {x.strip() for x in s.split(";") if x.strip()}

def extract_temp_from_text(text):
    if not isinstance(text, str):
        return None
    
    m = re.search(
        r'([0-9]+(?:\s*[-–]\s*[0-9]+|\s+between\s+[0-9]+)?)\s*[째°]', 
        text
    )
    if not m:
        return None

    span = m.group(1)
    m2 = re.search(r'([0-9]+)', span)
    if not m2:
        return None
    return float(m2.group(1))

ZERO_TOKENS = {
    "0 - 5 ℃", "0 - 5ºc", "0 -10 ℃", "0 -10℃", "0 -20 ℃", "0 -25 ℃", "0 -25℃", "-0 oC",
    "0 -5 ℃", "0 celsius", "0 deg", "0 deg c", "0 degree", "0 degree celsius",
    "0 degrees", "0 degrees c", "0 degrees celsius", "0 degrees centigrade",
    "0 o c", "0 oc", "0 oc to rt", "0 oc to r.t.", "0 oc to rt",
    "0 oc – room temperature", "0 oc-rt", "0 oc.", "0 oc~10 oc", "0 oc~r.t.",
    "0 to 10 degrees c", "0 to 10 oc", "0 to 10oc", "0 to 10℃",
    "0 to 25 oc", "0 to 4 ℃", "0 to 5 oc", "0 to 5 ºc", "0 ~ 5 ℃",
    "0 º c", "0 ºc", "0 ºc - 5 ºc", "0 ºc to 25 ºc", "0 ºc.",
    "0 ℃", "0 ℃ to 10 ℃", "0 ℃ to 20 ℃", "0 ℃ to 25 ℃", "0 ℃ to rt",
    "0 ℃ to r.t.", "0 ℃ to rt", "0 ℃.", "0 ℃～rt", "0 ℃～rt",
    "0 ～ 25 ℃", "0 ～ 4 ℃", "0 ～ 5 ℃",
    "0-10 degrees c", "0-10 oc", "0-10 ℃", "0-10oc", "0-10℃",
    "0-15 ℃", "0-2 ºc", "0-20 ℃", "0-20℃", "0-25 ℃", "0-25oc", "0-25℃",
    "0-4 ℃", "0-5 oc", "0-5 ºc", "0-5 ℃", "0-50 ℃", "0-5oc", "0-5ºc", "0-5℃",
    "0-degree", "0deg c", "0o c", "0oc", "0oc to rt", "0oc to rt", "0oc.",
    "0oc~room temperature", "0~10 oc", "0~10 ℃", "0~5 oc", "0~5oc",
    "0º", "0º c", "0ºc", "0ºc ~ 5ºc", "0ºc-room temp",
    "0℃", "0℃ -rt", "0℃ to rt", "0℃ to rt", "0℃ ～ rt",
    "0℃ ～ room temperature", "0℃ ～ rt", "0℃-10℃", "0℃-5℃", "0℃.",
    "0℃～10℃", "0℃～rt", "0−5 oc", "0～10℃", "0～19℃", "0～20 ℃", "0～5 ℃", "0～5℃",
    "about 0 degrees", "about 0 oc", "about 0 ºc", "about 0 ℃",
    "about 0-10ºc", "about 0ºc", "about 0℃",
    "below 0 oc", "below 0 ℃", "below 0ºc",
    "zero degree", "zero degrees", "zero degrees c", "zero degrees celsius",
    "zero ° c", "zero °c", "∼0 degrees celsius", "˜0 degrees Celsius","~0 degrees Celsius",
    'zero degree', 'zero degrees', 'zero degrees C', 'zero degrees Celsius', 'zero degrees Centigrade',

    "zero degree Celsius",
    "zero ° C",
    "zero ° C.; zero ° C.; zero ° C.",
    "zero °C",
    "zero °C; zero °C",
}
ZERO_TOKENS = {t.lower().strip() for t in ZERO_TOKENS}

RT_TOKENS = {
    "rt", "RT", "ambient temperature","room temperature", "to room temperature",
    "~ rt", 'R.T', 'R.T.', 'about RT', 'about room temperature', 'ambient temp', 'ambient temp.', 'ambient tem\xadperature', 'around room temperature', 'below RT', 'below room temperature', 'r.t', 'r.t.', 'room temp', 'room temp.', 'room temperature approximately', 'room tempera\xadture', 'room temper\xadature', 'room tem\xadperature', '~ rt', '˜room temperature', '˜rt', '∼ rt',
    "r.t", "r.t.",
    "room temp", "room temp.",
    "room temperature approximately",
    "room temper­ature", "room tempera­ture",
    "room tempe­rature", "room temp­erature", "room tem­perature",
    "rt", "r.t", "r.t.", "r.t;",
    "room temp", "room temp.",
    "room temp", "room temp.",
    "room temp", "room temp.",
    "room temp", "room temp.",
    "room temp", "room temp.",
    "room temperature approximately",
    "room tempera­ture", "room temper­ature",
    "room tempe­rature", "room temp­erature", "room tem­perature",
    "r.t", "r.t.",
    "room temp", "room temp.",
    "ambient temp", "ambient temp.",
    "room temp", "room temp.",
    "room temp", "room temp.",
    "about rt", "about rt.",
    "about room temperature",
    "about ambient temperature",
    "around room temperature",
    "about rt",
    "about 25ºc", "about 25℃",
    "room temp", "room temp.",
    "room temp", "room temp.",
    "room temp", "room temp.",
    "room temp", "room temp.",
    "room temp", "room temp.",
    "rt", "r.t", "r.t.",
    "～ 25℃", "～25 ℃", "～25℃",
    "˜room temperature", "˜rt", "∼ rt", "below ambient temperature", "below room temperature",

    "> RT",
    "> room temperatur",
    "above RT",
    "above room temperature",
    "ambient temperature to RT",
    "ambient temperature",
    "approximately room temperature",
    "around ambient temperature",
    "around rt",
    "at least room temperature",
    "ca. room temperature",
    "room temperature or below",
    "room temperature or lower",
    "roughly ambient temperature",

}
RT_TOKENS = {t.lower().strip() for t in RT_TOKENS}

other_temp_tokens_75 = {"seventy- five degrees Celsius"}
other_temp_tokens_85 = {"eighty-five degrees Celsius"}

def parse_single_temp_token(tok: str):

    if not isinstance(tok, str):
        return None
    t = tok.strip().lower()

    if t in ZERO_TOKENS:
        return 0.0

    if t in RT_TOKENS:
        return 25.0

    nums = re.findall(r'-?\d+\.?\d*', t)
    if nums:
        first = float(nums[0])
        return first

    return None

def fill_temp_value_from_text(s):
    if pd.isna(s) or s == "":
        return s

    try:
        steps = json.loads(s)
    except Exception:
        return s

    for step in steps:
        temp = step.get("Temperature", None)

        if isinstance(temp, dict):
            v = temp.get("value", "")
            t = temp.get("text", "")
            if (v in ("", None)):
                if isinstance(t, str) and ("째" in t or "°" in t):
                    parsed = extract_temp_from_text(t)
                    if parsed is not None:
                        temp["value"] = parsed
                elif isinstance(t, str):
                    parsed = parse_single_temp_token(t)
                    if parsed is not None:
                        temp["value"] = parsed    

    return json.dumps(steps, ensure_ascii=False)

def parse_temp(s):
    steps = safe_load(s)
    if not steps:
        return pd.Series({"Main[1]_Temp": None,
                          "Main[1]_Temp_text": None})

    values, texts = [], []

    for step in steps:
        temp = step.get("Temperature", None)

        temp = safe_load(temp)
        if not temp:
            continue

        v = temp.get("value", "")
        t = temp.get("text", "")

        if v not in ("", None):
            values.append(v)
        if t:
            texts.append(t)

    if values:
        main_val = max(values)
        main_txt = None
    else:
        main_val = None
        main_txt = "; ".join(texts) if texts else None

    return pd.Series({"Main[1]_Temp": main_val,
                      "Main[1]_Temp_text": main_txt})

from typing import List, Dict, Any, Tuple, Optional, Union

def _project_component_fields(comp: Dict[str, Any]) -> Dict[str, Any]:
    def pick(d, keys):
        for k in keys:
            if k in d and d[k] not in (None, "", []):
                return d[k]
        return None
    role   = pick(comp, ["role", "Role"])
    name   = pick(comp, ["name", "Name", "component_name", "ComponentName"])
    smiles = pick(comp, ["smiles", "SMILES", "Smiles"])
    role   = str(role).strip()   if isinstance(role, (str, int, float)) and role is not None else None
    name   = str(name).strip()   if isinstance(name, (str, int, float)) and name is not None else None
    smiles = str(smiles).strip() if isinstance(smiles, (str, int, float)) and smiles is not None else None
    return {"role": role, "name": name, "smiles": smiles}

def make_main1_comp(row):
    s = row["Main[1]_Actions"]
    if not isinstance(s, str) or not s.strip():
        return ""

    try:
        steps = json.loads(s)
    except Exception:
        return ""

    comps_flat = []
    for step in steps:
        comps = step.get("Component", []) or step.get("Components", []) or []
        if isinstance(comps, list):
            for c in comps:
                if isinstance(c, dict):
                    comps_flat.append(_project_component_fields(c))

    return json.dumps(comps_flat, ensure_ascii=False)

def _get(d, keys):
    for k in keys:
        if isinstance(d.get(k), str) and d[k].strip():
            return d[k].strip()
    return None

def normalize_role(s: str) -> str:
    s = (s or "").strip().lower()
    if s in {"reactant","reagent-reactant","substrate"}: return "Reactant"
    if s in {"product"}: return "Product"
    if s in {"catalyst"}: return "Catalyst"
    if s in {"agent","reagent","additive","base","acid","oxidant","reducing agent"}: return "Agent"
    if s in {"solvent","co-solvent"}: return "Solvent"
    return "Unknown"

def split_comp_to_roles(comp_json_str: str):
    buckets = {
        "Reactant": [], "Product": [], "Catalyst": [],
        "Agent": [], "Solvent": [], "Unknown": []
    }
    if not isinstance(comp_json_str, str) or not comp_json_str.strip():
        return {k: "" for k in buckets}

    try:
        comps = json.loads(comp_json_str)
    except Exception:
        try:
            comps = json.loads(comp_json_str.replace("'", '"'))
        except Exception:
            return {k: "" for k in buckets}

    if not isinstance(comps, list):
        return {k: "" for k in buckets}

    for c in comps:
        if not isinstance(c, dict):
            continue
        role_raw = _get(c, ["role","Role"])
        role = normalize_role(role_raw or "")
        smiles = _get(c, ["smiles","SMILES","Smiles"])
        if smiles:
            buckets[role].append(smiles)

    return {k: ";".join(v) if v else "" for k, v in buckets.items()}


CHUNK_SIZE = 2000000

input_file = "Raw_rxn_R_P_R_Canon_withtemp_tagged_main.csv"
output_file = "Raw_rxn_R_P_R_Canon_withtemp_tagged_main_temp_comp.csv"

first_chunk = True
total_processed_df = 0
total_processed_df_empty = 0

for chunk_num, df in enumerate(pd.read_csv(input_file,
                                           chunksize=CHUNK_SIZE,
                                           encoding="utf-8"),
                               1):


    df["Main[1]_Actions"] = df.apply(keep_before_main, axis=1)

    df["Main[1]_Actions"] = df["Main[1]_Actions"].apply(fill_temp_value_from_text)
    temp_parsed = df["Main[1]_Actions"].apply(parse_temp)
    df = pd.concat([df, temp_parsed], axis=1)


    df["Main1_Comp"] = df.apply(make_main1_comp, axis=1)

    split_cols = df["Main1_Comp"].apply(split_comp_to_roles)
    df["Main1_Catalyst"] = split_cols.map(lambda d: d["Catalyst"])
    df["Main1_Agent"]    = split_cols.map(lambda d: d["Agent"])
    df["Main1_Solvent"]  = split_cols.map(lambda d: d["Solvent"])

    
    mask_nonempty = (
        df["Main[1]_Temp"].notna()
        & (df["Main[1]_Temp"].astype(str).str.strip() != "")
    )
    df_temp = df[mask_nonempty]
    df_empty = df[~mask_nonempty]


    mode = "w" if first_chunk else "a"
    header = first_chunk

    df_temp.to_csv(output_file, mode=mode, header=header,
              index=False, encoding="utf-8-sig")
    df_empty.to_csv(output_file.replace(".csv", "_tempempty.csv"), mode=mode, header=header, index=False, encoding="utf-8-sig")
    total_processed_df += len(df_temp)
    total_processed_df_empty += len(df_empty)
    first_chunk = False


    del df
    del df_empty
    gc.collect()


base_file = "Raw_rxn_R_P_R_Canon_withtemp_tagged_main_temp_comp.csv"
append_file = "Raw_rxn_R_P_R_Canon_withtemp_tagged_main_temp_comp_temptext_only_withvalue.csv"
out_file = "Raw_rxn_R_P_R_Canon_withtemp_tagged_main_temp_comp_MERGED.csv"

chunksize = 2500000

first = True
total = 0
for chunk in pd.read_csv(base_file, encoding="utf-8-sig", chunksize=chunksize):
    chunk.to_csv(out_file, index=False, encoding="utf-8-sig", mode="w" if first else "a", header=first)
    total += len(chunk)
    first = False

df_append = pd.read_csv(append_file, encoding="utf-8-sig")
df_append.to_csv(out_file, index=False, encoding="utf-8-sig", mode="a", header=False)


import math


base_file = "Raw_rxn_R_P_R_Canon_withtemp_tagged_main_temp_comp.csv"
out_file = "Raw_rxn_R_P_R_Canon_Temp.csv"


cols_2 = [
    'row_id', 'Patent Title', 'Date', 'Quality', 'Reaction Class', 'Reaction Name',
    'SMILES', 'Reactants', 'Products', 'Catalysts', 'Agents', 'Solvents', 'Reagents', 'Main[1]_Temp',
]

base_cols = ["Catalysts", "Solvents", "Agents"]

next_row_id = 0

def split_cell(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return []
    if not isinstance(x, str):
        x = str(x)
    return [s.strip() for s in x.split(";") if s.strip()]

def join_unique(items):
    return ";".join(sorted(set(items))) if items else ""

def union_from_cols(row, cols):
    items = []
    for c in cols:
        items.extend(split_cell(row.get(c, "")))
    return join_unique(items)


chunksize = 2500000

first = True
total = 0
for chunk_num, chunk in enumerate(
    pd.read_csv(base_file, encoding="utf-8-sig", chunksize=chunksize),
    start=1):
    chunk["Reagents"] = chunk.apply(lambda r: union_from_cols(r, base_cols), axis=1)
    chunk = chunk[cols_2]

    chunk.to_csv(out_file, index=False, encoding="utf-8-sig", mode="w" if first else "a", header=first)
    total += len(chunk)
    first = False

del chunk, chunk_num


def sort_cell(cell):
    if pd.isna(cell) or not cell.strip():
        return ""
    parts = [s.strip() for s in cell.split(";") if s.strip()]
    parts_sorted = sorted(parts)
    return ";".join(parts_sorted)


INPUT = "Raw_rxn_R_P_R_Canon_Temp.csv"
OUTPUT = f"Raw_rxn_R_P_R_Canon_Temp_unique_smiles.csv"
OUT_KEYS = "Raw_rxn_R_P_R_Canon_Temp_keys_smiles.csv"

df = pd.read_csv(INPUT, dtype=str, encoding="utf-8-sig")

df["RP_key"] = df["SMILES"]
df["Full_key"] = df["RP_key"] + "::" + df["Reagents"].astype(str)


df_keys = df[["row_id","RP_key", "Full_key"]].copy()
df_keys.to_csv(OUT_KEYS, index=False)


df_unique = df.drop_duplicates(subset="Full_key", keep="first").copy()

cols_2 = [
    'row_id', 'Patent Title', 'Date', 'Quality', 'Reaction Class', 'Reaction Name',
    'SMILES', 'Reactants', 'Products', 'Catalysts', 'Agents', 'Solvents', 'Reagents', 'Main[1]_Temp',
]

df_unique = df_unique[cols_2]

df_unique.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick  

def count_reagents(cell):
    if pd.isna(cell) or not cell.strip():
        return 0
    return len([x for x in cell.split(";") if x.strip()])

def plot_distribution_with_cumulative(df, cols, font_size=16, figsize = (14, 10), row = 2, col = 2):
    fig, axes = plt.subplots(row, col, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(cols):
        count_series = df[col].apply(count_reagents)
        dist = count_series.value_counts().sort_index()
        cumulative = dist.cumsum() / dist.sum() * 100

        ax1 = axes[i]
        dist.plot(kind="bar", ax=ax1, color='skyblue', width=0.7)

        ax1.set_title(f"{col} Count Distribution", fontsize=font_size + 2)
        ax1.set_xlabel(f"Number of {col}", fontsize=font_size)
        ax1.set_ylabel("Frequency", fontsize=font_size)
        ax1.tick_params(axis='x', labelsize=font_size - 2)
        ax1.tick_params(axis='y', labelsize=font_size - 2)

        ax1.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

        ax2 = ax1.twinx()
        ax2.plot(cumulative.index, cumulative.values, color='red', marker='o', linestyle='--')
        ax2.set_ylabel("Cumulative Percentage (%)", fontsize=font_size)
        ax2.tick_params(axis='y', labelsize=font_size - 2)
        ax2.set_ylim(0, 110)

        shown_100 = False
        for x, y in zip(cumulative.index, cumulative.values):
            if y >= 99.99:
                if not shown_100:
                    ax2.text(x, 102, "100%", ha='center', va='bottom',
                             fontsize=font_size - 2, color='red')
                    shown_100 = True
                continue
            ax2.text(x, y + 2, f"{y:.1f}%", ha='center', va='bottom',
                     fontsize=font_size - 2, color='red')

        ax2.legend(["Cumulative %"], loc="lower right", fontsize=font_size - 2)

    plt.tight_layout()
    plt.show()


def get_frequent_set(df, threshold=30):
    all_reagents_flat = []
    for x in df["Reagents"].dropna():
        all_reagents_flat.extend([r.strip() for r in str(x).split(";") if r.strip()])
    counts = Counter(all_reagents_flat)
    rare = {r for r, c in counts.items() if c < threshold}
    frequent = set(counts.keys()) - rare
    return frequent, counts

def contains_only_frequent(reagent_str, frequent_set):
    if pd.isna(reagent_str) or reagent_str.strip() == "":
        return True
    reagents = [r.strip() for r in reagent_str.split(";") if r.strip()]
    return all(r in frequent_set for r in reagents)

cols = ['Catalysts', 'Agents', 'Solvents', 'Reagents']
plot_distribution_with_cumulative(df_unique, cols, font_size=12, figsize=(12, 8), row=2, col=2)


df = df_unique

catalysts_count = df['Catalysts'].apply(count_reagents)
catalysts_dist = catalysts_count.value_counts().sort_index()

for num, cnt in catalysts_dist.items():
    pct = cnt / len(df) * 100


for col in ['Agents', 'Solvents', 'Catalyst', 'Agent', 'Solvent']:
    if col in df.columns:
        count_dist = df[col].apply(count_reagents).value_counts().sort_index()


df = pd.read_csv('Raw_rxn_R_P_R_Canon_Temp_unique_smiles.csv')


all_lines = df[df['Reagents'] != '']['Reagents'].tolist()

all_reagents_flat = []
for line in all_lines:
    line = str(line).strip()
    if line and line != 'nan':
        all_reagents_flat.extend([x.strip() for x in line.split(';') if x.strip()])

counts = Counter(all_reagents_flat)

rare_reagents_50 = {reagent: count for reagent, count in counts.items() if count > 100}
rare_reagents_40 = {reagent: count for reagent, count in counts.items() if count > 50 and count <= 100}
rare_reagents_30 = {reagent: count for reagent, count in counts.items() if count <= 50}


from solvents import SOLVENTS as SOLVENTS_ALL

WATER =  {'O': 'water'}

def split_to_set(s):
    return {x.strip() for x in s.split(";") if x.strip()}

solvent_keys = set(SOLVENTS_ALL.keys())
water = set(WATER.keys())

df_unique = pd.read_csv('Raw_rxn_R_P_R_Canon_unique_smiles.csv', dtype=str)

full_solvent_water_map = {}
full_solvent_map = {}
water_reagent_map = {}
solvent_reagent_map = {}
no_solvent_reagent_map = {}

for s in df_unique['Reagents'].dropna():
    reagents = split_to_set(s)
    for reagent in reagents:
        fragments = {frag.strip() for frag in reagent.split(".") if frag.strip()}
        if not fragments:
            continue
        if len(fragments) > 1:
            if fragments.issubset(solvent_keys) and (fragments & water):
                full_solvent_water_map[reagent] = fragments
            elif fragments.issubset(solvent_keys):
                full_solvent_map[reagent] = fragments
            elif any(frag in water for frag in fragments):
                water_reagent_map[reagent] = fragments
            elif any(frag in solvent_keys for frag in fragments):
                solvent_reagent_map[reagent] = fragments
            else:
                no_solvent_reagent_map[reagent] = fragments


maps = {
    'full_solvent_water_map': full_solvent_water_map,
    'full_solvent_map': full_solvent_map,
    'water_reagent_map': water_reagent_map,
    'solvent_reagent_map': solvent_reagent_map,
    'no_solvent_reagent_map': no_solvent_reagent_map,
}

for map_name, mapping in maps.items():
    map_keys = set(mapping.keys())
    key_counts = Counter()
    
    for reagents in df_unique['Reagents'].fillna(''):
        reagents_list = [r.strip() for r in reagents.split(';') if r.strip()]
        
        for reagent in reagents_list:
            if reagent in map_keys:
                key_counts[reagent] += 1
    
    df_result = pd.DataFrame(
        sorted(key_counts.items(), key=lambda x: x[1], reverse=True),
        columns=['key', 'count']
    )
    
    filename = f'before_rule_{map_name}_key_counts_in_df.csv'
    df_result.to_csv(filename, index=False, encoding='utf-8-sig')
    


RDLogger.DisableLog('rdApp.*')


def canonicalize_smiles(smiles):
    if not smiles or not isinstance(smiles, str) or smiles.strip() == '':
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return smiles
        return Chem.MolToSmiles(mol)
    except:
        return smiles

def canon_list(smiles_list):
    return [canonicalize_smiles(s) for s in smiles_list if s and s.strip()]

history_tracker = defaultdict(lambda: defaultdict(int))


water_reagent_map = water_reagent_map
full_solvent_water_map = full_solvent_water_map
solvent_reagent_map = solvent_reagent_map
full_solvent_map = full_solvent_map
no_solvent_reagent_map = no_solvent_reagent_map

KNOWN_SOLVENTS = set(SOLVENTS_ALL.keys())


def process_row(row):
    target_cols = ['Catalysts', 'Solvents', 'Agents']
    
    temp_data = {col: [] for col in target_cols}

    for col in target_cols:
        raw_val = str(row[col])
        if not raw_val.strip(): continue
        
        raw_items = [item.strip() for item in raw_val.split(';') if item.strip()]
        
        for orig in raw_items:
            next_parts = []
            
            if orig in water_reagent_map:
                comps = [x for x in water_reagent_map[orig] if x != 'O']
                if comps:
                    next_parts.append('.'.join(sorted(comps)))
                
            elif orig in full_solvent_water_map:
                comps = [x for x in full_solvent_water_map[orig] if x != 'O']
                if comps:
                    next_parts.append('.'.join(sorted(comps)))
                
            else:
                next_parts.append(orig)
            
            cleaned_parts = canon_list(next_parts)
            
            if not cleaned_parts:
                temp_data[col].append((orig, None))
            else:
                for p in cleaned_parts:
                    temp_data[col].append((orig, p))

    has_pure_solvent = False
    for col in target_cols:
        for _, curr_smi in temp_data[col]:
            if '.' not in curr_smi:
                if curr_smi in KNOWN_SOLVENTS:
                    has_pure_solvent = True
                    break
        if has_pure_solvent: break

    final_processed_cols = {col: [] for col in target_cols}

    for col in target_cols:
        for orig, curr in temp_data[col]:
            
            parts_list = [curr]
            
            next_parts = []
            for item in parts_list:
                if item in solvent_reagent_map:
                    components = solvent_reagent_map[item]
                    
                    if has_pure_solvent:
                        reagents_only = []
                        for comp in components:
                            if canonicalize_smiles(comp) not in KNOWN_SOLVENTS:
                                reagents_only.append(comp)
                        
                        if reagents_only:
                            next_parts.append('.'.join(reagents_only))
                    else:
                        next_parts.extend(list(components))
                else:
                    next_parts.append(item)
            parts_list = canon_list(next_parts)

            next_parts = []
            for item in parts_list:
                if item in full_solvent_map:
                    components = full_solvent_map[item]
                    
                    if has_pure_solvent:
                        pass 
                    else:
                        next_parts.extend(list(components))
                else:
                    next_parts.append(item)
            parts_list = canon_list(next_parts)

            
            next_parts = parts_list
            
            final_parts = canon_list(next_parts)
            for final_smi in final_parts:
                final_processed_cols[col].append(final_smi)
                history_tracker[final_smi][orig] += 1
    
    merged_reagents = sorted(list(set(
        list(final_processed_cols['Catalysts']) +
        list(final_processed_cols['Solvents']) +
        list(final_processed_cols['Agents'])
    )))

    return pd.Series([
        ';'.join(sorted(list(set(final_processed_cols['Catalysts'])))),
        ';'.join(sorted(list(set(final_processed_cols['Solvents'])))),
        ';'.join(sorted(list(set(final_processed_cols['Agents'])))),
        ';'.join(merged_reagents)
    ], index=['Catalysts', 'Solvents', 'Agents', 'Reagents'])


df = df_unique
df[['Catalysts', 'Solvents', 'Agents', 'Reagents']] = df.apply(process_row, axis=1)

df.to_csv('Raw_rxn_Final_Corrected_SetBased.csv', index=False, encoding='utf-8-sig')


report_data = []

for final_smi, origins in history_tracker.items():
    total_count = sum(origins.values())
    
    sorted_origins = sorted(origins.items(), key=lambda item: item[1], reverse=True)
    detail_str = ", ".join([f"{src} : {cnt}" for src, cnt in sorted_origins])
    
    
    report_data.append({
        'Reagent_SMILES': final_smi,
        'count': total_count,
        'detail': detail_str
    })

df_report = pd.DataFrame(report_data)

df_report = df_report.sort_values('count', ascending=False).reset_index(drop=True)

output_report_file = 'Processing_Detail_Report.csv'
df_report.to_csv(output_report_file, index=False, encoding='utf-8-sig')


def regenerate_reagents(row):
    parts = []
    for col in ['Catalysts', 'Agents', 'Solvents']:
        val = row.get(col, '')
        if pd.isna(val):
            continue
        val = str(val).strip()
        if val and val.lower() not in ['nan', 'none', '']:
            items = [x.strip() for x in val.split(';') if x.strip() and x.strip().lower() not in ['nan', 'none']]
            parts.extend(items)
    
    unique_parts = sorted(set(parts))
    return ';'.join(unique_parts)

df = pd.read_csv('Raw_rxn_Final_Corrected_SetBased.csv')
df['Reagents'] = df.apply(regenerate_reagents, axis=1)


df.to_csv('Raw_rxn_Final_Corrected_SetBased.csv', index=False, encoding='utf-8-sig')


df = pd.read_csv('Raw_rxn_Final_Corrected_SetBased.csv')


nan_count = df['Reagents'].isna().sum()

all_lines = df[(df['Reagents'] != '') & (df['Reagents'].notna())]['Reagents'].tolist()

all_reagents_flat = []
for line in all_lines:
    all_reagents_flat.extend([x.strip() for x in line.split(';') if x.strip()])

counts = Counter(all_reagents_flat)

rare_reagents_50 = {reagent: count for reagent, count in counts.items() if count > 100}
rare_reagents_40 = {reagent: count for reagent, count in counts.items() if count > 50 and count <= 100}
rare_reagents_30 = {reagent: count for reagent, count in counts.items() if count <= 50}


df_reagents = pd.DataFrame(list(counts.items()), columns=['reagent', 'count'])
df_reagents = df_reagents.sort_values('count', ascending=False).reset_index(drop=True)
df_reagents.to_csv('Raw_rxn_Final_Corrected_SetBased_reagent_counts.csv', index=False)

cols = ['Catalysts', 'Agents', 'Solvents', 'Reagents']
plot_distribution_with_cumulative(df, cols, font_size=12, figsize=(12, 8), row=2, col=2)

df = pd.read_csv('Raw_rxn_Final_Corrected_SetBased.csv')
threshold = 100
prev_len = None


while True:
    frequent_reagents, counts = get_frequent_set(df, threshold=threshold)
    df["Reagents_frequent_only"] = df["Reagents"].apply(
        lambda x: contains_only_frequent(x, frequent_reagents)
    )
    new_df = df[df["Reagents_frequent_only"]].copy()
    
    if prev_len is not None and len(new_df) == prev_len:
        break
    
    prev_len = len(new_df)
    df = new_df

df_final = df.drop(columns=["Reagents_frequent_only"])

cols = ['Catalysts', 'Agents', 'Solvents', 'Reagents']
plot_distribution_with_cumulative(df_final, cols)

df_final.to_csv('Raw_rxn_Final_Corrected_SetBased_100.csv', index=False, encoding='utf-8-sig')

from string import digits
from typing import Tuple, List, Callable
from rdkit.Chem import Fragments
import typing as tp


from reagents_classification import HeuristicRoleClassifier
from solvents import SOLVENTS

def classify_reagents(reagent_str):
    if pd.isna(reagent_str) or not str(reagent_str).strip():
        return {
            "Catalyst": "",
            "Agent": "",
            "Solvent": "",
        }
    
    try:
        reagent_str = str(reagent_str).strip()
        cats, ox, red, acid, base, unspec, solv = HeuristicRoleClassifier.classify(reagent_str)
        agents = ox + red + acid + base + unspec
        return {
            "Catalyst": ";".join(cats),
            "Agent": ";".join(agents),
            "Solvent": ";".join(solv),
        }
    except Exception as e:
        return {
            "Catalyst": "",
            "Agent": "",
            "Solvent": "",
        }

RDLogger.DisableLog('rdApp.*')

df = pd.read_csv('Raw_rxn_Final_Corrected_SetBased_100.csv', dtype=str, encoding='utf-8-sig')
classified = df["Reagents"].apply(classify_reagents)
classified_df = pd.DataFrame(classified.tolist())

df = pd.concat([df, classified_df], axis=1)
df.to_csv('Raw_rxn_Final_Corrected_SetBased_100_reclassified.csv', index=False, encoding='utf-8-sig') 

cols = ['Reactants', 'Products', 'Reagents']
plot_distribution_with_cumulative(df, cols, font_size=12, figsize=(18, 4), row=1, col=3)

cols = ['Catalyst', 'Agent', 'Solvent']
plot_distribution_with_cumulative(df, cols, font_size=12, figsize=(18, 4), row=1, col=3)


INPUT = "Raw_rxn_Final_Corrected_SetBased_100_reclassified.csv"
OUT_SUMMARY = "RP_conflict_summary_smiles_100.csv"
OUT_DETAIL = "RP_conflict_detail_smiles_100.csv"


def sort_cell(cell):
    if pd.isna(cell) or not str(cell).strip():
        return ""
    parts = [s.strip() for s in str(cell).split(";") if s.strip()]
    return ";".join(sorted(parts))

cols_needed = [
    "row_id", "Patent Title", "Reaction Class", "Reaction Name",
    "SMILES", "Reagents", "Main[1]_Temp"
]
df = pd.read_csv(INPUT, dtype=str, encoding="utf-8-sig", usecols=lambda c: c in cols_needed)

grp = df.groupby("SMILES", dropna=False)
summary = grp.agg(
    n_rows=("SMILES", "size"),
    n_reagents=("Reagents", "nunique"),
    n_temps=("Main[1]_Temp", "nunique"),
).reset_index()

conflict_keys = summary.loc[(summary["n_reagents"] > 1) | (summary["n_temps"] > 1), "SMILES"]

summary_conflict = (
    summary[summary["SMILES"].isin(conflict_keys)]
    .sort_values(["n_rows", "n_reagents", "n_temps"], ascending=False)
    .reset_index(drop=True)
)

detail_conflict = (
    df[df["SMILES"].isin(conflict_keys)]
    .sort_values(["SMILES", "Reagents", "Main[1]_Temp", "row_id"], ascending=True)
    .reset_index(drop=True)
)

summary_conflict.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
detail_conflict.to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")


df = pd.read_csv('Raw_rxn_Final_Corrected_SetBased_100_reclassified.csv', dtype=str, encoding='utf-8-sig')


conflict_data = []
for rp, group in df.groupby('SMILES'):
    if len(group) > 1:
        conflict_data.extend(group.values)

df_conflict = pd.DataFrame(conflict_data, columns=df.columns)


conflict_types = {
    'reagent_and_temp_diff': 0,
    'reagent_same_temp_diff': 0,
    'reagent_diff_temp_same': 0,
}

rp_conflict_breakdown = []

for rp, group in df_conflict.groupby('SMILES'):
    group_reagents = group['Reagents'].unique()
    group_temps = group['Main[1]_Temp'].unique()
    
    n_rows = len(group)
    n_reagent_variants = len(group_reagents)
    n_temp_variants = len(group_temps)
    
    types_in_group = {
        'reagent_and_temp_diff': 0,
        'reagent_same_temp_diff': 0,
        'reagent_diff_temp_same': 0,
    }
    
    group_list = group.reset_index(drop=True)
    for i in range(len(group_list)):
        for j in range(i + 1, len(group_list)):
            row_i = group_list.iloc[i]
            row_j = group_list.iloc[j]
            
            reagent_same = row_i['Reagents'] == row_j['Reagents']
            temp_same = row_i['Main[1]_Temp'] == row_j['Main[1]_Temp']
            
            if not reagent_same and not temp_same:
                types_in_group['reagent_and_temp_diff'] += 1
            elif reagent_same and not temp_same:
                types_in_group['reagent_same_temp_diff'] += 1
            elif not reagent_same and temp_same:
                types_in_group['reagent_diff_temp_same'] += 1
    
    for key in conflict_types:
        conflict_types[key] += types_in_group[key]
    
    rp_conflict_breakdown.append({
        'SMILES': rp,
        'n_rows': n_rows,
        'n_reagent_variants': n_reagent_variants,
        'n_temp_variants': n_temp_variants,
        'reagent_and_temp_diff_pairs': types_in_group['reagent_and_temp_diff'],
        'reagent_same_temp_diff_pairs': types_in_group['reagent_same_temp_diff'],
        'reagent_diff_temp_same_pairs': types_in_group['reagent_diff_temp_same'],
    })


total_pairs = sum(conflict_types.values())

df_breakdown = pd.DataFrame(rp_conflict_breakdown)


df_breakdown.to_csv('conflict_analysis_by_rp_type.csv', index=False, encoding='utf-8-sig')

summary_stats = {
    'Category': [
        'Reagent+Temp Different',
        'Reagent Same, Temp Different',
        'Temp Same, Reagent Different',
        'TOTAL'
    ],
    'Pair_Count': [
        conflict_types['reagent_and_temp_diff'],
        conflict_types['reagent_same_temp_diff'],
        conflict_types['reagent_diff_temp_same'],
        total_pairs
    ],
    'Percentage': [
        f"{conflict_types['reagent_and_temp_diff']/total_pairs*100:.2f}%" if total_pairs > 0 else "N/A",
        f"{conflict_types['reagent_same_temp_diff']/total_pairs*100:.2f}%" if total_pairs > 0 else "N/A",
        f"{conflict_types['reagent_diff_temp_same']/total_pairs*100:.2f}%" if total_pairs > 0 else "N/A",
        "100.00%"
    ]
}

df_summary = pd.DataFrame(summary_stats)
df_summary.to_csv('conflict_analysis_summary_100.csv', index=False, encoding='utf-8-sig')


df = pd.read_csv('Raw_rxn_Final_Corrected_SetBased_100_reclassified.csv', dtype=str, encoding='utf-8-sig')


def parse_set(reagents_str):
    if pd.isna(reagents_str):
        return set()
    s = str(reagents_str).strip()
    if not s or s.lower() in ['nan', 'none']:
        return set()
    items = [x.strip() for x in s.split(';') if x.strip() and x.strip().lower() not in ['nan', 'none']]
    return set(items)

def get_temp_value(temp_str):
    if pd.isna(temp_str):
        return float('-inf')
    s = str(temp_str).strip()
    if not s or s.lower() in ['nan', 'none']:
        return float('-inf')
    try:
        return float(s)
    except:
        return float('-inf')

merged_data = []
selection_records = []

rp_counter = 0

for rp, group_df in df.groupby('SMILES', dropna=False):
    rp_counter += 1
    rp_key = f"RP_{rp_counter}"
    
    if len(group_df) == 1:
        idx = group_df.index[0]
        row = group_df.iloc[0].copy()
        merged_data.append(row)
        
        selection_records.append({
            'RP_key': rp_key,
            'original_index': idx,
            'is_selected': True,
            'selection_reason': 'Only option (single reaction)',
            'SMILES': rp
        })
        continue
    
    row_infos = []
    for idx, row in group_df.iterrows():
        reagents_set = parse_set(row['Reagents'])
        temp = get_temp_value(row.get('Main[1]_Temp', ''))
        row_infos.append({
            'original_idx': idx,
            'set': reagents_set,
            'set_size': len(reagents_set),
            'temp': temp,
            'row': row
        })
    
    set_to_rows = {}
    for info in row_infos:
        key = frozenset(info['set'])
        if key not in set_to_rows:
            set_to_rows[key] = []
        set_to_rows[key].append(info)
    
    representatives = []
    for set_key, items in set_to_rows.items():
        best = max(items, key=lambda x: x['temp'])
        representatives.append(best)
        
        for item in items:
            if item['original_idx'] != best['original_idx']:
                temp_display = f"{item['temp']:.1f}" if item['temp'] > float('-inf') else 'unknown'
                best_temp_display = f"{best['temp']:.1f}" if best['temp'] > float('-inf') else 'unknown'
                selection_records.append({
                    'RP_key': rp_key,
                    'original_index': item['original_idx'],
                    'is_selected': False,
                    'selection_reason': f'Lower temperature ({temp_display}°C) than selected ({best_temp_display}°C)',
                    'SMILES': rp
                })
    
    final_selected = []
    for rep in representatives:
        is_subset = False
        larger_set_info = None
        for other in representatives:
            if rep['set'] != other['set'] and rep['set'] < other['set']:
                is_subset = True
                larger_set_info = other
                break
        
        if not is_subset:
            final_selected.append(rep)
            merged_data.append(rep['row'].copy())
            
            temp_display = f"{rep['temp']:.1f}" if rep['temp'] > float('-inf') else 'unknown'
            selection_records.append({
                'RP_key': rp_key,
                'original_index': rep['original_idx'],
                'is_selected': True,
                'selection_reason': f'Largest reagent set (size={rep["set_size"]}, temp={temp_display}°C)',
                'SMILES': rp
            })
        else:
            reagents_display = ';'.join(sorted(rep['set'])) if rep['set'] else '(empty)'
            larger_reagents = ';'.join(sorted(larger_set_info['set'])) if larger_set_info['set'] else '(empty)'
            
            selection_records.append({
                'RP_key': rp_key,
                'original_index': rep['original_idx'],
                'is_selected': False,
                'selection_reason': f'Cannot merge - subset [{reagents_display}] < larger set [{larger_reagents}]',
                'SMILES': rp
            })

df_merged = pd.DataFrame(merged_data).reset_index(drop=True)
df_merged.to_csv('Raw_rxn_Final_Corrected_SetBased_100_reclassified_RP_merged.csv', index=False, encoding='utf-8-sig')


df_selection = pd.DataFrame(selection_records)
df_selection.to_csv('RP_selection_details.csv', index=False, encoding='utf-8-sig')


df = pd.read_csv('Raw_rxn_Final_Corrected_SetBased_100_reclassified_RP_merged.csv')
df.head(3)

def remove_unwanted_agents(cell):
    if pd.isna(cell) or not cell.strip():
        return ""
    agents = [s.strip() for s in cell.split(";") if s.strip()]
    filtered = [s for s in agents if s not in ["O", "N#N", 'O.O']]
    filtered_sorted = sorted(filtered)
    return ";".join(filtered_sorted)


df = pd.read_csv('Raw_rxn_Final_Corrected_SetBased_100_reclassified_RP_merged.csv')

cols = ['Catalyst', 'Agent', 'Solvent']
plot_distribution_with_cumulative(df, cols, font_size=12, figsize=(18, 4), row=1, col=3)

df['Catalyst'] = df['Catalyst'].apply(remove_unwanted_agents)
df['Solvent'] = df['Solvent'].apply(remove_unwanted_agents)
df['Agent'] = df['Agent'].apply(remove_unwanted_agents)
df['Reagents'] = df['Reagents'].apply(remove_unwanted_agents)

cols = ['Catalyst', 'Agent', 'Solvent']
plot_distribution_with_cumulative(df, cols, font_size=12, figsize=(18, 4), row=1, col=3)


def sort_cell(cell):
    if pd.isna(cell) or not cell.strip():
        return ""
    parts = [s.strip() for s in cell.split(";") if s.strip()]
    parts_sorted = sorted(parts)
    return ";".join(parts_sorted)


df["RP_key"] = df["SMILES"]
df["Full_key"] = df["RP_key"] + "::" + df["Reagents"].astype(str)
df["Full_key_temp"] = df["Full_key"] + "::(" + df["Main[1]_Temp"].astype(str) + ")"


df_unique = df.drop_duplicates(subset="Full_key_temp", keep="first").copy()

cols_2 = [
    'row_id', 'Patent Title', 'Date', 'Quality', 'Reaction Class', 'Reaction Name',
    'SMILES', 'Reagents', 'Main[1]_Temp', 'Catalyst', 'Agent', 'Solvent']

df_unique = df_unique[cols_2]

cols = ['Catalyst', 'Agent', 'Solvent']
plot_distribution_with_cumulative(df_unique, cols, font_size=12, figsize=(18, 4), row=1, col=3)


df = df[(df["Reagents"].apply(count_reagents) >= 1) & (df["Reagents"].apply(count_reagents) <= 5)]

df_limited = df[(df["Catalyst"].apply(count_reagents) <= 1) & (df["Agent"].apply(count_reagents) <= 2) & (df["Solvent"].apply(count_reagents) <= 2)]

df = df_limited
threshold = 100
prev_len = None


while True:
    frequent_reagents, counts = get_frequent_set(df, threshold=threshold)
    df["Reagents_frequent_only"] = df["Reagents"].apply(
        lambda x: contains_only_frequent(x, frequent_reagents)
    )
    new_df = df[df["Reagents_frequent_only"]].copy()
    
    if prev_len is not None and len(new_df) == prev_len:
        break
    
    prev_len = len(new_df)
    df = new_df

df_final = df.drop(columns=["Reagents_frequent_only"])

cols = ['Catalysts', 'Agents', 'Solvents', 'Reagents']
plot_distribution_with_cumulative(df_final, cols)

df_final.to_csv('P2026_Final.csv', index=False, encoding='utf-8-sig')

# Also save as data_total.csv in Data/, the input expected by prepare_model_data.py
_data_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../Data'))
os.makedirs(_data_dir, exist_ok=True)
df_final.to_csv(os.path.join(_data_dir, 'data_total.csv'), index=False, encoding='utf-8-sig')


df = pd.read_csv('P2026_Final.csv')

cols = ['Catalyst', 'Agent', 'Solvent', ]
plot_distribution_with_cumulative(df, cols, font_size=14, figsize=(20, 5), row=1, col=3)

def plot_distribution_single(df, col, font_size=14, figsize=(7, 5)):
    def count_reagents(cell):
        if pd.isna(cell) or not cell.strip():
            return 0
        return len([x for x in cell.split(";") if x.strip()])

    count_series = df[col].apply(count_reagents)
    dist = count_series.value_counts().sort_index()
    cumulative = dist.cumsum() / dist.sum() * 100

    fig, ax1 = plt.subplots(figsize=figsize)
    dist.plot(kind="bar", ax=ax1, color='skyblue', width=0.7)

    ax1.set_title(f"{col} Count Distribution", fontsize=font_size + 2)
    ax1.set_xlabel(f"Number of {col}", fontsize=font_size)
    ax1.set_ylabel("Frequency", fontsize=font_size)
    ax1.tick_params(axis='x', labelsize=font_size - 2)
    ax1.tick_params(axis='y', labelsize=font_size - 2)
    ax1.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    x_pos = range(len(cumulative))
    ax2.plot(x_pos, cumulative.values, color='red', marker='o', linestyle='--')
    ax2.set_ylabel("Cumulative Percentage (%)", fontsize=font_size)
    ax2.tick_params(axis='y', labelsize=font_size - 2)
    ax2.set_ylim(0, 110)

    shown_100 = False
    for x, y in zip(x_pos, cumulative.values):
        if y >= 99.99:
            if shown_100:
                continue
            shown_100 = True
        ax2.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                     xytext=(5, 3), fontsize=font_size - 4, color='red')

    plt.tight_layout()
    plt.show()

plot_distribution_single(df, 'Reagents', font_size=14, figsize=(7, 5))
