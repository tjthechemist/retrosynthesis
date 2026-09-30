import re
from copy import deepcopy
from random import shuffle

from rdkit import Chem
from rdkit.Chem import AllChem

def molecules_from_smiles_list(smiles_list: list[str]) -> list[Chem.Mol]:
    molecules: list[Chem.Mol] = []
    for smiles in smiles_list:
        if not smiles:
            continue
        molecules.append(Chem.MolFromSmiles(smiles))
    return molecules

def replace_deuterated(smiles: str) -> str:
    return re.sub(r"\[2H\]", r"\[H\]", smiles)

def clear_mapnumber(molecule: Chem.Mol) -> Chem.Mol:
    [atom.ClearProp("molAtomMapNumber") for atom in molecule.GetAtoms()]
    return molecule

def get_tagged_atoms_from_molecule(molecule: Chem.Mol) -> tuple[list[Chem.Atom], list[str]]:
    atoms: list[Chem.Atom] = []
    atom_tags: list[str]   = []
    for atom in molecule.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atoms.append(atom)
            atom_tags.append(str(atom.GetProp("molAtomMapNumber")))
    return atoms, atom_tags

def get_tagged_atoms_from_molecules(molecules: list[Chem.Mol]) -> tuple[list[Chem.Atom], list[str]]:
    atoms: list[Chem.Atom] = []
    atom_tags: list[str]   = []
    for molecule in molecules:
        new_atoms, new_atom_tags = get_tagged_atoms_from_molecule(molecule)
        atoms     += new_atoms
        atom_tags += new_atom_tags
    return atoms, atom_tags

def bond_to_label(bond: Chem.Bond) -> str:
    atom_1_label = str(bond.GetBeginAtom().GetAtomicNum())
    atom_2_label = str(bond.GetEndAtom().GetAtomicNum())
    if bond.GetBeginAtom().HasProp("molAtomMapNumber"):
        atom_1_label += bond.GetBeginAtom().GetProp("molAtomMapNumber")
    if bond.GetEndAtom().HasProp("molAtomMapNumber"):
        atom_2_label += bond.GetEndAtom().GetProp("molAtomMapNumber")
    atoms = sorted([atom_1_label, atom_2_label])

    return f"{atoms[0]}{bond.GetSmarts(bond)}{atoms[1]}"

def are_atom_different(atom_1: Chem.Atom, atom_2: Chem.Atom) -> bool:
    if atom_1.GetAtomicNum() != atom_2.GetAtomicNum():
        return True
    if atom_1.GetTotalNumHs() != atom_2.GetTotalNumHs():
        return True
    if atom_1.GetFormalCharge() != atom_2.GetFormalCharge():
        return True
    if atom_1.GetDegree() != atom_2.GetDegree():
        return True
    if atom_1.GetNumRadicalElectrons() != atom_2.GetNumRadicalElectrons():
        return True
    if atom_1.GetIsAromatic() != atom_2.GetIsAromatic():
        return True

    bonds_1 = sorted([bond_to_label(bond) for bond in atom_1.GetBonds()])
    bonds_2 = sorted([bond_to_label(bond) for bond in atom_2.GetBonds()])
    if bonds_1 != bonds_2:
        return True

    return False

def find_map_number(molecule: Chem.Mol, mapnumber: str) -> tuple[int, Chem.Atom]:
    return [(atom.GetIdx(), atom) for atom in molecule.GetAtoms() if atom.HasProp("molAtomMapNumber") and atom.GetProp("molAtomMapNumber") == str(mapnumber)][0]

def get_tetrahedral_atoms(reactants: list[Chem.Mol], products: list[Chem.Mol]) -> list[tuple[str, Chem.Atom, Chem.Atom]]:
    tetrahedral_atoms: list[tuple[str, Chem.Atom, Chem.Atom]] = []
    for reactant in reactants:
        for reactant_atom in reactant.GetAtoms():
            if not reactant_atom.HasProp("molAtomMapNumber"):
                continue
            atom_tag     = reactant_atom.GetProp("molAtomMapNumber")
            reactant_idx = reactant_atom.GetIdx()
            for product in products:
                try:
                    product_idx, product_atom = find_map_number(product, atom_tag)
                    if reactant_atom.GetChiralTag() != Chem.CHI_UNSPECIFIED or product_atom.GetChiralTag() != Chem.CHI_UNSPECIFIED:
                        tetrahedral_atoms.append((atom_tag, reactant_atom, product_atom))
                except IndexError:
                    pass
    return tetrahedral_atoms

def set_isotope_to_equal_mapnumber(molecule: Chem.Mol) -> None:
    for atom in molecule.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.SetIsotope(int(atom.GetProp("molAtomMapNumber")))

def get_fragment_around_tetrahedral_center(molecule: Chem.Mol, idx: int) -> str:
    ids_to_include: list[int] = [idx]
    for neighbor in molecule.GetAtomWithIdx(idx).GetNeighbors():
        ids_to_include.append(neighbor.GetIdx())
    symbols = [f"[{atom.GetIsotope()}{atom.GetSymbol()}]" if atom.GetIsotope() != 0 else f"[#{atom.GetAtomicNum()}]" for atom in molecule.GetAtoms()]
    return Chem.MolFragmentToSmiles(molecule, ids_to_include, isomericSmiles=True, atomSymbols=symbols, allBondsExplicit=True, allHsExplicit=True)

def check_tetrahedral_center_equivalent(atom_1: Chem.Atom, atom_2: Chem.Atom) -> bool:
    atom_1_fragments    = get_fragment_around_tetrahedral_center(atom_1.GetOwningMol(), atom_1.GetIdx())
    atom_1_neighborhood = Chem.MolFromSmiles(atom_1_fragments, sanitize=False)
    for matched_ids in atom_2.GetOwningMol().GetSubstructMatches(atom_1_neighborhood, useChirality=True):
        if atom_2.GetIdx() in matched_ids:
            return True
    return False

def clear_isotope(molecule: Chem.Mol) -> None:
    [atom.SetIsotope(0) for atom in molecule.GetAtoms()]

def get_changed_atoms(reactants: list[Chem.Mol], products: list[Chem.Mol]) -> tuple[list[Chem.Atom], list[str]]:

    product_atoms, product_atom_tags = get_tagged_atoms_from_molecules(products)
    print(f"Products contains {len(product_atoms)} tagged atoms.")
    print(f"Products contains {len(set(product_atom_tags))} unique atom numbers.")

    reactant_atoms, reactant_atom_tags = get_tagged_atoms_from_molecules(reactants)

    if len(set(product_atom_tags)) != len(set(reactant_atom_tags)):
        print("Warning: Different atom tags appear in reactants and products")
    if len(product_atoms) != len(reactant_atoms):
        print("Warning: Total number of tagged atom differ, stoichiometry != 1???")

    changed_atoms: list[Chem.Atom] = []
    changed_atom_tags: list[str]   = []

    for i, product_tag in enumerate(product_atom_tags):
        for j, reactant_tag in enumerate(reactant_atom_tags):
            if reactant_tag != product_tag:
                continue
            if reactant_tag not in changed_atom_tags:
                if are_atom_different(product_atoms[i], reactant_atoms[j]):
                    changed_atoms.append(reactant_atoms[j])
                    changed_atom_tags.append(reactant_tag)
                    break
                if product_atom_tags.count(reactant_tag) > 1:
                    changed_atoms.append(reactant_atoms[j])
                    changed_atom_tags.append(reactant_tag)
                    break

    for j, reactant_tag in enumerate(reactant_atom_tags):
        if reactant_tag not in changed_atom_tags:
            if reactant_tag not in product_atom_tags:
                changed_atoms.append(reactant_atoms[j])
                changed_atom_tags.append(reactant_tag)

    tetrahedral_atoms = get_tetrahedral_atoms(reactants, products)
    print(f"Found {len(tetrahedral_atoms)} atom-mapped tetrahedral atoms that have chirality specified at least partially.")
    [set_isotope_to_equal_mapnumber(reactant) for reactant in reactants]
    [set_isotope_to_equal_mapnumber(product) for product in products]

    for atom_tag, reactant_atom, product_atom in tetrahedral_atoms:
        print(f"For atom tag: {atom_tag}")
        print(f"Reactant:     {reactant_atom.GetChiralTag()}")
        print(f"Products:     {product_atom.GetChiralTag()}")
        if atom_tag in changed_atom_tags:
            print("-> atoms have changed (by more than just chirality!)")
        else:
            unchanged = check_tetrahedral_center_equivalent(reactant_atom, product_atom) and Chem.CHI_UNSPECIFIED not in [reactant_atom.GetChiralTag(), product_atom.GetChiralTag()]
            if unchanged:
                print("-> atoms confirmed to have same chirality, no change")
            else:
                print("-> atom changed chirality!!!")
                tetra_adjust_to_reaction = False
                for neighbor in product_atom.GetNeighbors():
                    if neighbor.HasProp("molAtomMapNumber"):
                        if neighbor.GetProp("molAtomMapNumber") in changed_atom_tags:
                            tetra_adjust_to_reaction = True
                            break

                if tetra_adjust_to_reaction:
                    print("-> atom adjust to reaction center, now included")
                    changed_atoms.append(reactant_atom)
                    changed_atom_tags.append(atom_tag)
                else:
                    print("-> adjust far from reaction center, not including")

    [clear_isotope(reactant) for reactant in reactants]
    [clear_isotope(product) for product in products]

    print(f"{len(changed_atom_tags)} tagged atoms in reactants change 1-atom properties.")
    for smarts in [atom.GetSmarts() for atom in changed_atoms]:
        print(f"{smarts}")

    return changed_atoms, changed_atom_tags

def get_special_groups(molecule: Chem.Mol) -> list[tuple[list[int], list[int]]]:

    group_templates  = [
        (range(3), "[OH0,SH0]=C[O,Cl,I,Br,F]",),
        (range(3), "[OH0,SH0]=CN",),
        (range(4), "S(O)(O)[Cl]",),
        (range(3), "B(O)(O)",),
        ((0,), "[Si](C)(C)C",),
        ((0,), "[Si](OC)(OC)OC",),
        (range(3), "[N;H0;$(N-[#6]);D2]-,=[N;D2]-,=[N;D1]",),
        (range(8), "O=C1N([Br,I,F,Cl])C(=O)CC1",),
        (range(11), "Cc1ccc(S(=O)(=O)O)cc1",),
        ((7,), "CC(C)(C)OC(=O)[N]",),
        ((4,), "[CH3][CH0]([CH3])([CH3])O",),
        (range(2), "[C,N]=[C,N]",),
        (range(2), "[C,N]#[C,N]",),
        ((2,), "C=C-[*]",),
        ((2,), "C#C-[*]",),
        ((2,), "O=C-[*]",),
        ((3,), "O=C([CH3])-[*]",),
        ((3,), "O=C([O,N])-[*]",),
        (range(4), "ClS(Cl)=O",),
        (range(2), "[Mg,Li,Zn,Sn][Br,Cl,I,F]",),
        (range(3), "S(O)(O)",),
        (range(2), "N~N",),
        ((1,), "[!#6;R]@[#6;R]",),
        ((2,), "[a!c]:a:a",),
        ((0,), "[B,C](F)(F)F",),
        ((1, 2,), "[*]/[CH]=[CH]/[*]",),
        ((1, 2,), "[*]/[CH]=[CH]\\[*]",),
        ((1, 2,), "[*]/[CH]=[CH0]([*])\\[*]",),
        ((1, 2,), "[*]/[D3;H1]=[!D1]",),
    ]

    groups: list[tuple[list[int], list[int]]] = []
    for add_if_match, template in group_templates:
        matches = molecule.GetSubstructMatches(Chem.MolFromSmarts(template), useChirality=True)
        for match in matches:
            add_if: list[int] = []
            for pattern_idx, atom_idx in enumerate(match):
                if pattern_idx in add_if_match:
                    add_if.append(atom_idx)
            groups.append((add_if, match))
    return groups

def convert_atom_to_wildcard(atom: Chem.Atom) -> str:
    if atom.GetDegree() == 1:
        symbol = f"[{atom.GetSymbol()};D1;H{atom.GetTotalNumHs()}"
        if atom.GetFormalCharge():
            charges = re.search(r"([-+]+[1-9]?)", atom.GetSmarts())
            symbol  = symbol.replace(";D1", f";{charges.group()};D1")

    else:
        symbol = "["
        if atom.GetAtomicNum() != 6:
            symbol += f"#{atom.GetAtomicNum()};"
            if atom.GetIsAromatic():
                symbol += "a;"
        elif atom.GetIsAromatic():
            symbol += "c;"
        else:
            symbol += "C;"

        if atom.GetFormalCharge() != 0:
            charges = re.search(r"([-+]+[1-9]?)", atom.GetSmarts())
            if charges:
                symbol += f"{charges.group()};"

        if symbol[-1] == ";":
            symbol = symbol[:-1]

    label = re.search(r"\:[0-9]+\]", atom.GetSmarts())
    if label:
        symbol += label.group()
    else:
        symbol += "]"

    if symbol != atom.GetSmarts():
        print(f"Improved generality of atom SMARTS {atom.GetSmarts()} -> {symbol}")

    return symbol

def expand_atoms_to_use_atom(molecule: Chem.Mol, atoms_to_use: list[int], atom_idx: int, groups: list[tuple[list[int], list[int]]] = [], symbol_replacements: list[tuple[int, str]] = []) -> tuple[list[int], list[tuple[int, str]]]:
    found_in_group = False
    for group in groups:
        if int(atom_idx) in group[0]:
            print("Adding group due to match")
            try:
                print(f"Match from molAtomMapNum {molecule.GetAtomWithIdx(atom_idx).GetProp("molAtomMapNUmber")}")
            except KeyError:
                pass
            for idx in group[1]:
                if idx not in atoms_to_use:
                    atoms_to_use.append(idx)
                    symbol_replacements.append((idx, convert_atom_to_wildcard(molecule.GetAtomWithIdx(idx))))
            found_in_group = True

    if found_in_group:
        return atoms_to_use, symbol_replacements

    if atom_idx in atoms_to_use:
        return atoms_to_use, symbol_replacements

    atoms_to_use.append(atom_idx)
    symbol_replacements.append((atom_idx, convert_atom_to_wildcard(molecule.GetAtomWithIdx(atom_idx))))

    return atoms_to_use, symbol_replacements

def expand_atoms_to_use(molecule: Chem.Mol, atoms_to_use: list[int], groups: list[tuple[list[int], list[int]]] = [], symbol_replacements: list[tuple[int, str]] = []) -> tuple[list[int], list[tuple[int ,str]]]:
    new_atom_to_use = atoms_to_use[:]
    for atom in molecule.GetAtoms():
        if atom.GetIdx() not in atoms_to_use:
            continue
        for group in groups:
            if int(atom.GetIdx()) in group[0]:
                print("Adding group due to match")
                try:
                    print(f"Match from molAtomMapNum {atom.GetProp("molAtomMapNumber")}")
                except KeyError:
                    pass
                for idx in group[1]:
                    if idx not in atoms_to_use:
                        new_atom_to_use.append(idx)
                        symbol_replacements.append((idx, convert_atom_to_wildcard(molecule.GetAtomWithIdx(idx))))

        for neighbor in atom.GetNeighbors():
            new_atom_to_use, symbol_replacements = expand_atoms_to_use_atom(molecule, new_atom_to_use, neighbor.GetIdx(), groups, symbol_replacements)

    return new_atom_to_use, symbol_replacements

def reassign_atom_mapping(transform: str) -> str:
    all_labels                       = re.findall(r"\:([0-9]+)\]", transform)
    replacements: list[str]          = []
    replacement_dict: dict[str, str] = {}
    counter          = 1
    for label in all_labels:
        if label not in replacement_dict:
            replacement_dict[label] = str(counter)
            counter += 1
        replacements.append(replacement_dict[label])

    transform_newmaps = re.sub(r"\:[0-9]+\]", lambda match: (f":{replacements.pop(0)}]"), transform)
    return transform_newmaps

def get_strict_smarts_for_atom(atom: Chem.Atom) -> str:
    symbol = atom.GetSmarts()
    if atom.GetSymbol() == "H":
        symbol = "[#1]"

    if "[" not in symbol:
        symbol = f"[{symbol}]"

    if atom.GetChiralTag() != Chem.CHI_UNSPECIFIED:
        if "@" not in symbol:
            if atom.GetChiralTag() == Chem.CHI_TETRAHEDRAL_CCW:
                tag = "@"
            elif atom.GetChiralTag() == Chem.CHI_TETRAHEDRAL_CW:
                tag = "@@"
            if ":" in symbol:
                symbol = symbol.replace(":", f";{tag}:")
            else:
                symbol = symbol.replace("]", f";{tag}]")

    if "H" not in symbol:
        H_symbol = f"H{atom.GetTotalNumHs()}"
        if ":" in symbol:
            symbol = symbol.replace(":", f";{H_symbol}:")
        else:
            symbol = symbol.replace("]", f";{H_symbol}]")

    if ":" in symbol:
        symbol = symbol.replace(":", f";D{atom.GetDegree()}:")
    else:
        symbol = symbol.replace("]", f";D{atom.GetDegree()}]")

    if "+" not in symbol and "-" not in symbol:
        charge = atom.GetFormalCharge()
        charge_symbol = "+" if (charge >= 0) else "-"
        charge_symbol += f"{abs(charge)}"
        if ":" in symbol:
            symbol = symbol.replace(":", f";{charge_symbol}:")
        else:
            symbol = symbol.replace("]", f";{charge_symbol}]")

    return symbol

def expand_changed_atom_tags(changed_atom_tags: list[str], reactant_fragment: str) -> list[str]:
    expansion: list[str] = []
    atom_tags_in_reactant_fragments = re.findall(r"\:([0-9]+)\]", reactant_fragment)
    for atom_tag in atom_tags_in_reactant_fragments:
        if atom_tag not in changed_atom_tags:
            expansion.append(atom_tag)
    print(f"After building reactant fragments, additional labels include {expansion}.")
    return expansion

def get_fragments_for_changed_atoms(molecules: list[Chem.Mol], changed_atom_tags: list[str], radius: int = 0, category: str = "reactants", expansion: list[str] = []) -> str:
    fragments: str = ""
    mol_changed = []
    for molecule in molecules:
        symbol_replacements: list[tuple[int, str]] = []

        if category == "reactants":
            groups = get_special_groups(molecule)
        elif category == "products":
            groups: list[tuple[list[int], list[int]]] = []
        else:
            raise ValueError("")

        atoms_to_use: list[int] = []
        for atom in molecule.GetAtoms():
            if ":" in atom.GetSmarts():
                if atom.GetSmarts().split(":")[1][:-1] in changed_atom_tags:
                    atoms_to_use.append(atom.GetIdx())
                    symbol = get_strict_smarts_for_atom(atom)
                    if symbol != atom.GetSmarts():
                        symbol_replacements.append((atom.GetIdx(), symbol))

        if len(atoms_to_use) > 0:
            if category == "reactants":
                for atom in molecule.GetAtoms():
                    if not atom.HasProp("molAtomMapNumber"):
                        atoms_to_use.append(atom.GetIdx())

        for k in range(radius):
            atoms_to_use, symbol_replacements = expand_atoms_to_use(molecule, atoms_to_use, groups, symbol_replacements)

        if category == "products":
            if expansion:
                for atom in molecule.GetAtoms():
                    if ":" not in atom.GetSmarts():
                        continue
                    label = atom.GetSmarts().split(":")[1][:-1]
                    if label in expansion and label not in changed_atom_tags:
                        atoms_to_use.append(atom.GetIdx())
                        symbol_replacements.append((atom.GetIdx(), convert_atom_to_wildcard(atom)))
                        print(f"Expanded label {label} to wildcard in products")

            for atom in molecule.GetAtoms():
                if not atom.HasProp("molAtomMapNumber"):
                    atoms_to_use.append(atom.GetIdx())
                    symbol = get_strict_smarts_for_atom(atom)
                    symbol_replacements.append((atom.GetIdx(), symbol))

        symbols = [atom.GetSmarts() for atom in molecule.GetAtoms()]
        for i, symbol in symbol_replacements:
            symbols[i] = symbol

        if not atoms_to_use:
            continue

        tetra_consistent = False
        num_tetra_flip   = 0
        while not tetra_consistent and num_tetra_flip < 100:
            molecule_copied           = deepcopy(molecule)
            [atom.ClearProp("molAtomMapNumber") for atom in molecule_copied]
            this_fragments            = Chem.MolFragmentToSmiles(molecule_copied, atoms_to_use, atomSymbols=symbols, allHsExplicit=True, isomericSmiles=True, allBondsExplicit=True)
            this_fragment_molecule    = Chem.MolFromSmarts(this_fragments)
            tetra_map_nums: list[str] = []
            for atom in this_fragment_molecule.GetAtoms():
                if atom.HasProp("molAtomMapNumber"):
                    atom.SetIsotope(int(atom.GetProp("molAtomMapNumber")))
                    if atom.GetChiralTag() != Chem.CHI_UNSPECIFIED:
                        tetra_map_nums.append(atom.GetProp("molAtomMapNumber"))

            map_to_id: dict[str, int] = {}
            for atom in molecule.GetAtoms():
                if atom.HasProp("molAtomMapNumber"):
                    atom.SetIsotope(int(atom.GetProp("molAtomMapNumber")))
                    map_to_id[atom.GetProp("molAtomMapNumber")] = atom.GetIdx()

            tetra_consistent = True
            all_matched_ids: list[int] = []

            fragment_smiles = Chem.MolToSmiles(this_fragment_molecule)
            if fragment_smiles.count(".") > 5:
                break

            for match_ids in molecule.GetSubstructMatches(this_fragment_molecule, useChirality=True):
                all_matched_ids.extend(match_ids)
            shuffle(tetra_map_nums)
            for tetra_map_num in tetra_map_nums:
                print(f"Checking consistency of tetrahedral {tetra_map_num}")
                if map_to_id[tetra_map_num] not in all_matched_ids:
                    tetra_consistent = False
                    print("@@@@@@@@@@@ FRAGMENT DOES NOT MATCH PARENT MOL @@@@@@@@@@")
                    print("@@@@@@@@@@@ FLIPPING CHIRALITY SYMBOL NOW      @@@@@@@@@@")
                    prev_symbol = symbols[map_to_id[tetra_map_num]]
                    if "@@" in prev_symbol:
                        symbol = prev_symbol.replace("@@", "@")
                    elif "@" in prev_symbol:
                        symbol = prev_symbol.replace("@", "@@")
                    else:
                        raise ValueError("Need to modified symbol of tetra atom without @ or @@???")
                    symbols[map_to_id[tetra_map_num]] = symbol
                    num_tetra_flip += 1
                    break

            for atom in molecule.GetAtoms():
                atom.SetIsotope(0)

        if not tetra_consistent:
            raise ValueError(f"Could not find tetra consistent tetrahedral mapping, {len(tetra_map_nums)} centers.")

        fragments += f"({this_fragments})."
        mol_changed.append(Chem.MolToSmiles(clear_mapnumber(Chem.MolFromSmiles(Chem.MolToSmiles(molecule, True))), True))

    return fragments[:-1]

def canonicalize_template(template: str) -> str:
    template_nolabels           = re.sub(r"\:[0-9]+\]", r"]", template)
    template_nolabels_molecules = template_nolabels[1:-1].split(").(")
    template_molecules          = template[1:-1].split(").(")
    for i in range(len(template_molecules)):
        nolabel_molecular_fragments    = template_nolabels_molecules[i].split(".")
        molecular_fragment             = template_molecules[i].split(".")
        sortorder                      = [j[0] for j in sorted(enumerate(nolabel_molecular_fragments), key = lambda x:x[1])]
        template_nolabels_molecules[i] = ".".join([nolabel_molecular_fragments[j] for j in sortorder])
        template_molecules[i]          = ".".join([molecular_fragment[j] for j in sortorder])

    sortorder = [j[0] for j in sorted(enumerate(template_nolabels_molecules), key = lambda x:x[1])]
    template  = f"({").(".join([template_molecules[i] for i in sortorder])})"
    return template

def canonicalize_transform(transform: str) -> str:
    transform_reordered = ">>".join([canonicalize_template(x) for x in transform.split(">>")])
    return reassign_atom_mapping(transform_reordered)