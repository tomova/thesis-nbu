import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import rdPartialCharges

# Function to calculate dipole moment
def calculate_dipole_moment(mol):
    conf = mol.GetConformer()
    dipole = [0.0, 0.0, 0.0]
    for atom in mol.GetAtoms():
        charge = float(atom.GetProp('_GasteigerCharge'))
        position = conf.GetAtomPosition(atom.GetIdx())
        dipole[0] += charge * position.x
        dipole[1] += charge * position.y
        dipole[2] += charge * position.z
    return dipole

# Function to calculate quadrupole moment
def calculate_quadrupole_moment(mol):
    conf = mol.GetConformer()
    quadrupole = np.zeros((3, 3))
    for atom in mol.GetAtoms():
        charge = float(atom.GetProp('_GasteigerCharge'))
        position = conf.GetAtomPosition(atom.GetIdx())
        for i in range(3):
            for j in range(3):
                quadrupole[i, j] += charge * position[i] * position[j]
    return quadrupole

def mol_to_graph(mol):
    # Convert atoms to features 
    atom_features = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        valence = atom.GetTotalValence()
        hybridization = atom.GetHybridization()
        num_hydrogens = atom.GetTotalNumHs()
        is_in_ring = int(atom.IsInRing())
        
        atom_features.append([atomic_num, valence, hybridization, num_hydrogens, is_in_ring])

    # Get adjacency matrix
    adjacency_matrix = Chem.GetAdjacencyMatrix(mol)

    # Get atom coordinates
    conformer = mol.GetConformer()
    atom_positions = [conformer.GetAtomPosition(atom.GetIdx()) for atom in mol.GetAtoms()]
    atom_positions = [np.array([pos.x, pos.y, pos.z]) for pos in atom_positions]

    # Bond features: Bond Type
    bond_features = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=float)
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble() # 1: Single, 2: Double, etc.
        bond_features[start, end] = bond_type
        bond_features[end, start] = bond_type # Symmetric

    return np.array(atom_features), adjacency_matrix, atom_positions, bond_features


# Load molecules from SDF file
sdf_supplier = Chem.SDMolSupplier('gdb9.sdf')

# List to store results
dataset = []

# Process each molecule
for idx, mol in enumerate(sdf_supplier):
    if mol is None:
        print(f"Skipping molecule at index {idx} due to loading error.")
        continue

    try:
        # Compute Gasteiger charges
        rdPartialCharges.ComputeGasteigerCharges(mol)
    except:
        print(f"Skipping molecule at index {idx} due to sanitization error.")
        continue

    # Convert molecule to graph representation
    atom_features, adjacency_matrix, atom_positions, bond_features = mol_to_graph(mol)

    # Calculate and store dipole and quadrupole moments
    dipole = calculate_dipole_moment(mol)
    quadrupole = calculate_quadrupole_moment(mol)

    # Store graph representation and properties
    dataset.append((atom_features, atom_positions, adjacency_matrix, bond_features, dipole, quadrupole))

# Save to disk
with open('datasetQM9.pkl', 'wb') as file:
    pickle.dump(dataset, file)

# Load the data
# with open('datasetQM9.pkl', 'rb') as f:
#     data = pickle.load(f)