from pathlib import Path
import os
import numpy as np
import pandas as pd
import scipy as sp
import timeit
from pymatgen.io.cif import CifParser
from pymatgen.core import Composition
# Bond Features
from matminer.featurizers.site import GaussianSymmFunc, SiteElementalProperty,AGNIFingerprints

# Structure Features
from matminer.featurizers.structure.bonding import GlobalInstabilityIndex, StructuralHeterogeneity
from matminer.featurizers.structure.composite import JarvisCFID
from matminer.featurizers.structure.order import StructuralComplexity, MaximumPackingEfficiency, DensityFeatures
from matminer.featurizers.structure.misc import StructureComposition
from matminer.featurizers.composition.element import ElementFraction
from matminer.featurizers.conversions import CompositionToOxidComposition
from matminer.featurizers.site.chemical import EwaldSiteEnergy 

BOND_MAX_DIST = 2.9  # Max distance for a bond in angstroms
POLYHEDRA_BOND_DIST = 2.35 # TODO: CHECK THIS


def featurize_bonds(cifs: list, verbose=False, saveto: str = "features.csv") -> pd.DataFrame:
    """Featurize crystal structures using elemetal, geometric, and chemical descriptors for local environments.

    :params cifs: list of paths to crystal structure in cif format
    :params verbos: prints each step of the processing
    :params saveto: filename to save the generated features
    """
    
    ## Process Input Files
    if verbose: print("Parsing CIFs")
    features = {}
    for cif in cifs:
        structure = CifParser(cif).get_structures()[0]
        structure_name = Path(cif).name
        features[structure_name] = {}
        features[structure_name]["structure"] = structure
        features[structure_name]["structure_name"] = Path(cif).name
        features[structure_name]["structure_path"] = str(Path(cif).parent)
    data = pd.DataFrame.from_dict(features).T
    
    ### SITE PROPERTIES ###
    # These will be paired as features
    ## 1. Initialize the dictionary for each site
    if verbose: print("Assembling site property dictionary")
    site_features = {}
    for index, row in data.iterrows():
        structure = row["structure"]
        for atomidx in range(structure.num_sites):
            site_name = "%s_%i" % (index, atomidx)
            site_features[site_name] = {}
            site_features[site_name] = {"structure_name": row["structure_name"]}
            site_features[site_name].update({"structure_path": row["structure_path"]})

    ## Loop through Site featurizers
    property_list = ("Number", "AtomicWeight", "Electronegativity", "CovalentRadius")  # For SiteElementalProperty function
    site_feature_functions = [SiteElementalProperty(properties=property_list), AGNIFingerprints(cutoff=5, directions=[None]), GaussianSymmFunc(cutoff=5)]
    # Wishlist: EwaldSiteEnergy(), CompositionToOxidComposition, (Row, column?)
    for featurizer in site_feature_functions:
        if verbose: print("Using: ", featurizer)
        colnames = featurizer._generate_column_labels(multiindex=False, return_errors=False)
        for index, row in data.iterrows():
            structure = row["structure"]
            if verbose: print(index)
            for atomidx in range(structure.num_sites):
                feat = featurizer.featurize(structure, idx=atomidx)
                site_name = "%s_%i" % (index, atomidx)
                site_features[site_name].update(dict(zip(colnames, feat)))
 
    ### BOND PAIRS AND BOND PROPERTIES ###
    if verbose: print("Generating bond library")
    structures_bonds = {}  # Store bond pairs
    bond_properties = {}  # Store bond properties
    for index, row in data.iterrows():
        if verbose: print(index)
        structure = row["structure"]
        structures_bonds[index] = []
        bond_properties[index] = []
        neighbors = structure.get_neighbor_list(BOND_MAX_DIST)  # (center_indices, points_indices, offset_vectors, distances)
        for bond in range(len(neighbors[0])):
            if neighbors[0][bond] < neighbors[1][bond]:  # Don't double count bonds
                # Bonded indices
                structures_bonds[index].append((neighbors[0][bond], neighbors[1][bond]))
                # Bond properties (coord-num, bond-len)
                coord_num = list(neighbors[0]).count(neighbors[0][bond])
                bond_properties[index].append((coord_num, neighbors[3][bond]))

    # Build Dataframe by bonds
    if verbose: print("Copying over data to final dataframe")
    delta_properties = ["site Electronegativity", "site AtomicWeight", "CovalentRadius"]  # For these properties, take the difference as a feature
    bond_features = {}  # Final dictionary for saving features format: bond_features['material_bond#']["feature_name"] = data
    for index, row in data.iterrows():
        bond_len_sum = 0
        if verbose: print(index)
        for bond_idx in range(len(structures_bonds[index])):
            bond = structures_bonds[index][bond_idx]
            bond_name = "%s_Atom%i_Bond%i" % (index, bond[0], bond_idx)
            bond_features[bond_name] = {}
            site1_name = "%s_%i" % (index, bond[0])
            site2_name = "%s_%i" % (index, bond[1])
            
            # Add Site features to dictionary
            # Order putting heavier element first
            # TODO: this works but is not very efficient, save data directly to final dataframe in the end?
            site_feat_labels = site_features[site1_name].keys()
            site_feat_labels = [k for k in site_feat_labels if k not in ["structure_path", "structure_name"]]
            bond_features[bond_name]["structure_name"] = site_features[site1_name]["structure_name"]
            bond_features[bond_name]["structure_path"] = site_features[site1_name]["structure_path"]
            if site_features[site1_name]["site AtomicWeight"] > site_features[site2_name]["site AtomicWeight"]:
                for k in site_feat_labels:
                    if k in delta_properties:
                        bond_features[bond_name][k+"_diff"] = site_features[site1_name][k] - site_features[site2_name][k]
                    bond_features[bond_name][k+"_atom1"] = site_features[site1_name][k]
                    bond_features[bond_name][k+"_atom2"] = site_features[site2_name][k]
            else:
                for k in site_feat_labels:
                    if k in delta_properties:
                        bond_features[bond_name][k+"_diff"] = site_features[site2_name][k] - site_features[site1_name][k]
                    bond_features[bond_name][k+"_atom1"] = site_features[site2_name][k]
                    bond_features[bond_name][k+"_atom2"] = site_features[site1_name][k]
                    
            # Insert bond properties        
            coord_num, bond_len = bond_properties[index][bond_idx]
            bond_features[bond_name]["coordination_number"] = coord_num
            bond_features[bond_name]["bond_length"] = bond_len
            bond_len_sum += bond_len  # TODO: There's a bug somewhere around here
            
        # Now add each bond's fraction of lattice volume
        for bond_idx in range(len(structures_bonds[index])):
            bond = structures_bonds[index][bond_idx]
            bond_name = "%s_Atom%i_Bond%i" % (index, bond[0], bond_idx)
            _, bond_len = bond_properties[index][bond_idx]
            bond_features[bond_name]["volume_fraction"] = bond_len/bond_len_sum
    
    ### SAVE FILE
    bond_feat_df = pd.DataFrame.from_dict(bond_features).T
    if os.path.isfile(saveto+"_bond.csv"):  # Append
        bond_feat_df.to_csv(saveto+"_bond.csv", mode='a', header=False)
    else:  # New file
        bond_feat_df.to_csv(saveto+"_bond.csv")
        
    return bond_feat_df

def featurize_structure(cifs: list, verbose=False, saveto: str = "features.csv") -> pd.DataFrame:
    # Note: limiting function is MaximumPackingEfficiency()
    
    ## Process Input Files
    if verbose: print("Parsing CIFs")
    features = {}
    for cif in cifs:
        structure = CifParser(cif).get_structures()[0]
        structure_name = Path(cif).name
        features[structure_name] = {}
        features[structure_name]["structure"] = structure
        features[structure_name]["structure_name"] = Path(cif).name
        features[structure_name]["structure_path"] = str(Path(cif).parent)
    data = pd.DataFrame.from_dict(features).T
    
    ### STRUCTURE PROPERTIES ###
    ## 1. Initialize the dictionary for each site
    if verbose: print("Assembling Structure property dictionary")
    structure_features = {}
    for index, row in data.iterrows():
        structure = row["structure"]
        structure_features[index] = {}
        structure_features[index] = {"structure_name": row["structure_name"]}
        structure_features[index].update({"structure_path": row["structure_path"]})

    ## Structure Featurizers
    structure_feature_functions = [ StructuralComplexity(), JarvisCFID(use_chem=False, use_rdf=False, use_chg=False, use_adf=False, use_ddf=False, use_nn=False), MaximumPackingEfficiency(), DensityFeatures()]
    # Wishlist: add jarvisCFID flags, CompositionToOxidComposition(), StructuralHeterogeneity(stats=('range', 'avg_dev'))
    for index, row in data.iterrows():
        structure = row["structure"]
        for featurizer in structure_feature_functions:
            if verbose: print(featurizer)
            try:
                colnames = featurizer._generate_column_labels(multiindex=False, return_errors=False)
                feat = featurizer.featurize(structure)
                structure_features[index].update(dict(zip(colnames, feat)))
            except:
                print('Exception occured with ', featurizer, " in ", row["structure_name"])
            # TODO: Structural Complexity only first entry, select certain features from others

    # Generate Elemental Metrics from Structure including #elements and entropy of mixing
    for index, row in data.iterrows():
        try:
            structure = row["structure"]
            featurizer = ElementFraction()
            # Convert structure object to Composition object
            composition_from_structure = Composition({element: count for element, count in structure.composition.items()})
            mole_frac_list = featurizer.featurize(composition_from_structure)
            # Calculate features from composition
            mole_frac_list = [i for i in mole_frac_list if i != 0] # Remove all zero entries for elements
            feat = [len(mole_frac_list), sum([-i*np.log(i) for i in mole_frac_list])]  # [num_elements, entropy_of_mixing]
            colnames = ["number of elements", "entropy of mixing"]
            structure_features[index].update(dict(zip(colnames, feat)))
        except:
            print('Exception occured with ', featurizer, " in ", row["structure_name"])
   
    # POLYHEDRA FEATURIZING
    # 1. make connectivity sparse matrix
    # 2. num of polyhedra: number of non-zero elements in a row (how many connections)
    # 3. num of shared points in each polyhedra: for each pair of polyhedra find logic AND of connectivity
    # 4. Classify and generate features
    for index, row in data.iterrows(): # For each structure
        structure = row["structure"]
        n_atoms = len(structure)
        
        # Generate adjacency matrix
        if n_atoms < 750: # Make extra big if cell risks double counting bonds
            structure.make_supercell(2, to_unit_cell=False) 
            n_atoms = len(structure)
        neighbors = structure.get_neighbor_list(POLYHEDRA_BOND_DIST)  # (center_indices, points_indices, offset_vectors, distances)
        adjacency_matrix = sp.sparse.csr_matrix((np.ones(len(neighbors[0])), (neighbors[0], neighbors[1])), shape = (n_atoms, n_atoms)).toarray()
        if verbose:
            print("POLYHEDRA FEATURIZING")
            print(adjacency_matrix)
            
        # Check supercell is large enough for algorithm to work, if first supercell not big enough
        if np.sum(adjacency_matrix>1)>0:
            print("Creating a supercell for polyhedra featurizing: ", row["structure_name"]," not compatible with polyhedra featurizing algorithm")
            structure.make_supercell(2, to_unit_cell=False)
            n_atoms = len(structure)
            neighbors = structure.get_neighbor_list(POLYHEDRA_BOND_DIST)  # (center_indices, points_indices, offset_vectors, distances)
            adjacency_matrix = sp.sparse.csr_matrix((np.ones(len(neighbors[0])), (neighbors[0], neighbors[1])), shape = (n_atoms, n_atoms)).toarray()
        
        if np.sum(adjacency_matrix>1)>0:  # Check again to be sure
            print("ERROR (second attempt): ", row["structure_name"]," not compatible with polyhedra featurizing algorithm")
            print("Number of atoms: ", n_atoms)
            
        
        # Pull features from adjacency matrix
        poly_features = np.zeros(8)
        for a in range(n_atoms): # for each atom TODO: only for O,F
            
            # Only calculate for for anions (of oxides or flruorides)
            atom_element =  structure[a].species.elements[0].symbol
            if atom_element != "O" and atom_element != "F": 
                continue   
            connected_poly = list(np.nonzero(adjacency_matrix[a,:])[0])
            num_poly = len(connected_poly) # How many polyhedra are connected to this atom
            if verbose: 
                print("Connected poly: ", connected_poly)
                print("Num Poly: ", num_poly)
            poly_shared_pts = []
            if num_poly >= 2:
                for p1 in range(num_poly):
                    for p2 in range(p1+1,num_poly):
                        if verbose: print(connected_poly[p1], ":", connected_poly[p2])
                        shared_pts = np.logical_and(adjacency_matrix[connected_poly[p1],:], adjacency_matrix[connected_poly[p2],:])
                        poly_shared_pts.append(len(np.nonzero(shared_pts)[0]))
            if verbose: print("Shared pts: ", poly_shared_pts)
            
            # Now Classify based on num_poly and poly_shared_pts (Zhang et al, 2023)
            poly_feat = 0
            if num_poly <= 1:
                poly_feat = 8 # C8
            elif num_poly == 2:
                if sum(poly_shared_pts) == 1:
                    poly_feat = 3 # C3
                elif sum(poly_shared_pts) == 2:
                    poly_feat = 2 # C2
                else: # poly share 3 or more pts
                    poly_feat = 1 # C1
            elif num_poly == 3:
                if sum(poly_shared_pts) == 3:
                    poly_feat = 4 # C4
                elif sum(poly_shared_pts) == 4:
                    poly_feat = 6 # C6
                else: # poly share in total 5 or more pts
                    poly_feat = 7 # C7
            else: # num_poly >= 4
                if sum(poly_shared_pts) == 4:
                    poly_feat = 5 # C5
                else: # poly share 3 or more pts
                    poly_feat = 7 # C7
            if verbose: print("Poly feat: ", poly_feat)
            poly_features[poly_feat-1] += 1
        n_anions = sum(poly_features)
        if n_anions != 0:
            poly_features =  [i/n_anions for i in poly_features]  # turn into fractions of each type
            poly_features.append(sum(poly_features[0:2])+sum(poly_features[5:7])) # Low DOF polyhedra
            poly_features.append(sum(poly_features[2:5])+poly_features[7]) # High DOF polyhedra
        else:
            poly_features = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1]  # no bonds found, assume all are high-dof unabridged
        colnames = ["C1 polyhedra frac", "C2 polyhedra frac","C3 polyhedra frac","C4 polyhedra frac","C5 polyhedra frac","C6 polyhedra frac","C7 polyhedra frac","C8 polyhedra frac", "Low DOF polyhedra frac", "High DOF polyhedra frac"]
        structure_features[index].update(dict(zip(colnames, poly_features)))
            
    # Compile all features and append to save file
    structure_feat_df = pd.DataFrame.from_dict(structure_features).T
    if os.path.isfile(saveto+"_structure.csv"):  # Append
        structure_feat_df.to_csv(saveto+"_structure.csv", mode='a', header=False)
    else:  # New file
        structure_feat_df.to_csv(saveto+"_structure.csv")
        
    return structure_feat_df

def remove_files(filename):
    if os.path.isfile(filename+"_bond.csv"):  # Clean up any previous runs
        os.remove(filename+"_bond.csv")
    if os.path.isfile(filename+"_structure.csv"):  # Clean up any previous runs
        os.remove(filename+"_structure.csv")
        
        
# Batching files to reduce memory use
BATCH_SIZE = 5

# Load all CIF files in directory
file_type = "_super.cif"  # Use files with this ending in input_dir
input_dir = "supercells_data/"  # Input data directory
output_dir = "features/"  # Output directory
filename = "features_extra"  # Output filename for features, no file extension

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
elif os.path.isdir(output_dir+filename):
    os.remove(output_dir+filename)  # Remove existing file

files = os.listdir(input_dir)
cif_files = [input_dir+file for file in files if file.endswith(file_type)]

# Featurize all structures
n_batches = int(np.ceil(len(cif_files)/BATCH_SIZE))
# Remove previous output files
remove_files(output_dir+filename)
    
# Solve in batches to limit memory use
print("{} Files Total: ".format(len(cif_files)))
print("{} Batches Total: ".format(n_batches))
for b in range(n_batches):
    print("Starting batch ", b)
    # Define which files are in each batch
    idx_start = int(b*BATCH_SIZE)
    idx_end = int(min((b+1)*BATCH_SIZE, len(cif_files)))
    start = timeit.default_timer()
    bond_df = featurize_bonds(cif_files[idx_start:idx_end], saveto=output_dir+filename, verbose=False)
    struc_df = featurize_structure(cif_files[idx_start:idx_end], saveto=output_dir+filename, verbose=False)
    print("Time elapsed: ", timeit.default_timer() - start)

print("Files processed: ", len(cif_files))
# On 1 Sherlock core, ~2.25 min per batch -> 4hr 20 min
