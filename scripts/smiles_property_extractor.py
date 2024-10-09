import numpy as np
import pandas as pd
import jazzy.api as jazzyAPI
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Draw


class ChemicalInfoFromSmiles:
    '''
    Class containing methods to extract chemical information from SMILES strings.
    '''

    @staticmethod
    def get_mol_from_smiles(smiles: str) -> Chem.Mol:
        '''
        Converts a SMILES string to an RDKit molecule object.
        
        Parameters:
        smiles (str): A string representing the molecule in SMILES notation.
        
        Returns:
        RDKit.Chem.rdchem.Mol: A molecule object corresponding to
            the input SMILES string.

        Raises:
        TypeError: If the input is not a string object.
        ValueError: If the input is not a valid Chem.rdchem.Mol object.
        '''
        if not isinstance(smiles, str):
            raise TypeError("Input must be a string object")
        mol = Chem.MolFromSmiles(smiles)

        if not mol:
            raise ValueError("Invalid molecule structure")
        return mol

    @staticmethod
    def is_aromatic(smiles: str) -> bool:
        '''
        Determines if the given molecule contains any aromatic atoms.
        
        Parameters:
        smiles (str): A string representing the molecule in SMILES notation.
        
        Returns:
        bool: True if any atom in the molecule is aromatic, False otherwise.
        '''
        mol = ChemicalInfoFromSmiles.get_mol_from_smiles(smiles)
        return any(atom.GetIsAromatic() for atom in mol.GetAtoms())

    @staticmethod
    def is_cyclic(smiles: str):
        '''
        Determines if the given molecule contains any ring atoms.
        
        Parameters:
        smiles (str): A string representing the molecule in SMILES notation.
        
        Returns:
        bool: True if the molecule contains a ring, False otherwise.
        '''
        mol = ChemicalInfoFromSmiles.get_mol_from_smiles(smiles)
        return mol.GetRingInfo().NumRings() > 0

    @staticmethod
    def is_neutral(smiles: str) -> bool:
        '''
        Checks if the given molecule is charged neutral or not.

        Parameters:
        smiles (str): A string representing the molecule in SMILES notation.
        
        Returns:
        bool: True if total charge of the molecule is zero, False otherwise.
        '''
        mol = ChemicalInfoFromSmiles.get_mol_from_smiles(smiles)
        total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        return total_charge == 0

    @staticmethod
    def is_composed_of_specific_elements(smiles: str) -> bool:
        '''
        Determines if the molecule represented by a SMILES string
        is exclusively composed of specified elements:
        Br, I, Cl, F, H, C, N, O, S, P.

        Parameters:
        smiles (str): A string representing the molecule in SMILES notation.

        Returns:
        bool: True if the molecule is composed only of the specified elements, False otherwise.
        '''
        key_elements = {'Br', 'I', 'Cl', 'F', 'H', 'C', 'N', 'O', 'S', 'P'}
        mol = ChemicalInfoFromSmiles.get_mol_from_smiles(smiles)
        mol_elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
        
        return mol_elements.issubset(key_elements)      

    @staticmethod
    def get_pngs_from_simles(dataframe: pd.DataFrame, dirname : str = 'images'):
        '''
        Converts SMILES strings from a DataFrame into PNG images
        and saves them in the specified directory.
        
        Parameters:
        dataframe (pd.DataFrame): DataFrame containing the "SMILES" column.
        dirname (str): Directory name where the images will be saved. Defaults to 'images'.
        
        Raises:
        TypeError: If the input is not a pandas DataFrame object.
        ValueError: If the DataFrame does not contain a "SMILES" column.
        '''
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame object")

        if 'SMILES' not in dataframe.columns:
            raise ValueError("DataFrame does not have a 'SMILES' column")

        dirpath = Path(dirname)
        if not dirpath.exists():
            dirpath.mkdir(parents=True, exist_ok=True)
            print(f'Directory {dirname} just created!')

        for idx, smiles in enumerate(dataframe['SMILES']):
            mol = ChemicalInfoFromSmiles.get_mol_from_smiles(smiles)
            img = Draw.MolToImage(mol)

            img_path = Path(f'{dirname}/mol-{idx}.png')
            with img_path.open('wb') as imgfile:
                img.save(imgfile)

    @staticmethod
    def get_rdkit_2dDescriptors_from_smiles(smiles: str, missingVal=None) -> dict:
        '''
        Calculate 2D molecular descriptors for a given SMILES string using RDKit.

        Parameters:
        smiles (str): A SMILES string representing the molecule.
        missingVal (optional): The value to be assigned to a descriptor
            if its calculation results in a RuntimeError. Defaults to None.

        Returns:
        dict: A dictionary where keys are descriptor names and values are
            the calculated descriptor values.
        '''
        mol = ChemicalInfoFromSmiles.get_mol_from_smiles(smiles)
        
        result = {}
        for descriptor_name, descriptor_object in Descriptors._descList:
            try:
                val = descriptor_object(mol)
            except RuntimeError:
                val = missingVal
                
            result[descriptor_name] = val
        return result

    @staticmethod
    def generate_3D_coordinates_from_smiles(smiles: str) -> list:
        '''
        Generate 3D coordinates for a molecule from SMILES representation.
    
        Parameters:
        smiles (str): A SMILES string representing the molecule.
    
        Returns
        list: A list of tuples, each containing the atomic symbol
            and a Point3D object representing the XYZ coordinates of
            each atom in the molecule.
        '''
        mol = ChemicalInfoFromSmiles.get_mol_from_smiles(smiles)    
        mol = Chem.AddHs(mol)

        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)

        conf = mol.GetConformer()
        xyz_coordinates = [
            (atom.GetSymbol(), conf.GetAtomPosition(atom.GetIdx()))
            for atom in mol.GetAtoms()
        ]

        return xyz_coordinates

    @staticmethod
    def get_atomic_map_from_smiles_jazzy(smiles: str) -> any:
        '''
        Generates an atomic map from a given SMILES string
            using the MMFF94 minimization method.

        Parameters:
        smiles (str): A SMILES string representing the molecule.

        Returns:
        list[dict]: A list where each element is a dictionary containing
            information about an atom in the molecule.
            In the case of errors returns "None".
        '''
        try:
            return jazzyAPI.atomic_map_from_smiles(smiles,
                                              minimisation_method="MMFF94")
        except:
            print(f'An error occured for {smiles}')
            return None

    @staticmethod
    def get_molecular_vector_from_smiles_jazzy(smiles: str) -> any:
        '''
        Generates an atomic map from a given SMILES string
            using the MMFF94 minimization method.

        Parameters:
        smiles (str): A SMILES string representing the molecule.

        Returns:
        list[dict]: A list where each element is a dictionary containing
            information about an atom in the molecule.
            In the case of errors returns "None".
        '''
        try:
            return jazzyAPI.molecular_vector_from_smiles(smiles)
        except:
            print(f'An error occured for {smiles}')
            return None

    @staticmethod
    def get_atomic_masses_from_smiles(smiles: str) -> np.ndarray:
        '''
        Extracts atomic masses from a SMILES string.

        Parameters:
        - smiles (str): A SMILES string representing the molecule.

        Returns:
        - list[float]: A list of atomic masses (float numbers).
        '''
        mol = ChemicalInfoFromSmiles.get_mol_from_smiles(smiles)
        mol = Chem.AddHs(mol)
        atoms = mol.GetAtoms()
        periodic_table = Chem.GetPeriodicTable()
        atomic_masses = np.array([periodic_table.GetAtomicWeight(
                            atom.GetAtomicNum()) for atom in atoms])
        return atomic_masses

    @staticmethod
    def get_center_of_mass_coordinates(smiles: str) -> np.ndarray:
        '''
        Calculates the center of mass coordinates for
            a molecule given its SMILES string.

        Parameters:
        - smiles (str): A SMILES string representing the molecule.

        Return:
        - np.ndarray: An array with 3 elements (float numbers).
        '''
        atomic_coordinates = ChemicalInfoFromSmiles.generate_3D_coordinates_from_smiles(smiles)
        coords = np.array([coord[1] for coord in atomic_coordinates])
        atomic_masses = ChemicalInfoFromSmiles.get_atomic_masses_from_smiles(smiles)

        total_mass = np.sum(atomic_masses)
        if total_mass == 0:
            raise ValueError("Total atomic mass is zero, cannot compute center of mass.")
        weighted_coords = coords * atomic_masses[:, np.newaxis]
        
        return np.sum(weighted_coords, axis=0) / total_mass
        
    @staticmethod
    def transform_coords_to_center_of_mass(smiles: str) -> np.ndarray:
        '''
        Transforms atomic coordinates to be relative to the center of mass of the molecule.

        Parameters:
        - smiles (str): A SMILES string representing the molecule.

        Returns:
        - A 2D array of transformed atomic coordinates, where each row corresponds to
            an atom and the columns correspond to x, y, and z coordinates.

        Raises:
        - ValueError: If the SMILES string is invalid or the atomic coordinates cannot be generated.
        '''
        try:
            atomic_coordinates = ChemicalInfoFromSmiles.generate_3D_coordinates_from_smiles(smiles)
            coords = np.array([coord[1] for coord in atomic_coordinates])

            center_of_mass = ChemicalInfoFromSmiles.get_center_of_mass_coordinates(smiles)
            transformed_coords = coords - center_of_mass

            return transformed_coords

        except Exception as e:
            raise ValueError(f"Failed to transform coordinates for SMILES string '{smiles}': {e}")

    @staticmethod
    def get_yukawa_potential_from_jazzy(smiles: str,
                                        atomic_property: str = 'eeq',
                                        beta: int = 0.01) -> float:
        '''
        Computes the Yukawa potential for a given atomic property from a SMILES string
            using Jazzy's outputs and distances between atomic coordinates.
        
        Parameters:
        - smiles (str): A SMILES string representing a chemical compound.
        - atomic_property (str): The atomic property to be used in the calculation (default is 'eeq').
        - beta (float): A parameter controlling the exponential decay (default is 0.01).

        Returns:
        - float: The calculated Yukawa potential value.

        Raises:
        - ValueError: If the SMILES string cannot be processed.
        '''
        try:
            jazzy_output = ChemicalInfoFromSmiles.get_atomic_map_from_smiles_jazzy(smiles)
            atomic_coords = ChemicalInfoFromSmiles.transform_coords_to_center_of_mass(smiles)
        except Exception as e:
            raise ValueError(f"Error processing SMILES string: {e}")
        
        yukawa_potential = 0.0
        for i, coord_i in enumerate(atomic_coords):
            feature_i = jazzy_output[i].get(atomic_property, 0.0)

            for j in range(i + 1, len(atomic_coords)):
                feature_j = jazzy_output[j].get(atomic_property, 0.0)
                distance = np.linalg.norm(coord_i - atomic_coords[j])
                yukawa_potential += feature_i * feature_j * np.exp(-distance * beta)
        return yukawa_potential

class Fingerprint:
    '''
    Class containing methods to generate Extended Connectivity Fingerprint "ECFP" from SMILES strings.
    '''
    def __init__(self,
                 data: pd.DataFrame,
                 data_path: str,
                 is_numbering_fragments: bool = False):
        assert isinstance(data, pd.DataFrame), "data must be a pandas DataFrame"
        assert 'SMILES' in data.columns, "data must contain a 'SMILES' column"
        self.data = data
        self.data_path = data_path
        self.is_numbering_fragments = is_numbering_fragments

    def generate_ecfp(self,
                      nbits: int = 1024,
                      diameter: int = 4):
        self.fingerprints = np.zeros((self.data.shape[0], nbits))
        smiles = self.data['SMILES'].to_numpy()
        submol_smiles = [[] for _ in range(nbits)]

        for i, smi in enumerate(tqdm(enumerate(smiles), total=len(smiles), desc="Processing fingerprints")):
            mol = Chem.MolFromSmiles(smi[1])
            bitinfo = {}
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, diameter // 2, nBits=nbits, bitInfo=bitinfo, useChirality=True
            )
            self.fingerprints[i] = np.array(fp)
            self._process_bitinfo(i, mol, bitinfo, submol_smiles)

        self.subsmiles = self._extract_unique_subsmiles(submol_smiles, nbits)
        self._save_results()

    def _process_bitinfo(self, indx, mol, bitinfo, submol_smiles):
        for bit, info in bitinfo.items():
            atom, radius = info[0]
            if self.is_numbering_fragments:
                freq = len(bitinfo[bit])
                self.fingerprints[indx][bit] = freq

            atoms, env = self._get_atom_environment(mol, atom, radius)
            frag_smiles = self._get_fragment_smiles(mol, atoms, env, atom)
            submol_smiles[bit].append(frag_smiles)

    @staticmethod
    def _get_atom_environment(mol, atom, radius):
        if radius > 0:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom)
            atoms = set(
                mol.GetBondWithIdx(b).GetBeginAtomIdx() for b in env
            ).union(
                mol.GetBondWithIdx(b).GetEndAtomIdx() for b in env
            )
        else:
            atoms, env = [atom], None
        return list(atoms), env

    @staticmethod
    def _get_fragment_smiles(mol, atoms, env, atom):
        return Chem.MolFragmentToSmiles(
            mol, atoms, bondsToUse=env, allHsExplicit=True,
            allBondsExplicit=True, rootedAtAtom=atom, isomericSmiles=False
        )

    @staticmethod
    def _extract_unique_subsmiles(submol_smiles, nbits):
        subsmiles = ['NaN'] * nbits
        for i in range(nbits):
            unique_subsmiles, pos_subsmiles = np.unique(
                submol_smiles[i], return_inverse=True
            )
            if unique_subsmiles.size > 0:
                subsmiles[i] = unique_subsmiles[np.argmax(np.bincount(pos_subsmiles))]
        return subsmiles

    def _save_results(self):
        if self.is_numbering_fragments:
            submolecules_name = 'wecfp-submolecules.csv'
            fingerprint_name = 'wecfp-fingerprints.csv'
        else:
            submolecules_name = 'submolecules'
            fingerprint_name = 'fingerprints.csv'
            
        with open(f'{self.data_path}/{submolecules_name}', 'w') as f:
            pd.DataFrame(
                {'Bit': np.arange(len(self.subsmiles)), 'SMILES': self.subsmiles}
            ).to_csv(f, index=False)

        with open(f'{self.data_path}/{fingerprint_name}', 'w') as f:
            pd.DataFrame(self.fingerprints).to_csv(f, index=False)

