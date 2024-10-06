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
