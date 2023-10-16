#import functions
from IPython.display import clear_output
from rdkit import Chem
from rdkit.Chem import PandasTools
from chembl_structure_pipeline import standardizer
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem import inchi as rd_inchi
import pandas as pd
import statistics as st
import warnings; warnings.simplefilter('ignore')
import csv
import math


#all errors that might appear while curating or preparing the data will be saved here
errorverbose = './data/removed_during_curation/'

#save tables
save_data = "./data/curated_data/"

def check_extention(fname, step, encode = 'latin-1'):
    """ check the file extention and convert to ROMol if necessary """

    #import data from a csv file or sdf file (include file extention)
    if step == 1:
        file = f"./data/raw_data/{fname}"
    elif step == 2:
        file = f"./data/curated_data/{fname}"
    else:
        print("please, entender a valid step.")
        return ("1, loads raw data and 2, loads duplicate removed data.")

    if file[-3:] == "sdf":       
        imported_file = PandasTools.LoadSDF(file, smilesName='SMILES', includeFingerprints=False)
        return imported_file
    
    elif file[-4:] == "xlsx":
        imported_file = pd.read_excel(file)
        return imported_file
        
    elif file[-3:] == "csv":
        imported_file = pd.read_csv(file, encoding=encode)
        return imported_file
    
    else:
        return ("file extension not supported, supported extentions are: csv, xlsx and sdf")


def metal_atomic_numbers(at):
    """ This function checks the atomic number of an atom """
    
    n = at.GetAtomicNum()
    return (n==13) or (n>=21 and n<=31) or (n>=39 and n<=50) or (n>=57 and n<=83) or (n>=89 and n<=115)

def is_metal(smile):
    """ This function checks if an atom is a metal based on its atomic number """

    mol = Chem.MolFromSmiles(smile)
    rwmol = Chem.RWMol(mol)
    rwmol.UpdatePropertyCache(strict=False)
    metal = [at.GetSymbol() for at in rwmol.GetAtoms() if metal_atomic_numbers(at)]
    return len(metal) == 1

def smiles_preparator(smile : str or list):
    """ This function prepares smiles by removing stereochemistry """

    if type(smile) == str:
        return smile.strip("@/\\")
    
    elif type(smile) == list:
        return [smile.strip("@/\\") for smile in smile]

def neutralizeRadicals(mol):
    """ This functions neutrilizes radicals """

    for a in mol.GetAtoms():
        if a.GetNumRadicalElectrons()==1 and a.GetFormalCharge()==1:
            a.SetNumRadicalElectrons(0)         
            a.SetFormalCharge(0)

def salt_remover(mol):
    """ This function removes salts, see complete list of possible salts in https://github.com/rdkit/rdkit/blob/master/Data/Salts.txt """

    salt_list = [None, "[Cl,Br,I]", "[Li,Na,K,Ca,Mg]", "[O,N]", "[H]", "[Ba]", "[Al]", "[Cu]", "[Cs]", "[Zn]", 
    "[Mn]", "Cl[Cr]Cl", "COS(=O)(=O)[O-]", "[Sb]", "[Cr]", "[Ni]", "[B]", "CCN(CC)CC", "NCCO", "O=CO", "O=S(=O)([O-])C(F)(F)F"]

    stripped = 0

    for salt in salt_list:
        remover = SaltRemover(defnData=salt)
        stripped = remover.StripMol(mol, dontRemoveEverything=True)
    
    return stripped

#remove salts
def removeSalts(data):
    wrongSmiles = []
    new_smiles = []
    indexDropList_salts = []
    for index, smile in enumerate(data['SMILES_no_stereo']):
        try:
            mol = Chem.MolFromSmiles(smile)
            remov = salt_remover(mol)
            if remov.GetNumAtoms() <= 2:
                indexDropList_salts.append(index)
            else:
                new_smiles.append(Chem.MolToSmiles(remov, kekuleSmiles=True))
        except:
            wrongSmiles.append(data.iloc[[index]])
            indexDropList_salts.append(index)

    if len( indexDropList_salts ) == 0:
        data['SMILES_no_salts'] = data['SMILES_no_stereo']
    else:
        #drop wrong smiles
        data = data.drop(indexDropList_salts, errors="ignore")
        #save removes wrong smiles
        mask = data.iloc[indexDropList_salts]
        mask.to_csv("{}invalid_smiles.csv".format(errorverbose), sep=',', header=True, index=False)
        
    data["SMILES_no_salts"] = new_smiles
    data = data.reset_index(drop = True)
    return data

#remove organometallics
def remove_metal(data):
    organometals = []
    indexDropList_org = []
    for index, smile in enumerate(data['SMILES_no_salts']):
        if is_metal(smile) == True:
            organometals.append(data.iloc[[index]])
            indexDropList_org.append(index)

    if len(indexDropList_org) == 0:
        pass
    else:
        #drop organometallics
        data = data.drop(data.index[indexDropList_org])
        #save droped organometallics
        organmetal = pd.concat(organometals)
        organmetal.to_csv("{}organometallics.csv".format(errorverbose), sep=',', header=True, index=False)

    data = data.reset_index(drop = True)
    return data

#remove mixtures
def remove_mixture(data):
    mixtureList = []
    indexDropList_mix = []
    for index, smile in enumerate (data['SMILES_no_salts']):
        for char in smile:
            if char == '.':
                mixtureList.append(data.iloc[[index]])
                indexDropList_mix.append(index)
                break
    if len(indexDropList_mix) == 0:
        pass
    else:
        #drop mixtures
        data = data.drop(data.index[indexDropList_mix])
        #save removes mixtures
        mixtures = pd.concat(mixtureList)
        mixtures.to_csv("{}mixtures.csv".format(errorverbose), sep=',', header=True, index=False)

    data = data.reset_index(drop = True)
    return data

def standardise(data):
    """
        -Standardize unknown stereochemistry (Handled by the RDKit Mol file parser)
            Fix wiggly bonds on sp3 carbons - sets atoms and bonds marked as unknown stereo to no stereo
            Fix wiggly bonds on double bonds – set double bond to crossed bond
        -Clears S Group data from the mol file
        -Kekulize the structure
        -Remove H atoms (See the page on explicit Hs for more details)
        -Normalization:
            Fix hypervalent nitro groups
            Fix KO to K+ O- and NaO to Na+ O- (Also add Li+ to this)
            Correct amides with N=COH
            Standardise sulphoxides to charge separated form
            Standardize diazonium N (atom :2 here: [*:1]-[N;X2:2]#[N;X1:3]>>[*:1]) to N+
            Ensure quaternary N is charged
            Ensure trivalent O ([*:1]=[O;X2;v3;+0:2]-[#6:3]) is charged
            Ensure trivalent S ([O:1]=[S;D2;+0:2]-[#6:3]) is charged
            Ensure halogen with no neighbors ([F,Cl,Br,I;X0;+0:1]) is charged
        -The molecule is neutralized, if possible. See the page on neutralization rules for more details.
        -Remove stereo from tartrate to simplify salt matching
        -Normalise (straighten) triple bonds and allenes

        https://github.com/chembl/ChEMBL_Structure_Pipeline
    """

    rdMol = [Chem.MolFromSmiles(smile, sanitize=True) for smile in data['SMILES_no_salts']]

    molBlock = [Chem.MolToMolBlock(mol) for mol in rdMol]

    stdMolBlock = [standardizer.standardize_molblock(mol_block) for mol_block in molBlock]

    molFromMolBlock = [Chem.MolFromMolBlock(std_molblock) for std_molblock in stdMolBlock]

    mol2smiles = [Chem.MolToSmiles(m) for m in molFromMolBlock]
    
    data['Stand_smiles'] = mol2smiles
    
    
    #remove salts second time
    wrongSmiles = []
    new_smiles = []
    indexDropList_salts = []
    for index, smile in enumerate(data['Stand_smiles']):
        try:
            mol = Chem.MolFromSmiles(smile)
            remov = salt_remover(mol)
            if remov.GetNumAtoms() <= 2:
                indexDropList_salts.append(index)
            else:
                new_smiles.append(Chem.MolToSmiles(remov, kekuleSmiles=True))

        except:
            wrongSmiles.append(data.iloc[[index]])
            indexDropList_salts.append(index)


    if len( indexDropList_salts ) == 0:
        pass
    else:
        #drop wrong smiles
        data = data.drop(indexDropList_salts, errors="ignore")
        #save removes wrong smiles
        mask = data.iloc[indexDropList_salts]
        mask.to_csv("{}invalid_smiles_afterstd_2.csv".format(errorverbose), sep=',', header=True, index=False)

        data = data.reset_index(drop = True)

    data["SMILES_salts_removed_1"] = new_smiles


    #data summary
    row = ['after wrongs smiles removed second time', len(data)]
    with open('./data/data_summary/preparation_summary.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    #remove radicals and standalone salts
    mols_noradical = []
    standAlone_salts = []
    indexDropList_salts = []
    for index, smile in enumerate(data['SMILES_salts_removed_1']):
        try:
            m = Chem.MolFromSmiles(smile, False)
            m = rd_inchi.MolToInchi(m)
            m = Chem.MolFromInchi(m)
            neutralizeRadicals(m)
            Chem.SanitizeMol(m)
            mols_noradical.append(Chem.MolToSmiles(m, False))
        except:
            indexDropList_salts.append(index)
            standAlone_salts.append(data.iloc[[index]])
    if len(standAlone_salts) == 0:
        pass
    else:
        data = data.drop(data.index[indexDropList_salts])
        salts = pd.concat(standAlone_salts)
        salts.to_csv("{}salts.csv".format(errorverbose), sep=',', header=True, index=False)
    data['removed_radicals_smile'] = mols_noradical
    data = data.reset_index(drop = True)


    #data summary
    row = ['after standalone salts removed', len(data)]
    with open('./data/data_summary/preparation_summary.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    #remove salts second time
    wrongSmiles = []
    new_smiles = []
    indexDropList_salts = []
    for index, smile in enumerate(data['removed_radicals_smile']):
        try:
            mol = Chem.MolFromSmiles(smile)
            remov = salt_remover(mol)
            if remov.GetNumAtoms() <= 2:
                indexDropList_salts.append(index)
            else:
                new_smiles.append(Chem.MolToSmiles(remov, kekuleSmiles=True))

        except:
            wrongSmiles.append(data.iloc[[index]])
            indexDropList_salts.append(index)


    if len( indexDropList_salts ) == 0:
        pass
    else:
        #drop wrong smiles
        data = data.drop(indexDropList_salts, errors="ignore")
        #save removes wrong smiles
        mask = data.iloc[indexDropList_salts]
        mask.to_csv("{}invalid_smiles_afterstd_3.csv".format(errorverbose), sep=',', header=True, index=False)
    data["SMILES_salts_removed_2"] = new_smiles
    data = data.reset_index(drop = True)


    #data summary
    row = ['after wrong smiles removed third time', len(data)]
    with open('./data/data_summary/preparation_summary.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    #remove mixture second time
    mixtureList = []
    indexDropList_mix = []
    for index, smile in enumerate (data['SMILES_salts_removed_2']):
        for char in smile:
            if char == '.':
                mixtureList.append(data.iloc[[index]])
                indexDropList_mix.append(index)
                break
                
    if len(indexDropList_mix) == 0:
        pass
        
    else:
        #drop mixtures
        data = data.drop(data.index[indexDropList_mix])        
        #save removes mixtures
        mixtures = pd.concat(mixtureList)
        mixtures.to_csv("{}mixture_afterstd_2.csv".format(errorverbose), sep=',', header=True, index=False)
        data = data.reset_index()
        data = data.drop(columns = 'index')

    #data summary
    row = ['after mixtures removed second time', len(data)]
    with open('./data/data_summary/preparation_summary.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    #final std
    rdMol = [Chem.MolFromSmiles(smile, sanitize=True) for smile in data['SMILES_salts_removed_2']]

    molBlock = [Chem.MolToMolBlock(mol) for mol in rdMol]

    stdMolBlock = [standardizer.standardize_molblock(mol_block) for mol_block in molBlock]

    molFromMolBlock = [Chem.MolFromMolBlock(std_molblock) for std_molblock in stdMolBlock]

    mol2smiles = [Chem.MolToSmiles(m) for m in molFromMolBlock]
    
    #remove unwanted columns
    dropList = ['SMILES', 'SMILES_no_stereo', 'SMILES_no_salts', 'Stand_smiles', 'SMILES_salts_removed_1', 'removed_radicals_smile', 'SMILES_salts_removed_2']
    data = data.drop(columns = dropList)
    data['SMILES'] = mol2smiles
    data = data.reset_index(drop = True)
    return data


def curate(data, save_data):

    print("preparing smiles...")
    smiles = [smiles_preparator(str(smile)) for smile in data['SMILES']]
    data['SMILES_no_stereo'] = smiles
    clear_output(wait=True)

    print("removing salts...")
    data = removeSalts(data)
    clear_output(wait=True)

    #data summary
    row = ['after wrong smiles removed', len(data)]
    with open('./data/data_summary/preparation_summary.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    print("removing organometallics...")
    data = remove_metal(data)
    clear_output(wait=True)

    #data summary
    row = ['after organometalics removed', len(data)]
    with open('./data/data_summary/preparation_summary.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    print("removing mixtures...")
    data = remove_mixture(data)
    clear_output(wait=True)

    #data summary
    row = ['after mixtures removed', len(data)]
    with open('./data/data_summary/preparation_summary.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    print("standardising...")
    data = standardise(data)
    clear_output(wait=True)

    data.to_csv(f'{save_data}standardised_but_no_duplicates_removed.csv', header=True, index=False)
    print('Done')
    return data

def group (dataset, aggregate_column):
    """ groups duplicated values based on """

    inchikey = [Chem.MolToInchiKey(Chem.MolFromSmiles(smile)) for smile in dataset['SMILES']]
    dataset['InchiKey'] = inchikey

    groupby_column = 'InchiKey'

    dfs = []
    for aggregate_column in aggregate_column:
        dfs.append(dataset.groupby(groupby_column).aggregate({aggregate_column: list}))

    if len(dfs) == 1:
        return dfs[0].reset_index()
    elif len(dfs) == 2:
        return pd.merge(dfs[0], dfs[1], on="InchiKey")
    else:
        firstconcat = pd.merge(dfs[0], dfs[1], on="InchiKey")

        for index in range(2, len(dfs), 1):
            firstconcat = pd.merge(firstconcat, dfs[index], on="InchiKey")
    firstconcat = firstconcat.reset_index()
    firstconcat = firstconcat.drop(columns = 'InchiKey')
    return firstconcat

def dupRemovalClassification(dataset, columnname, curationtype):
    """ Removes duplicates with standard deviation > 0 """

    #calculate std
    stddev = [st.stdev(n) if len(n) > 1 else 0 for n in dataset[columnname]]

    discordant_index = []

    for index, std in enumerate(stddev):
        if std > 0:
            discordant_index.append(index)
        else:
            pass

    if len(discordant_index) == 0:
        pass
    
    else:
        #drop discordant data
        mask = dataset.iloc[discordant_index]
        dataset = dataset.drop(discordant_index, errors="ignore")
        mask.to_csv("{}discordant_dup_{}.csv".format(errorverbose, curationtype), sep=',', header=True, index=False)

    return dataset.reset_index(drop = True)


def removeListedValues (dataset):
    """ remove listed values and replaces it with its first indexed value """

    for column in dataset:

        for index, value in enumerate(dataset[column]):

            if type(value) == list:

                dataset.loc[index, column] = dataset[column][index][0]

            else:         
                pass

    return dataset

def dupRemovalRegression(dataset, errorverbose, columnname, threshold):
    """ Removes duplicates with standard deviation > 0.2 """

    #calculate std
    stddev = [st.stdev(n) if len(n) > 1 else 0 for n in dataset[columnname]]

    discordant_index = []

    for index, std in enumerate(stddev):
        if std > threshold:
            discordant_index.append(index)
        else:
            pass

    if len(discordant_index) == 0:
        pass
    
    else:
        #drop discordant data
        mask = dataset.iloc[discordant_index]
        dataset = dataset.drop(discordant_index, errors="ignore")
        mask.to_csv("{}discordant_dup_regression.csv".format(errorverbose), sep=',', header=True, index=False)


    dataset = dataset.reset_index()

    for index, h in enumerate(dataset[columnname]):
        if type(h) == list:
            dataset.loc[index, columnname] = st.mean(dataset[columnname][index])
        else:
            pass

    dataset = dataset.drop(columns = 'index')

    return dataset


def relationTreat(dataset, relationcolumn, activitycolumn, threshold, curationtype):
    """ This functions treats relations with  > and < activities """


    #relations > [threshold] uM will remain
    #relations < [threshold] uM will remain
    #relations = will remain
    #anything else will be removed


    equal_dataset = []
    greater_dataset = []
    lower_dataset = []
    other_relations = []

    

    for index, relation in enumerate(dataset[relationcolumn]):

        #check if relation equals "'='":
        if relation == "'='":
            equal_dataset.append(index)

        #check if relation equals "'>'":
        elif relation == "'>'":
            greater_dataset.append(index)
        
        #check if relation equals "'<'":
        elif relation == "'<'":
            lower_dataset.append(index)

        else:
            other_relations.append(index)

    if len(equal_dataset) > 0:
        equal_dataset = dataset.iloc[equal_dataset]
    else:
        pass

    if len(greater_dataset) > 0:
        greater_dataset = dataset.iloc[greater_dataset]
    else:
        pass

    if len(lower_dataset) > 0:
        lower_dataset = dataset.iloc[lower_dataset]
    else:
        pass

    if len(other_relations) > 0:
        other_relations = dataset.iloc[other_relations]
        other_relations.to_csv(f'{errorverbose}other_relations_removed_binary_curation.csv')
    else:
        pass


    # summary
    row = ["values with relation =", len(equal_dataset)]
    with open(f'./data/data_summary/{curationtype}_dupremoval.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(row)
    row = ["values with relation >", len(greater_dataset)]
    with open(f'./data/data_summary/{curationtype}_dupremoval.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    row = ["values with relation >", len(lower_dataset)]
    with open(f'./data/data_summary/{curationtype}_dupremoval.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    row = ["values with relation <= or >= removed", len(other_relations)]
    with open(f'./data/data_summary/{curationtype}_dupremoval.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    removeActivityGreater = []
    remainGreater = []
    #remove everything lower than the threshold defined
    for index, activity in enumerate(greater_dataset[activitycolumn]):

        if activity > threshold:
            remainGreater.append(index)
        else:
            removeActivityGreater.append(index)

    removeActivitylower = []
    remainLower = []
    #remove everything greater than the threshold defined
    for index, activity in enumerate(lower_dataset[activitycolumn]):

        if activity >= threshold:
            removeActivitylower.append(index)
        else:
            remainLower.append(index)


    if len(removeActivityGreater) > 0:
        removeActivityGreater = greater_dataset.iloc[removeActivityGreater]
        removeActivityGreater.to_csv(f'{errorverbose}ActivityGreater_removed_{curationtype}_curation.csv')
    else:
        pass


    if len(remainGreater) > 0:
        remainGreater = greater_dataset.iloc[remainGreater]
    else:
        pass


    if len(removeActivitylower) > 0:
        removeActivitylower = lower_dataset.iloc[removeActivitylower]
        removeActivitylower.to_csv(f'{errorverbose}Activitylower_removed_{curationtype}_curation.csv')
    else:
        pass

    if len(remainLower) > 0:
        remainLower = lower_dataset.iloc[remainLower]
    else:
        pass

    #concat ramain datasets
    concatList = [equal_dataset, remainGreater, remainLower]
    concatList = [i  for i in concatList if len(i) > 0]
    finalDataset = pd.concat(concatList).reset_index()
    finalDataset = finalDataset.drop(columns = 'index')
    finalDataset = finalDataset.reset_index(drop=True)

    #summary
    row = ["values with relation > removed", len(removeActivityGreater)]
    with open(f'./data/data_summary/{curationtype}_dupremoval.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(row)
    row = ["values with relation < removed", len(removeActivitylower)]
    with open(f'./data/data_summary/{curationtype}_dupremoval.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    return finalDataset