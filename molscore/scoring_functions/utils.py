from typing import Union, Sequence, Callable
import subprocess
import multiprocessing
import threading
import os
import signal
import numpy as np
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import rdMolDescriptors, rdmolops, DataStructs
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.Avalon import pyAvalonTools
from dask.distributed import Client, LocalCluster

from pathlib import Path
import pystow
os.environ['PYSTOW_NAME'] = '.pidgin_data'
from zenodo_client import Zenodo as ZenodoBase

# ----- Multiprocessing related -----
def Pool(*args):
    context = multiprocessing.get_context("fork")
    return context.Pool(*args)


class timedThread(object):
    """
    Subprocess wrapped into a thread to add a more well defined timeout, use os to send a signal to all PID in
    group just to be sure... (nothing else was doing the trick)
    """

    def __init__(self, timeout):
        self.cmd = None
        self.timeout = timeout
        self.process = None

    def run(self, cmd):
        self.cmd = cmd.split()
        def target():
            self.process = subprocess.Popen(self.cmd, preexec_fn=os.setsid,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = self.process.communicate()
            return

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(self.timeout)
        if thread.is_alive():
            print('Process timed out...')
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        return


class timedSubprocess(object):
    """
    Currently used
    """
    def __init__(self, timeout=None, shell=False):
        self.cmd = None
        self.cwd = None
        self.timeout = timeout
        self.shell = shell
        self.process = None

    def run(self, cmd, cwd=None):
        if not self.shell:
            self.cmd = cmd.split()
            self.process = subprocess.Popen(self.cmd, preexec_fn=os.setsid, cwd=cwd,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            self.cmd = cmd
            self.process = subprocess.Popen(self.cmd, shell=self.shell, cwd=cwd,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out, err = self.process.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            print('Process timed out...')
            out, err = ''.encode(), f'Timed out at {self.timeout}'.encode() # Encode for consistency
            if not self.shell:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            else:
                self.process.kill()
        return out, err


class DaskUtils:

    # TODO add dask-jobqueue templates https://jobqueue.dask.org/en/latest/

    @classmethod
    def setup_dask(cls, cluster_address_or_n_workers=None, local_directory=None, logger=None):
        client = None

        # Check if it's a string
        if isinstance(cluster_address_or_n_workers, str):
            client = Client(cluster_address_or_n_workers)
            print(f"Dask worker dashboard: {client.dashboard_link}")
        # Or a number
        elif isinstance(cluster_address_or_n_workers, float) or isinstance(cluster_address_or_n_workers, int):
            if int(cluster_address_or_n_workers) > 1:
                cluster = LocalCluster(n_workers=int(cluster_address_or_n_workers), threads_per_worker=1, local_directory=local_directory)
                client = Client(cluster)
                print(f"Dask worker dashboard: {client.dashboard_link}")
        # Or is unrecognized
        else:
            if (logger is not None) and (cluster_address_or_n_workers is not None):
                logger.warning(f"Unrecognized dask input {cluster_address_or_n_workers}")

        return client

# ----- Zenodo related ------
class Zenodo(ZenodoBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_latest_record(self, record_id: Union[int, str]) -> str:
        """Get the latest record related to the given record."""
        res_json = self.get_record(record_id).json()
        # Still works even in the case that the given record ID is the latest.
        #latest = res_json["links"]["latest"].split("/")[-1]
        latest = os.path.join(record_id, 'versions', 'latest') ##### Change in Zenodo API
        return latest

    def download(self, record_id: Union[int, str], name: str, *, force: bool = False, parts: Union[None, Sequence[str], Callable[[str, str, str], Sequence[str]]] = None) -> Path:
        """Download the file for the given record.

        :param record_id: The Zenodo record id
        :param name: The name of the file in the Zenodo record
        :param parts: Optional arguments on where to store with :func:`pystow.ensure`. If none given, goes in
            ``<PYSTOW_HOME>/zendoo/<CONCEPT_RECORD_ID>/<RECORD>/<PATH>``. Where ``CONCEPT_RECORD_ID`` is the
            consistent concept record ID for all versions of the same record. If a function is given, the function
            should take 3 position arguments: concept record id, record id, and version, then return a sequence for
            PyStow. The name of the file is automatically appended to the end of the sequence.
        :param force: Should the file be re-downloaded if it already is cached? Defaults to false.
        :returns: the path to the downloaded file.
        :raises FileNotFoundError: If the Zenodo record doesn't have a file with the given name

        For example, to download the most recent version of NSoC-KG, you can
        use the following command:

        >>> path = Zenodo().download('4574555', 'triples.tsv')

        Even as new versions of the data are uploaded, this command will always
        be able to check if a new version is available, download it if it is, and
        return the local file path. If the most recent version is already downloaded,
        then it returns the local file path to the cached file.

        The file path uses :mod:`pystow` under the ``zenodo`` module and uses the
        "concept record ID" as a submodule since that is the consistent identifier
        between different records that are versions of the same data.
        """
        res_json = self.get_record(record_id).json()
        # conceptrecid is the consistent record ID for all versions of the same record
        concept_record_id = res_json["conceptrecid"]
        # FIXME send error report to zenodo about this - shouldn't version be required?
        version = res_json["metadata"].get("version", "v1")
        
        try:
            for file in res_json["files"]:
                if file["filename"] == name:  ##### Change in Zenodo API, "filename" not "key"
                    url = os.path.join(file["links"]["self"].rsplit("/", 1)[0], name, "content") ##### file["links"]["self"] no longer works instead try e.g., https://zenodo.org/records/7547691/files/trained_models.zip/content
                    break
            else:
                raise FileNotFoundError(f"zenodo.record:{record_id} does not have a file with key {name}")
        except:
            for file in res_json["files"]:
            if file["key"] == name:
                url = file["links"]["self"]
                break
            else:
                raise FileNotFoundError(f"zenodo.record:{record_id} does not have a file with key {name}")

        if parts is None:
            parts = [self.module.replace(":", "-"), concept_record_id, version]
        elif callable(parts):
            parts = parts(concept_record_id, str(record_id), version)
        return pystow.ensure(*parts, name=name, url=url, force=force)


# ----- Chemistry related -----
def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog('rdApp.error')
    return


def get_mol(mol: Union[str, Chem.rdchem.Mol]):
    """
    Get RDkit mol
    :param mol:
    :return: RDKit Mol
    """
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    elif isinstance(mol, Chem.rdchem.Mol):
        pass
    else:
        raise TypeError("Molecule is not a string (SMILES) or rdkit.mol")

    if not mol:
        mol = None

    return mol


class Fingerprints:
    """
    Class to organise Fingerprint generation
    """

    @staticmethod
    def get(mol: Union[str, Chem.rdchem.Mol], name: str, nBits: int, asarray: bool = False):
        """
        Get fp by str instead of method
        :param mol: RDKit mol or Smiles
        :param name: Name of FP [ECFP4, ECFP4c, FCFP4, FCFP4c, ECFP6, ECFP6c, FCFP6, FCFP6c, Avalon, MACCSkeys, AP, hashAP, hashTT, RDK5, RDK6, RDK7, PHCO]
        :param nBits: Number of bits
        :return:
        """
        mol = get_mol(mol)
        generator = getattr(Fingerprints, name, None)

        if generator is None:
            raise KeyError(f"\'{name}\' not recognised as a valid fingerprint")

        if mol is not None:
            return generator(mol, nBits, asarray)

    # Circular fingerprints
    @staticmethod
    def ECFP4(mol, nBits, asarray):
        if asarray:
            return np.asarray(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits))
        else:
            return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits)

    @staticmethod
    def ECFP4c(mol, nBits, asarray):
        if asarray:
            fp = rdMolDescriptors.GetMorganFingerprint(mol, radius=2, useCounts=True)
            nfp = np.zeros((1, nBits), np.int32)
            for idx, v in fp.GetNonzeroElements().items():
                nidx = idx % nBits
                nfp[0, nidx] += int(v)
            return nfp.reshape(-1)
        else:
            return rdMolDescriptors.GetMorganFingerprint(mol, radius=2, useCounts=True)

    @staticmethod
    def FCFP4(mol, nBits, asarray):
        if asarray:
            np.asarray(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits, useFeatures=True))
        else:
            return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits, useFeatures=True)

    @staticmethod
    def FCFP4c(mol, nBits, asarray):
        if asarray:
            fp = rdMolDescriptors.GetMorganFingerprint(mol, radius=2, useCounts=True, useFeatures=True)
            nfp = np.zeros((1, nBits), np.int32)
            for idx, v in fp.GetNonzeroElements().items():
                nidx = idx % nBits
                nfp[0, nidx] += int(v)
            return nfp.reshape(-1)
        else:
            return rdMolDescriptors.GetMorganFingerprint(mol, radius=2, useCounts=True, useFeatures=True)

    @staticmethod
    def ECFP6(mol, nBits, asarray):
        if asarray:
            return np.asarray(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=nBits))
        else:
            return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=nBits)

    @staticmethod
    def ECFP6c(mol, nBits, asarray):
        if asarray:
            fp = rdMolDescriptors.GetMorganFingerprint(mol, radius=3, useCounts=True)
            nfp = np.zeros((1, nBits), np.int32)
            for idx, v in fp.GetNonzeroElements().items():
                nidx = idx % nBits
                nfp[0, nidx] += int(v)
            return nfp.reshape(-1)
        else:
            return rdMolDescriptors.GetMorganFingerprint(mol, radius=3, useCounts=True)

    @staticmethod
    def FCFP6(mol, nBits, asarray):
        if asarray:
            np.asarray(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=nBits, useFeatures=True))
        else:
            return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=nBits, useFeatures=True)

    @staticmethod
    def FCFP6c(mol, nBits, asarray):
        if asarray:
            fp = rdMolDescriptors.GetMorganFingerprint(mol, radius=3, useCounts=True, useFeatures=True)
            nfp = np.zeros((1, nBits), np.int32)
            for idx, v in fp.GetNonzeroElements().items():
                nidx = idx % nBits
                nfp[0, nidx] += int(v)
            return nfp.reshape(-1)
        else:
            return rdMolDescriptors.GetMorganFingerprint(mol, radius=3, useCounts=True, useFeatures=True)

    # Substructure fingerprints
    @staticmethod
    def Avalon(mol, nBits, asarray):
        if asarray:
            return np.asarray(pyAvalonTools.GetAvalonFP(mol, nBits=nBits))
        else:
            return pyAvalonTools.GetAvalonFP(mol, nBits=nBits)

    @staticmethod
    def MACCSkeys(mol, nBits, asarray):
        if asarray:
            return np.asarray(rdMolDescriptors.GetMACCSKeysFingerprint(mol))
        else:
            return rdMolDescriptors.GetMACCSKeysFingerprint(mol)

    # Path-based fingerprints
    @staticmethod
    def AP(mol, nBits, asarray):
        if asarray:
            fp = rdMolDescriptors.GetAtomPairFingerprint(mol, maxLength=10)
            nfp = np.zeros((1, nBits), np.int32)
            for idx, v in fp.GetNonzeroElements().items():
                nidx = idx % nBits
                nfp[0, nidx] += int(v)
            return nfp.reshape(-1)
        else:
            return rdMolDescriptors.GetAtomPairFingerprint(mol, maxLength=10)

    @staticmethod
    def hashAP(mol, nBits, asarray):
        if asarray:
            return np.asarray(rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nBits))
        else:
            return rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nBits)

    @staticmethod
    def hashTT(mol, nBits, asarray):
        if asarray:
            return np.asarray(rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=nBits))
        else:
            return rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=nBits)

    @staticmethod
    def RDK5(mol, nBits, asarray):
        if asarray:
            return np.asarray(rdmolops.RDKFingerprint(mol, maxPath=5, fpSize=nBits, nBitsPerHash=2))
        else:
            return rdmolops.RDKFingerprint(mol, maxPath=5, fpSize=nBits, nBitsPerHash=2)

    @staticmethod
    def RDK6(mol, nBits, asarray):
        if asarray:
            return np.asarray(rdmolops.RDKFingerprint(mol, maxPath=6, fpSize=nBits, nBitsPerHash=2))
        else:
            return rdmolops.RDKFingerprint(mol, maxPath=6, fpSize=nBits, nBitsPerHash=2)

    @staticmethod
    def RDK7(mol, nBits, asarray):
        if asarray:
            return np.asarray(rdmolops.RDKFingerprint(mol, maxPath=7, fpSize=nBits, nBitsPerHash=2))
        else:
            return rdmolops.RDKFingerprint(mol, maxPath=7, fpSize=nBits, nBitsPerHash=2)

    # Pharmacophore-based
    @staticmethod
    def PHCO(mol, nBits, asarray):
        if asarray:
            return np.asarray(Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory))
        else:
            return Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)


class SimilarityMeasures:

    @staticmethod
    def get(name, bulk=False):
        """
        Helper function to get correct RDKit similarity function by name
        :param name: RDKit similarity type [AllBit, Asymmetric, BraunBlanquet, Cosine, McConnaughey, Dice, Kulczynski, Russel, OnBit, RogotGoldberg, Sokal, Tanimoto]
        :param bulk: Whether get bulk similarity
        """
        if bulk:
            name = "Bulk" + name + "Similarity"
        else:
            name = name + "Similarity"

        similarity_function = getattr(DataStructs, name, None)
        if similarity_function is None:
            raise KeyError(f"\'{name}\' not found")

        return similarity_function