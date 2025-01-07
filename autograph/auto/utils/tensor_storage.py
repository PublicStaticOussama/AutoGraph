import numpy as np
import h5py
import json
from pathlib import Path
from typing import Union, Dict, Any, List
import scipy.sparse as sparse

class TensorStorage:
    @staticmethod
    def save_tensors(tensors: Dict[str, Union[np.ndarray, sparse.spmatrix, List[np.ndarray], List[sparse.spmatrix]]], 
                     filepath: str,
                     format: str = 'hdf5',
                     compress: bool = True,
                     metadata: Dict[str, Any] = None) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'hdf5':
            with h5py.File(filepath, 'w') as f:
                for key, tensor in tensors.items():
                    if isinstance(tensor, list):
                        g = f.create_group(key)
                        g.attrs['is_list'] = True
                        for i, t in enumerate(tensor):
                            if sparse.issparse(t):
                                sg = g.create_group(f'item_{i}')
                                t = t.tocoo()
                                sg.create_dataset('data', data=t.data, compression='gzip' if compress else None)
                                sg.create_dataset('row', data=t.row, compression='gzip' if compress else None)
                                sg.create_dataset('col', data=t.col, compression='gzip' if compress else None)
                                sg.attrs['shape'] = t.shape
                                sg.attrs['format'] = 'sparse'
                            else:
                                g.create_dataset(f'item_{i}', data=t, compression='gzip' if compress else None)
                    else:
                        if sparse.issparse(tensor):
                            g = f.create_group(key)
                            tensor = tensor.tocoo()
                            g.create_dataset('data', data=tensor.data, compression='gzip' if compress else None)
                            g.create_dataset('row', data=tensor.row, compression='gzip' if compress else None)
                            g.create_dataset('col', data=tensor.col, compression='gzip' if compress else None)
                            g.attrs['shape'] = tensor.shape
                            g.attrs['format'] = 'sparse'
                        else:
                            f.create_dataset(key, data=tensor, compression='gzip' if compress else None)
                
                if metadata:
                    f.create_dataset('__metadata__', data=json.dumps(metadata))
        else:
            raise ValueError("Only HDF5 format supported for tensor lists")
    
    @staticmethod
    def load_tensors(filepath: str) -> tuple[Dict[str, Union[np.ndarray, sparse.spmatrix, List[Union[np.ndarray, sparse.spmatrix]]]], Dict[str, Any]]:
        filepath = Path(filepath)
        
        with h5py.File(filepath, 'r') as f:
            tensors = {}
            metadata = {}
            
            if '__metadata__' in f:
                metadata = json.loads(f['__metadata__'][()])
            
            for key in f.keys():
                if key == '__metadata__':
                    continue
                    
                if isinstance(f[key], h5py.Group):
                    if 'is_list' in f[key].attrs:
                        tensor_list = []
                        for i in range(len(f[key].keys())):
                            item = f[key][f'item_{i}']
                            if isinstance(item, h5py.Group):
                                tensor_list.append(sparse.coo_matrix(
                                    (item['data'][:], (item['row'][:], item['col'][:])),
                                    shape=item.attrs['shape']
                                ))
                            else:
                                tensor_list.append(item[:])
                        tensors[key] = tensor_list
                    else:
                        g = f[key]
                        tensors[key] = sparse.coo_matrix(
                            (g['data'][:], (g['row'][:], g['col'][:])),
                            shape=g.attrs['shape']
                        )
                else:
                    tensors[key] = f[key][:]
            
            return tensors, metadata

