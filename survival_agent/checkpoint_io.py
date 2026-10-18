"""Bounded structured checkpoints: safetensors + validated JSON, NEVER pickle.
Legacy pickle/Torch archives are rejected. No dynamic class/function loading.
"""
import base64
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile
import numpy as np
import torch
from safetensors.torch import save as tensor_save, load as tensor_load

MAX_TOTAL=128*1024*1024
MAX_JSON=32*1024*1024
DTYPES={np.dtype(x).str for x in ['bool','int8','int16','int32','int64','uint8','uint16','uint32','uint64','float16','float32','float64']}

def save(value,path):
    tensors={}
    def encode(v,depth=0):
        if depth>64:raise ValueError('Checkpoint nesting limit')
        if v is None or isinstance(v,(str,bool,int)):return v
        if isinstance(v,float):
            if not math.isfinite(v):raise ValueError('Non-finite metadata')
            return v
        if isinstance(v,np.generic):return encode(v.item(),depth+1)
        if isinstance(v,torch.Tensor):
            key=str(len(tensors));tensors[key]=v.detach().cpu().contiguous().clone()
            return {'type':'tensor','key':key}
        if isinstance(v,np.ndarray):
            if v.dtype.str not in DTYPES or v.nbytes>MAX_JSON:raise ValueError('Unsupported/oversize array')
            return {'type':'array','dtype':v.dtype.str,'shape':list(v.shape),'data':base64.b64encode(v.tobytes()).decode('ascii')}
        if isinstance(v,dict):return {'type':'dict','items':[[encode(k,depth+1),encode(x,depth+1)] for k,x in v.items()]}
        if isinstance(v,(list,tuple)):return {'type':'tuple' if isinstance(v,tuple) else 'list','items':[encode(x,depth+1) for x in v]}
        # This is the ONLY permitted structured domain object.
        from .muzero import Episode
        if isinstance(v,Episode):return {'type':'episode','fields':encode(vars(v),depth+1)}
        raise TypeError('Unsupported checkpoint value')
    metadata=json.dumps({'format':'survival-safe-checkpoint','version':1,'value':encode(value)},allow_nan=False,separators=(',',':')).encode()
    data=tensor_save(tensors)
    if len(metadata)>MAX_JSON or len(metadata)+len(data)>MAX_TOTAL:raise ValueError('Checkpoint exceeds size budget')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.checkpoint-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream,zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_STORED) as archive:
            archive.writestr('metadata.json',metadata);archive.writestr('tensors.safetensors',data)
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)

def load(path,*,map_location='cpu',weights_only=True):
    if map_location!='cpu' or weights_only is not True:raise ValueError('CPU safe structured loading only')
    path=Path(path)
    if path.stat().st_size>MAX_TOTAL+1024*1024:raise ValueError('Oversize checkpoint')
    try:
        with zipfile.ZipFile(path) as archive:
            infos=archive.infolist()
            if len(infos)!=2 or {i.filename for i in infos}!={'metadata.json','tensors.safetensors'}:raise ValueError('Not a safe checkpoint; legacy archives refused')
            if any(i.compress_type!=zipfile.ZIP_STORED or i.is_dir() for i in infos):raise ValueError('Compressed/unsupported checkpoint refused')
            if sum(i.file_size for i in infos)>MAX_TOTAL or archive.getinfo('metadata.json').file_size>MAX_JSON:raise ValueError('Oversize payload')
            raw=archive.read('metadata.json');tensor_data=archive.read('tensors.safetensors')
    except zipfile.BadZipFile as exc:raise ValueError('Not a safe checkpoint; no pickle fallback') from exc
    def bad_constant(_):raise ValueError('Non-finite metadata')
    root=json.loads(raw,parse_constant=bad_constant)
    if not isinstance(root,dict) or set(root)!={'format','version','value'} or root['format']!='survival-safe-checkpoint' or root['version']!=1:raise ValueError('Unsupported checkpoint schema')
    tensors=tensor_load(tensor_data);used=set();nodes=0
    def decode(v,depth=0):
        nonlocal nodes
        nodes+=1
        if nodes>500000 or depth>64:raise ValueError('Checkpoint complexity limit')
        if type(v) is float and not math.isfinite(v):raise ValueError('Non-finite metadata')
        if v is None or type(v) in [str,bool,int,float]:return v
        if not isinstance(v,dict) or 'type' not in v:raise ValueError('Invalid structured node')
        kind=v['type']
        if kind=='tensor' and set(v)=={'type','key'}:
            if not isinstance(v['key'],str) or v['key'] not in tensors:raise ValueError('Missing tensor')
            used.add(v['key']);return tensors[v['key']]
        if kind=='array' and set(v)=={'type','dtype','shape','data'}:
            if v['dtype'] not in DTYPES or not isinstance(v['shape'],list) or len(v['shape'])>8:raise ValueError('Unsafe array dtype/shape')
            if any(type(x) is not int or x<0 for x in v['shape']):raise ValueError('Invalid shape')
            size=math.prod(v['shape'])*np.dtype(v['dtype']).itemsize
            if size>MAX_JSON or not isinstance(v['data'],str):raise ValueError('Oversize array')
            binary=base64.b64decode(v['data'],validate=True)
            if len(binary)!=size:raise ValueError('Array size mismatch')
            return np.frombuffer(binary,dtype=v['dtype']).reshape(v['shape']).copy()
        if kind in ['list','tuple','dict'] and set(v)=={'type','items'} and isinstance(v['items'],list):
            if kind=='dict':
                result={}
                for pair in v['items']:
                    if not isinstance(pair,list) or len(pair)!=2:raise ValueError('Invalid mapping')
                    key=decode(pair[0],depth+1)
                    if type(key) not in [str,int] or key in result:raise ValueError('Invalid/duplicate mapping key')
                    result[key]=decode(pair[1],depth+1)
                return result
            result=[decode(x,depth+1) for x in v['items']]
            return tuple(result) if kind=='tuple' else result
        if kind=='episode' and set(v)=={'type','fields'}:
            from .muzero import Episode
            episode=Episode(**decode(v['fields'],depth+1));episode.validate();return episode
        raise ValueError('Unsupported structured type; no dynamic loading')
    value=decode(root['value'])
    if used!=set(tensors):raise ValueError('Unreferenced tensor payload')
    return value
