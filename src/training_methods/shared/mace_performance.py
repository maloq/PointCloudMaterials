"""Identical cloud ordering across MACE forward and gradient-cache replay."""
import torch


def encode_views(model, x, material, size, geometry_cache=None):
    views=x.shape[1]
    return encode_clouds(model,x.flatten(0,1),material.repeat_interleave(views),size,geometry_cache).reshape(len(x),views,256)


def encode_clouds(model,clouds,materials,size,geometry_cache=None):
    values=[]
    for start in range(0,len(clouds),size):
        x=clouds[start:start+size];m=materials[start:start+size]
        if geometry_cache is None:
            values.append(model.encoder(x,m))
        else:
            geometry=model.encoder.build_geometry(x,m)
            geometry_cache.append(geometry)
            values.append(model.encoder.forward_from_geometry(geometry))
    return torch.cat(values)


def replay_chunks(model,clouds,materials,size,geometries):
    if geometries is None:
        for start in range(0,len(clouds),size):
            sl=slice(start,start+size)
            yield model.encoder(clouds[sl],materials[sl]),sl
    else:
        start=0
        for geometry in geometries:
            sl=slice(start,start+geometry.batch_size)
            yield model.encoder.forward_from_geometry(geometry),sl
            start+=geometry.batch_size
