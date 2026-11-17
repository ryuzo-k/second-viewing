"""Builds inputs/typicality.csv: for each 001187 test image, the mean cosine similarity of its CLIP ViT-B/32 image
embedding to the other test images of the same category and image set, z-scored within category. The test images of
an image set are the union over all its sessions (some sessions are cut short); every session of a set must store the
same picture under the same image ID.

Usage: DATA=... python typicality.py"""
import csv, hashlib, os
import numpy as np, h5py, torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import analysis as A
import load_001187 as L


def embed(images):
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').eval()
    proc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    x = proc(images=[Image.fromarray(a) for a in images], return_tensors='pt')['pixel_values']
    with torch.no_grad():
        e = model.visual_projection(model.vision_model(pixel_values=x).pooler_output).numpy()
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def test_images(image_set):
    """Image ID -> picture (rows x columns x RGB) for every test image of the set, checked across its sessions."""
    pics, digest = {}, {}
    for p in L.files():
        if L.IMAGE_SET[os.path.basename(p)] != image_set: continue
        f = h5py.File(p, 'r'); templates = f['stimulus/templates/StimulusTemplates']
        for i in sorted(set(f['intervals/LTM_trials']['PicIDs'][:].astype(int))):
            a = np.ascontiguousarray(np.transpose(templates[f'image_{i}'][:], (1, 0, 2)))   # stored transposed
            h = hashlib.sha1(a.tobytes()).hexdigest()
            if digest.setdefault(i, h) != h: raise ValueError(f'image set {image_set}: image {i} differs between sessions')
            pics.setdefault(i, a)
    return pics


def typicality():
    out = {}
    for s in (1, 2, 3):
        pics = test_images(s)
        for c in range(1, 6):
            ii = [i for i in sorted(pics) if L.cat_of(i) == c]
            X = embed([pics[i] for i in ii])
            t = np.array([(X.sum(0) - x) @ x / (len(X) - 1) for x in X])
            for i, v in zip(ii, (t - t.mean()) / t.std()): out[(s, i)] = float(v)
    return out


if __name__ == '__main__':
    with open(os.path.join(A.INPUTS, 'typicality.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n'); w.writerow(['image_set', 'image_id', 'typicality'])
        for (s, i), v in sorted(typicality().items()): w.writerow([s, i, repr(v)])
