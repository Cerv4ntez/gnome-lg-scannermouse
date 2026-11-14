"""Build scannermouse/data/xfeat.onnx from the XFeat authors' weights, without PyTorch.

    curl -LO https://github.com/verlab/accelerated_features/raw/main/weights/xfeat.pt
    python3 -I tools/xfeat_onnx.py xfeat.pt scannermouse/data/xfeat.onnx

The weights file is a PyTorch pickle; it is read with a whitelist that only allows
plain tensor data, so it cannot run code. The network (XFeatModel in
verlab/accelerated_features, Apache-2.0) is written as an ONNX graph by hand, with
batch normalisation folded into the convolutions, for a fixed 224 x 640 input.
"""

import collections
import pickle
import struct
import sys
import zipfile

import numpy as np

DTYPES = {'FloatStorage': np.float32, 'LongStorage': np.int64, 'HalfStorage': np.float16, 'DoubleStorage': np.float64}
def load_state_dict(src):
    z = zipfile.ZipFile(src)
    root = z.namelist()[0].split('/')[0]
    class Storage:
        def __init__(self, name): self.name = name
    def rebuild(storage, offset, size, stride, *_):
        data = np.frombuffer(z.read(f'{root}/data/{storage[1]}'), DTYPES[storage[0].name])
        return np.lib.stride_tricks.as_strided(data[offset:], shape=size, strides=[s * data.itemsize for s in stride]).copy()
    class Safe(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) == ('collections', 'OrderedDict'): return collections.OrderedDict
            if (module, name) == ('torch._utils', '_rebuild_tensor_v2'): return rebuild
            if module == 'torch' and name in DTYPES: return Storage(name)
            raise pickle.UnpicklingError(f'refused {module}.{name}')
        def persistent_load(self, pid):
            # ('storage', storage_type, key, location, numel)
            return (pid[1], pid[2])
    state = Safe(z.open(f'{root}/data.pkl')).load()
    return {k: np.asarray(v) for k, v in state.items()}

def varint(n):
    out = bytearray()
    while True:
        b = n & 0x7f; n >>= 7
        if n: out.append(b | 0x80)
        else: out.append(b); return bytes(out)
def key(field, wire): return varint(field << 3 | wire)
def f_int(field, v): return key(field, 0) + varint(v if v >= 0 else v + (1 << 64))
def f_bytes(field, b): return key(field, 2) + varint(len(b)) + b
def f_str(field, s): return f_bytes(field, s.encode())
def f_float(field, v): return key(field, 5) + struct.pack('<f', v)

def tensor(name, arr):
    arr = np.ascontiguousarray(arr)
    dt = {np.dtype(np.float32): 1, np.dtype(np.int64): 7}[arr.dtype]
    return b''.join(f_int(1, d) for d in arr.shape) + f_int(2, dt) + f_str(8, name) + f_bytes(9, arr.tobytes())
def attr_ints(name, vals): return f_str(1, name) + b''.join(f_int(8, v) for v in vals) + f_int(20, 7)
def attr_int(name, v): return f_str(1, name) + f_int(3, v) + f_int(20, 2)
def attr_str(name, s): return f_str(1, name) + f_bytes(4, s.encode()) + f_int(20, 3)
def attr_floats(name, vals): return f_str(1, name) + b''.join(f_float(7, v) for v in vals) + f_int(20, 6)
def node(op, inputs, outputs, attrs=()):
    return (b''.join(f_str(1, i) for i in inputs) + b''.join(f_str(2, o) for o in outputs) +
            f_str(3, outputs[0] + '_' + op) + f_str(4, op) + b''.join(f_bytes(5, a) for a in attrs))
def value_info(name, shape):
    dims = b''.join(f_bytes(1, f_int(1, d)) for d in shape)
    return f_str(1, name) + f_bytes(2, f_bytes(1, f_int(1, 1) + f_bytes(2, dims)))

def build(w, H, W):
    nodes, inits = [], []
    cnt = [0]
    def fresh(p):
        cnt[0] += 1; return f'{p}{cnt[0]}'
    def conv(x, wname, k, stride, bias_name=None, bn=None, relu=False):
        Wt = w[wname].astype(np.float32); b = w[bias_name].astype(np.float32) if bias_name else np.zeros(Wt.shape[0], np.float32)
        if bn:
            m, v = w[bn + '.running_mean'], w[bn + '.running_var']; s = 1 / np.sqrt(v + 1e-5)
            Wt = Wt * s[:, None, None, None]; b = (b - m) * s
        wn, bnm = fresh('w'), fresh('b'); inits.append(tensor(wn, Wt)); inits.append(tensor(bnm, b))
        out = fresh('c'); p = k // 2
        nodes.append(node('Conv', [x, wn, bnm], [out], [attr_ints('kernel_shape', [k, k]), attr_ints('strides', [stride, stride]), attr_ints('pads', [p, p, p, p])]))
        if relu:
            r = fresh('r'); nodes.append(node('Relu', [out], [r])); out = r
        return out
    def basic(x, name, k=3, stride=1): return conv(x, name + '.layer.0.weight', k, stride, bn=name + '.layer.1', relu=True)
    def seq(x, name, spec):
        for i, (k, s) in enumerate(spec): x = basic(x, f'{name}.{i}', k, s)
        return x
    def add(a, b):
        o = fresh('a'); nodes.append(node('Add', [a, b], [o])); return o
    def up(x, f):
        sc = fresh('s'); inits.append(tensor(sc, np.array([1, 1, f, f], np.float32)))
        o = fresh('u'); nodes.append(node('Resize', [x, '', sc], [o], [attr_str('mode', 'linear'), attr_str('coordinate_transformation_mode', 'half_pixel')])); return o
    x = 'image'
    pooled = fresh('p'); nodes.append(node('AveragePool', [x], [pooled], [attr_ints('kernel_shape', [4, 4]), attr_ints('strides', [4, 4])]))
    skip = conv(pooled, 'skip1.1.weight', 1, 1, bias_name='skip1.1.bias')
    x1 = seq(x, 'block1', [(3, 1), (3, 2), (3, 1), (3, 2)])
    x2 = seq(add(x1, skip), 'block2', [(3, 1), (3, 1)])
    x3 = seq(x2, 'block3', [(3, 2), (3, 1), (1, 1)])
    x4 = seq(x3, 'block4', [(3, 2), (3, 1), (3, 1)])
    x5 = seq(x4, 'block5', [(3, 2), (3, 1), (3, 1), (1, 1)])
    f = seq(add(add(x3, up(x4, 2)), up(x5, 4)), 'block_fusion', [(3, 1), (3, 1)])
    feats = conv(f, 'block_fusion.2.weight', 1, 1, bias_name='block_fusion.2.bias')
    h = seq(feats, 'heatmap_head', [(1, 1), (1, 1)])
    hl = conv(h, 'heatmap_head.2.weight', 1, 1, bias_name='heatmap_head.2.bias')
    nodes.append(node('Sigmoid', [hl], ['heat']))
    nodes.append(node('Identity', [feats], ['feats']))
    # 8x8 unfold (channel = row * 8 + col) as a stride-8 convolution with one-hot kernels
    onehot = np.zeros((64, 1, 8, 8), np.float32)
    for r in range(8):
        for c in range(8): onehot[r * 8 + c, 0, r, c] = 1
    wn = fresh('w'); inits.append(tensor(wn, onehot))
    u = fresh('f'); nodes.append(node('Conv', ['image', wn], [u], [attr_ints('kernel_shape', [8, 8]), attr_ints('strides', [8, 8])]))
    k = seq(u, 'keypoint_head', [(1, 1), (1, 1), (1, 1)])
    kl = conv(k, 'keypoint_head.3.weight', 1, 1, bias_name='keypoint_head.3.bias')
    nodes.append(node('Identity', [kl], ['logits']))
    graph = (b''.join(f_bytes(1, n) for n in nodes) + f_str(2, 'xfeat') + b''.join(f_bytes(5, t) for t in inits) +
             f_bytes(11, value_info('image', [1, 1, H, W])) +
             b''.join(f_bytes(12, value_info(n, s)) for n, s in [('feats', [1, 64, H // 8, W // 8]), ('logits', [1, 65, H // 8, W // 8]), ('heat', [1, 1, H // 8, W // 8])]))
    return f_int(1, 8) + f_str(2, 'scannermouse') + f_bytes(7, graph) + f_bytes(8, f_str(1, '') + f_int(2, 17))

if __name__ == '__main__':
    weights = load_state_dict(sys.argv[1])
    open(sys.argv[2], 'wb').write(build(weights, 224, 640))
