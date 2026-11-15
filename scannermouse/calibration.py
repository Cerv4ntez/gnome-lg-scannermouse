"""Per-device calibration: values fitted to one mouse, overriding the built-in ones.

The constants in stitcher.py for the two motion sensors, the LED's brightness
profile and the LED reflections were measured on one LSM-100. Another unit may
differ a little; tools/calibrate.py fits them from recordings and writes them to
~/.config/scannermouse/calibration.json, which is read here at start-up.

Keys (all optional):
  counts_to_pixels   4 x 2   both sensors' counts -> content shift in px
  counts_to_radians  4       both sensors' counts -> rotation
  single_to_pixels   2 x 2 x 2   one sensor's counts -> content shift, per sensor
  illumination       {"i,j": coefficient} of u^i v^j
  glare_spots        [[x, y], ...] in the cropped frame
"""

import json
import os
import sys

import numpy as np

PATH = os.path.join(os.environ.get('XDG_CONFIG_HOME') or os.path.expanduser('~/.config'),
                    'scannermouse', 'calibration.json')
SHAPES = {'counts_to_pixels': (4, 2), 'counts_to_radians': (4,), 'single_to_pixels': (2, 2, 2)}


def _checked(values):
    """`values` with everything converted; raises ValueError if anything is malformed."""
    out = {}
    for key, shape in SHAPES.items():
        if key in values:
            a = np.array(values[key], float)
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError(f'{key} must be {shape} numbers')
            out[key] = a
    if 'illumination' in values:
        out['illumination'] = {tuple(int(n) for n in k.split(',')): float(v)
                               for k, v in values['illumination'].items()}
        if (0, 0) not in out['illumination'] or any(len(k) != 2 for k in out['illumination']):
            raise ValueError('illumination must map "i,j" to coefficients, with "0,0"')
    if 'glare_spots' in values:
        out['glare_spots'] = tuple((int(x), int(y)) for x, y in values['glare_spots'])
    return out


def load(path=None):
    """The stored calibration, or {} if there is none (or it can't be used)."""
    try:
        with open(path or PATH) as f:
            return _checked(json.load(f))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, TypeError, AttributeError) as e:
        print(f'scannermouse: ignoring {path or PATH}: {e}', file=sys.stderr)
        return {}


def save(values, path=None):
    """Write calibration values (as from `load`; arrays and tuple keys are converted)."""
    _checked({k: ({f'{i},{j}': c for (i, j), c in v.items()} if k == 'illumination' else v)
              for k, v in values.items()})
    out = {}
    for key, value in values.items():
        if key == 'illumination':
            out[key] = {f'{i},{j}': round(float(c), 5) for (i, j), c in value.items()}
        else:
            out[key] = np.round(np.array(value, float), 6).tolist()
    os.makedirs(os.path.dirname(path or PATH), exist_ok=True)
    with open(path or PATH, 'w') as f:
        json.dump(out, f, indent=1)
