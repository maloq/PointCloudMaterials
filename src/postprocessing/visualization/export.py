"""Publish completed Ni example packets for offline and in-chat viewing.

Presentation only: measurements and their frozen definitions are copied verbatim.
"""
import argparse
import base64
import gzip
import json
from pathlib import Path
import re
import shutil
import zipfile

import numpy as np

from src.experiment_runner.artifacts import file_hash, write_json
from src.postprocessing.visualization.births import packet, slices
from src.postprocessing.io import read


def portable_page(text):
    pattern = r"const bytes=Uint8Array.from\(atob\('([A-Za-z0-9+/=]+)'\),c=>c.charCodeAt\(0\)\);\nconst D=JSON.parse\(await new Response\(new Blob\(\[bytes\]\).stream\(\).pipeThrough\(new DecompressionStream\('gzip'\)\)\).text\(\)\);"
    match = re.search(pattern, text)
    if match is None:
        raise ValueError('Unexpected birth viewer payload producer')
    payload = json.loads(gzip.decompress(base64.b64decode(match[1])))
    literal = json.dumps(payload, separators=(',', ':')).replace('</', '<\\/')
    text = text[:match.start()] + 'const D=' + literal + ';' + text[match.end():]
    if text.count('<script type="module">') != 1:
        raise ValueError('Unexpected birth viewer script producer')
    text = text.replace('<script type="module">', '<script>\n(function(){')
    text = text.replace('</script>', '})();\n</script>')
    # The initial Ni presentation replaced this identifier along with prose,
    # while the compressed scientific payload retained its correct field name.
    text = text.replace('trackedd_family', 'catalogued_family')
    text = text.replace('Dashed links bridge one missing observation;',
                        'Dashed links bridge one frame without a strong identity association;')
    text = text.replace('playback pauses at an excerpt boundary.',
                        'the saved movie is continuous at 0.1 ps.')
    return text


def inline_data(folders):
    result = []
    for folder in folders:
        birth, frames, movie, _, _ = packet(folder)
        confirm = birth['establishment_confirmation_ps']
        selected = [i for i, f in enumerate(frames)
                    if confirm - 3 - 1e-8 <= f['time_ps'] <= confirm + 2 + 1e-8]
        if len(selected) != 51:
            raise ValueError('Establishment excerpt must have all 51 native frames')
        shown = []
        for i in selected:
            frame = slices(movie, i)
            crystal = np.isin(frame['ptm'], [1, 2, 3])
            liquid = np.flatnonzero(~crystal)
            context = liquid[np.linspace(0, len(liquid) - 1, min(48, len(liquid)), dtype=int)]
            keep = np.sort(np.r_[np.flatnonzero(crystal), context])
            atoms = [[*[round(float(v), 2) for v in frame['positions'][j]],
                      int(frame['ptm'][j]) if crystal[j] else 0,
                      int(frame['catalogued_family'][j] or frame['precursor'][j])]
                     for j in keep]
            shown.append({**{k: frames[i][k] for k in
                             ('time_ps', 'largest_family', 'crystal_fraction', 'qbar6')},
                          'atoms': atoms})
        result.append(dict(label=f"{birth['source']} · confirmation {confirm:.1f} ps",
                           birth_event_id=birth['event_id'], frames=shown))
    return result


def history_data(folders):
    """All native frames, complete crystal/envelope atoms, stable outer context.

    The envelope only controls display sampling; it is not a phase boundary or
    a new scientific label. Coordinates are quantized to 0.02 A for display.
    """
    from scipy.spatial import ConvexHull

    result = []
    for folder in folders:
        birth, frames, movie, _, _ = packet(folder)
        metadata, blocks = [], []
        for i, observation in enumerate(frames):
            frame = slices(movie, i)
            positions, types = frame['positions'], frame['ptm']
            crystal = np.isin(types, [1, 2, 3])
            member = frame['catalogued_family'] | frame['precursor']
            cloud = positions[member].astype(float)
            inside = np.zeros(len(positions), bool)
            if len(cloud) >= 4 and np.linalg.matrix_rank(cloud - cloud.mean(0)) == 3:
                hull = ConvexHull(cloud)
                inside = np.all(positions @ hull.equations[:, :3].T + hull.equations[:, 3] <= 1e-5, axis=1)
            ids = frame['atom_ids'].astype(np.uint64)
            score = ((ids * 2654435761) & 0xffffffff) / 2**32
            keep = crystal | member | inside | (score < .12)
            xyz = np.rint(positions[keep] * 50).astype('<i2')
            flags = np.where(crystal[keep], types[keep], 0).astype('u1') | (member[keep].astype('u1') << 3)
            blocks.append(xyz.T.tobytes() + flags.tobytes())
            metadata.append([observation['time_ps'], observation['largest_family'],
                             observation['crystal_fraction'], int(keep.sum()),
                             int(frame['catalogued_family'].sum()), int((keep & ~crystal).sum())])
        raw = b''.join(blocks)
        compressed = gzip.compress(raw, mtime=0)
        if gzip.decompress(compressed) != raw:
            raise ValueError('Compressed atom display data failed round-trip verification')
        result.append(dict(label=birth['source'], birth_event_id=birth['event_id'],
                           onset=birth['first_time_ps'], confirmation=birth['establishment_confirmation_ps'],
                           frames=metadata, encoded=base64.b64encode(compressed).decode()))
    return dict(coordinate_scale=50, cases=result)


def context_page(text):
    """More visible unclassified atoms in the complete offline viewer."""
    replacements = {
        'grey atoms are liquid or other structures.': 'grey atoms have other or unclassified local structure.',
        '>Liquid context</label>': '>Other / unclassified</label>',
        '</select></div><input id="frame"': '</select><label>Grey visibility <input id="context-opacity" type="range" min="0" max="100" value="60" step="5" style="width:110px"></label></div><input id="frame"',
        'g.globalAlpha=crystal||member?1:.23;': "g.globalAlpha=crystal||member?1:Number($('context-opacity').value)/100;",
        '(crystal?4:1.7)*zoom/16': '(crystal?4:2.7)*zoom/16',
        "$('liquid').onchange=draw;": "$('liquid').onchange=draw;$('context-opacity').oninput=draw;",
    }
    for old, new in replacements.items():
        if old not in text:
            raise ValueError('Unexpected offline viewer presentation: ' + old)
        text = text.replace(old, new)
    return text


def export(root, inline_template, inline_output, profile='establishment'):
    completion = read(root / 'technical/complete.json')
    plan = read(root / 'technical/plan.json')
    if completion['state'] != 'complete' or completion['births'] != 4:
        raise ValueError('All four Ni examples must be complete before export')
    folders = [root / 'analyses/births' / slug for slug in plan['examples']]
    if len(folders) != 4:
        raise ValueError('Expected exactly four selected examples')
    for folder in folders:
        birth, frames, _, _, _ = packet(folder)
        if not birth['source'].startswith('ni-'):
            raise ValueError('Unexpected material source')
        if not np.allclose(np.diff([f['time_ps'] for f in frames]), .1, rtol=0, atol=1e-8):
            raise ValueError('Ni movie has an omitted or non-native frame')

    extended = profile == 'history-context'
    output = root / 'analyses' / ('history-context' if extended else 'desktop-export')
    gallery = output / 'ni-nucleus-gallery'
    gallery.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / 'index.html', gallery / 'index.html')
    shutil.copytree(root / 'tables', gallery / 'tables', dirs_exist_ok=True)
    for name in ('metric-contracts', 'table-contracts'):
        shutil.copytree(root / 'technical' / name, gallery / 'technical' / name, dirs_exist_ok=True)
    for name in ('search-plan.json', 'plan.json', 'complete.json'):
        shutil.copy2(root / 'technical' / name, gallery / 'technical' / name)
    for folder in folders:
        dest = gallery / folder.relative_to(root)
        originals = output / 'source-viewers'
        originals.mkdir(exist_ok=True)
        original = originals / (folder.name + '.html')
        if not original.exists():
            source = root / 'analyses/desktop-export/source-viewers' / original.name if extended else folder / 'viewer.html'
            shutil.copy2(source, original)
        # Preserve the original generated page, then publish the corrected
        # standalone presentation beside the unchanged numerical packet.
        page = portable_page(original.read_text())
        if extended:
            page = context_page(page)
        else:
            (folder / 'viewer.html').write_text(page)
        shutil.copytree(folder, dest, dirs_exist_ok=True)
        (dest / 'viewer.html').write_text(page)
    (gallery / 'README.txt').write_text(
        'Extract this ZIP, then open index.html in Chrome or Firefox. No server or network is required.\n'
        'Four illustrative Ni births from separate completed 1100 K melts, not a rate sample or catalogue.\n'
        'Interactive movies contain every saved 0.1 ps frame. GIF playback stride is 0.2 ps.\n'
        'The 14 angstrom crop can hide part of a larger component; family sizes are full-cell values.\n'
        'Time zero is the validated saved liquid; the cooling ramp ends at 20 ps.\n'
        'The historical shared figure axis says catalogue embryo onset: for these four examples this\n'
        'means the tracked eight-atom embryo onset, not an entry in a consolidated Ni catalogue.\n'
        'Original scientific packets and frozen metric definitions are included unchanged.\n')

    if extended:
        data = history_data(folders)
        inline_template = Path(__file__).with_name('ni_history_context.html')
        fragment = inline_template.read_text().replace('__NI_HISTORY_DATA__', json.dumps(data, separators=(',', ':')))
        frame_counts = [len(x['frames']) for x in data['cases']]
    else:
        data = inline_data(folders)
        fragment = inline_template.read_text()
        pattern = r'(<script id="nfp-data" type="application/json">).*?(</script>)'
        fragment, count = re.subn(pattern, lambda m: m[1] + json.dumps(data, separators=(',', ':')) + m[2], fragment, flags=re.S)
        if count != 1:
            raise ValueError('Unexpected inline template data block')
        fragment = fragment.replace('nucleus-formation-preview', 'ni-nucleus-formation-preview')
        fragment = fragment.replace('nfp-', 'nifp-').replace('Nucleus establishment', 'Ni nucleus establishment')
        frame_counts = [len(x['frames']) for x in data]
    if len(fragment.encode()) >= 1_000_000:
        raise ValueError('Inline Ni atom data exceed the 1 MB display budget')
    inline_output.parent.mkdir(parents=True, exist_ok=True)
    inline_output.write_text(fragment)

    archive = output / 'ni-nucleus-gallery.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for file in sorted(gallery.rglob('*')):
            if file.is_file():
                z.write(file, file.relative_to(output))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('Offline gallery ZIP verification failed')
    manifest = dict(search_identity=completion['search_identity'], examples=plan['examples'],
                    exporter_sha256=file_hash(Path(__file__)), archive=str(archive),
                    archive_sha256=file_hash(archive), archive_bytes=archive.stat().st_size,
                    inline=str(inline_output), inline_sha256=file_hash(inline_output),
                    inline_bytes=inline_output.stat().st_size, inline_frames=frame_counts, profile=profile,
                    inline_template_sha256=file_hash(inline_template),
                    inline_display_sampling=(dict(all_crystalline=True, all_linked=True,
                        all_atoms_inside_visible_linked_convex_envelope=True, other_outer_fraction=.12,
                        stable_atom_id_hash='((id * 2654435761) & 0xffffffff) / 2**32',
                        envelope_is_phase_boundary=False, coordinate_grid_A=.02,
                        scientific_measurements_unchanged=True) if extended else
                        dict(context_atoms_per_frame=48, position_decimals=2)),
                    scientific_packets={f.name: file_hash(f / 'complete.json') for f in folders},
                    original_viewers={f.name: file_hash(f) for f in sorted((output / 'source-viewers').glob('*.html'))},
                    gallery_files={str(f.relative_to(gallery)): file_hash(f)
                                   for f in sorted(gallery.rglob('*')) if f.is_file()})
    write_json(output / 'manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('archive', 'archive_bytes', 'inline', 'inline_bytes', 'inline_frames')}, indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--inline-template', type=Path)
    parser.add_argument('--inline-output', type=Path, required=True)
    parser.add_argument('--profile', choices=('establishment', 'history-context'), default='establishment')
    args = parser.parse_args()
    if args.profile == 'establishment' and args.inline_template is None:
        parser.error('--inline-template is required for the establishment profile')
    if args.profile == 'history-context' and args.inline_template is not None:
        parser.error('The history-context profile uses its repository template')
    export(args.root, args.inline_template, args.inline_output, args.profile)


if __name__ == '__main__':
    main()
