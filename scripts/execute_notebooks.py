"""Execute public notebooks anew, with a separate Python kernel for each file."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]

def execute(path, preview=None):
    document = nbformat.read(path, as_version=4)
    started = time.monotonic()
    for cell in document.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
            cell.metadata.pop('execution', None)
    # A temporary kernelspec selects the Poetry interpreter running this command,
    # without registering a global kernel or depending on another project's kernel.
    with tempfile.TemporaryDirectory(prefix='dpi-evt-kernel-') as runtime:
        spec = Path(runtime) / 'dpi-evt-release'
        spec.mkdir()
        (spec / 'kernel.json').write_text(json.dumps(dict(
            argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            display_name='DPI-EVT release', language='python')))
        manager = KernelManager(kernel_name='dpi-evt-release',
                                kernel_spec_manager=KernelSpecManager(kernel_dirs=[runtime]))
        client = NotebookClient(document, km=manager, timeout=600,
                                resources={'metadata': {'path': str(path.parent)}})
        try:
            client.execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
            manager.cleanup_resources()
    count = sum(cell.cell_type == 'code' for cell in document.cells)
    document.metadata['execution'] = dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        code_cells=count, mode='public aggregate tables', training_performed=False)
    nbformat.validate(document)
    nbformat.write(document, path)
    if preview:
        from nbconvert import HTMLExporter
        import base64
        preview.mkdir(parents=True, exist_ok=True)
        html, _ = HTMLExporter().from_notebook_node(document)
        (preview / (path.stem + '.html')).write_text(html)
        for index, cell in enumerate(document.cells):
            for number, output in enumerate(cell.get('outputs', [])):
                png = output.get('data', {}).get('image/png')
                if png:
                    (preview / f'{path.stem}_{index}_{number}.png').write_bytes(base64.b64decode(png))
    print(f'PASS {path.relative_to(ROOT)}: {count} code cells, {time.monotonic()-started:.1f}s', flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview-dir', type=Path, help='Optional HTML/PNG previews outside the repository')
    args=parser.parse_args()
    os.environ.setdefault('MPLCONFIGDIR', tempfile.mkdtemp(prefix='dpi-evt-matplotlib-'))
    os.environ.setdefault('IPYTHONDIR', tempfile.mkdtemp(prefix='dpi-evt-ipython-'))
    # Build the Matplotlib font cache once, before the kernels start, so that its
    # one-time message is not stored in the first notebook.
    from matplotlib import font_manager  # noqa: F401
    for path in sorted((ROOT / 'notebooks').glob('*/*.ipynb')):
        execute(path, args.preview_dir)

if __name__ == '__main__':
    main()
