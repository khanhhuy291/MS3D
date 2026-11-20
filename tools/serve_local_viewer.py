"""Start the read-only BEV/3D experiment viewer from any working directory."""
import runpy
from pathlib import Path

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'web/local_viewer/server.py'), run_name='__main__')
