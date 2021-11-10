
import hashlib
import sys
from pathlib import Path

# monkey patch Path to have a copy method
def _copy_to(self, target):
    import shutil
    assert self.is_file()
    shutil.copy(str(self), str(target))  # str() only there for Python < (3, 6)

Path.copy_to = _copy_to


if len(sys.argv) < 2:
    print()
    print("First parameter is the name of the folder to create.")
    print("All others are names of existing miniset folders.")
    print()
    exit()

dest = sys.argv[1]
sources = sys.argv[2:]

Path(dest).absolute().mkdir(parents=True, exist_ok=True)

for i,src in enumerate(sources):
    src = Path(src)
    for srcf in src.glob('class-*/im-*.png'):
        ds = hashlib.sha1(src.name.encode('utf-8')).hexdigest()[:4]
        dstf = str(srcf.relative_to(src)).replace('/im-', '/ds-'+ds+'-')
        dstf = Path(dest, Path(dstf))
        dstf.absolute().parent.mkdir(parents=True, exist_ok=True)
        print(srcf, '->', dstf)
        srcf.copy_to(dstf)
