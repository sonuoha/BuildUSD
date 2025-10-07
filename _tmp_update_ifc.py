from pathlib import Path
text = Path('src/ifc_converter/process_ifc.py').read_text(encoding='utf-8')
text = text.replace('from pxr import Gf\n', '')
text = text.replace('import hashlib\n\nif TYPE_CHECKING:', 'import hashlib\n\nfrom .pxr_utils import require_pxr_module\n\nif TYPE_CHECKING:')
text = text.replace('log = logging.getLogger(__name__)\n\n', 'log = logging.getLogger(__name__)\n\nGf = require_pxr_module("Gf")\n\n')
Path('src/ifc_converter/process_ifc.py').write_text(text, encoding='utf-8')
