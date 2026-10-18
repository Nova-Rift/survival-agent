import os
from pathlib import Path
import subprocess
import sys
import unittest

class IsolationTest(unittest.TestCase):
    def test_no_other_repository_imports(self):
        code="""import sys,importlib.abc,runpy
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in BLOCKED:raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
BODY
""".replace('BLOCKED',repr(['survival_env', 'survival_adapters'])).replace('BODY',"runpy.run_module('examples.inspect_model',run_name='__main__')")
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH='',CUDA_VISIBLE_DEVICES='')
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=Path(__file__).resolve().parent.parent,env=env,capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
