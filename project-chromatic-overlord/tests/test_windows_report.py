"""Reporter fixture tests; these tests do not execute a Windows engine."""
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class ReportPaths(unittest.TestCase):
 def test_relative_artifact_path_and_absolute_document(self):
  with tempfile.TemporaryDirectory(dir=ROOT/'reports') as tmp:
   folder=Path(tmp);results=folder/'native';(results/'platform').mkdir(parents=True);(results/'matrix').mkdir()
   for name,data in [('build.json',{'build_exit':0}),('ctest-status.json',{'exit_code':0}),('platform/protocol.json',{'passed':True,'native_windows':True}),('matrix/completed.json',{'cells':18,'os':'Windows'}),('matrix/summary.json',[])]:
    (results/name).write_text(json.dumps(data))
   document=folder/'fixture-report.md'
   args=[sys.executable,str(ROOT/'tools/emit_windows_report.py'),'--results',str(results.relative_to(ROOT)),'--document',str(document),'--source','unit-fixture','--run-url','https://example.invalid/unit-fixture']
   subprocess.run(args,cwd=ROOT,check=True,stdout=subprocess.PIPE)
   self.assertIn('**PASSED**',document.read_text());self.assertIn('native/build.json',document.read_text())
   # A Linux protocol record must never be accepted as native Windows evidence.
   (results/'platform/protocol.json').write_text(json.dumps({'passed':True,'native_windows':False}))
   subprocess.run(args,cwd=ROOT,check=True,stdout=subprocess.PIPE)
   self.assertIn('FAILED / INCOMPLETE',document.read_text())
if __name__=='__main__':unittest.main()
