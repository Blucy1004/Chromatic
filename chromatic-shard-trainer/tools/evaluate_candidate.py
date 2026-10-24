from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.dont_write_bytecode=True
from shardtrain.evaluate import main
if __name__=='__main__':raise SystemExit(main())
