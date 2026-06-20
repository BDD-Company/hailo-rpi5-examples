import os
import sys

# make `import guidance...` and `import intercept_config` resolve to BDD/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
