import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
build_dir = os.path.join(repo_root, "build")
sys.path.insert(0, build_dir)

import nanotorch

print("nanotorch.__version__:", nanotorch.__version__)
print("nanotorch.hello():", nanotorch.hello())
print("nanotorch.add_ints(2,3):", nanotorch.add_ints(2,3))
