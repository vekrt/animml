import numpy as np
from scipy.stats import *
import subprocess
import glob
import os

dir_sep = "/"
if os.name == "nt":
	dir_sep = "\\"

list_src = glob.glob(f"..{dir_sep}*.nim")
names = [os.path.basename(s).replace(".nim", "") for s in list_src]
list_tests = glob.glob("*.py")

print (list_tests, list_src)

for f in list_src:
	print (f"[TEST] Checking: {f}")
	modules = []
	with open(f) as F:
		while True:
			line = next(F)
			if "import" not in line:
				break
			module = line.split(" ")[1][:-1]
			if module in names:
				modules.append(module)

	test = "test_" + os.path.basename(f).replace(".nim", ".py")

	test_src = []
	with open(test) as F:
		while True:
			line = next(F)
			if "dep = [" in line:
				td = line[line.find("[")+1: line.find("]")]
				print (td)
				break

	src_time = os.path.getmtime(f)
	module_time = [os.path.getmtime(os.path.dirname(f)  + dir_sep + m + ".nim") for m in modules]
	test_time = os.path.getmtime(test)


	flag_uptodate = True
	flag_time = True

	src_newer_test = []
	for i, mt in enumerate(module_time):
		if (mt > src_time):
			flag_uptodate = False
		if (mt > test_time):
			flag_time = False
			src_newer_test.append(modules[i])

	if (not flag_time):
		print (f"[TEST] Warning: {test} has not been updated since last modifications of the sources {*src_newer_test,}")

	if flag_uptodate:
		print (f"[TEST] {f} is already tested")
	else:
		subprocess.run(["python3.7", test])
