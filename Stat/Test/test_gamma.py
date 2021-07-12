import numpy as np
from scipy.special import gammainc

xx = np.arange(0, 200, 10) / 100.0
aa = np.arange(1, 100, 10) / 10.0

print ("import ../gamma")
print ("from std/fenv import epsilon")
print ("import math")
for a in aa:
	for x in xx:
		print ("""assert abs(gamma_p({a:.3f}, {x:.3f}) - {res:.60f}) < 8 * epsilon(float), \"[GAMMA TEST] Failed at a={a:.3f} and x={x:.3f}\" """.format(a=a, x=x, res=gammainc(a, x)))
		#print ("""echo \"a={a:.3f} x={x:.3f}\"  """.format(a=a, x=x))
#

print ("echo \"[GAMMA TEST] passed\"")
