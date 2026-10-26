import run_evaluation as r
from exact_sparse import solve
import numpy as np

with np.load(r.CLOUDS_ROOT/'ks4d/sigma10.npz') as z:
    points={k:z[k] for k in ('reference','seed88_mf')}
measures,origin,widths=r.measures_for(points)
result=solve(measures['reference'],measures['seed88_mf'],verbose=True,grid=(origin,widths))
dense=r.js(r.OUT/'results/ks4d/sigma10/seed88_mf.json')
assert abs(result['w2_squared']-dense['w2_squared'])<1e-9
r.save(r.OUT/'solver_comparison.json',{'dense':dense,'sparse':result,'cost_difference':abs(result['w2_squared']-dense['w2_squared'])})
print(result,flush=True)
