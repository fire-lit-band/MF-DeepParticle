"""Exact sparse OT with exhaustive full-cost dual certification.

Missing edges are added by column generation. A result is returned only when
the dual constraints have been checked against EVERY pair of occupied bins.
"""
import math
import time
import numpy as np
import ot
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


def feasible_chain(a,b):
    i=j=0
    first,second=float(a[0]),float(b[0])
    rows,cols=[],[]
    while i<len(a) and j<len(b):
        rows.append(i);cols.append(j)
        moved=min(first,second)
        first-=moved;second-=moved
        if first<1e-14:
            i+=1
            if i<len(a):first=float(a[i])
        if second<1e-14:
            j+=1
            if j<len(b):second=float(b[j])
    return np.asarray(rows),np.asarray(cols)


def price(x,y,u,v,block=128):
    n,m=len(x),len(y)
    row_index=np.arange(n,dtype=np.int64)
    row_target=np.empty(n,dtype=np.int64)
    row_value=np.empty(n)
    col_source=np.zeros(m,dtype=np.int64)
    col_value=np.full(m,np.inf)
    for start in range(0,n,block):
        reduced=cdist(x[start:start+block],y,'sqeuclidean')
        reduced-=u[start:start+block,None]
        reduced-=v[None,:]
        ri=np.argmin(reduced,axis=1)
        row_target[start:start+len(ri)]=ri
        row_value[start:start+len(ri)]=reduced[np.arange(len(ri)),ri]
        ci=np.argmin(reduced,axis=0)
        cv=reduced[ci,np.arange(m)]
        better=cv<col_value
        col_value[better]=cv[better]
        col_source[better]=start+ci[better]
    badr=row_value < -1e-10
    badc=col_value < -1e-10
    keys=np.concatenate((row_index[badr]*m+row_target[badr],col_source[badc]*m+np.arange(m)[badc]))
    return max(0.,-float(row_value.min())),keys,np.maximum(-row_value,0.)


def lattice_price(x,y,u,v,origin,widths):
    """Exact separable squared-cost c-transform over the complete lattice.

    Each min-convolution visits all positions on its axis. The composition
    therefore considers every target bin, including those far from the source.
    """
    ix=np.rint((x-origin)/widths-.5).astype(np.int64)
    iy=np.rint((y-origin)/widths-.5).astype(np.int64)
    np.testing.assert_allclose(x,origin+(ix+.5)*widths,atol=1e-11,rtol=0)
    np.testing.assert_allclose(y,origin+(iy+.5)*widths,atol=1e-11,rtol=0)
    low=np.minimum(ix.min(0),iy.min(0));high=np.maximum(ix.max(0),iy.max(0))
    ix-=low;iy-=low
    shape=tuple(high-low+1)
    assert np.prod(shape)<15000000,shape
    def transform(support,potential,queries):
        data=np.full(shape,np.inf)
        labels=np.full(shape,-1,dtype=np.int32)
        data[tuple(support.T)]=-potential
        labels[tuple(support.T)]=np.arange(len(support),dtype=np.int32)
        for axis,width in enumerate(widths):
            front=np.moveaxis(data,axis,0)
            old_labels=np.moveaxis(labels,axis,0)
            size=front.shape[0]
            values=np.ascontiguousarray(front.reshape(size,-1))
            inds=np.ascontiguousarray(old_labels.reshape(size,-1))
            new_values=np.empty_like(values)
            new_labels=np.empty_like(inds)
            columns=np.arange(values.shape[1])
            for q in range(size):
                candidate=values+(((np.arange(size)-q)*width)**2)[:,None]
                best=np.argmin(candidate,axis=0)
                new_values[q]=candidate[best,columns]
                new_labels[q]=inds[best,columns]
            data=np.moveaxis(new_values.reshape(front.shape),0,axis)
            labels=np.moveaxis(new_labels.reshape(front.shape),0,axis)
        return data[tuple(queries.T)],labels[tuple(queries.T)]
    rowmin,rowarg=transform(iy,v,ix)
    colmin,colarg=transform(ix,u,iy)
    rowvalue=rowmin-u;colvalue=colmin-v
    badr=rowvalue < -1e-10;badc=colvalue < -1e-10
    keys=np.concatenate((np.arange(len(x))[badr]*len(y)+rowarg[badr],colarg[badc]*len(y)+np.arange(len(y))[badc]))
    return max(0.,-float(rowvalue.min())),keys,np.maximum(-rowvalue,0.)


def tree_price(x,y,u,v,k=16):
    """Exact global reduced-cost search, using a Euclidean lift.

    ||[x,0]-[y,sqrt(max(v)-v_y)]||^2 = ||x-y||^2-v_y+max(v).
    cKDTree uses eps=0, so the search is exact over all target bins.
    """
    def find(source,target,source_dual,target_dual):
        lifted=np.column_stack((target,np.sqrt(np.maximum(0.,target_dual.max()-target_dual))))
        query=np.column_stack((source,np.zeros(len(source))))
        _,indices=cKDTree(lifted).query(query,k=min(k,len(target)),eps=0,workers=1)
        if indices.ndim==1: indices=indices[:,None]
        delta=source[:,None,:]-target[indices]
        reduced=np.einsum('ijk,ijk->ij',delta,delta)-source_dual[:,None]-target_dual[indices]
        bad=reduced < -1e-10
        rr,rank=np.nonzero(bad)
        return reduced.min(1),rr,indices[rr,rank]
    rowvalue,rr,cc=find(x,y,u,v)
    _,cc2,rr2=find(y,x,v,u)
    keys=np.concatenate((rr*len(y)+cc,rr2*len(y)+cc2))
    return max(0.,-float(rowvalue.min())),keys,np.maximum(-rowvalue,0.)


def repair_dual(g,x,y,a,b,u,v,value,max_rounds=12):
    """Shift balanced components of a primal plan without changing its cost.

    A component shift adds delta to source potentials and subtracts it from
    target potentials. Only a globally certified dual is accepted.
    """
    n,m=len(x),len(y)
    gc=g.tocoo()
    active=gc.data>1e-14
    rows=gc.row[active];cols=gc.col[active]+n
    adjacency=coo_matrix((np.ones(2*len(rows)),(np.r_[rows,cols],np.r_[cols,rows])),shape=(n+m,n+m)).tocsr()
    components,labels=connected_components(adjacency,directed=False)
    left,right=labels[:n],labels[n:]
    balance=np.bincount(left,weights=a,minlength=components)-np.bincount(right,weights=b,minlength=components)
    if np.max(np.abs(balance))>1e-10:return None
    delta=np.zeros(components)
    er=ec=ew=np.empty(0)
    for step in range(max_rounds):
        uu=u+delta[left];vv=v-delta[right]
        violation,extra,correction=tree_price(x,y,uu,vv,k=1)
        gap=abs(value-float(a@uu+b@vv-a@correction))
        if gap<=1e-10:return {'gap':gap,'violation':violation,'rounds':step+1,'components':components}
        ri,ci=extra//m,extra%m
        r,c=left[ri],right[ci]
        weights=np.einsum('ij,ij->i',x[ri]-y[ci],x[ri]-y[ci])-u[ri]-v[ci]
        if np.any((r==c)&(weights < -1e-8)):return None
        er=np.r_[er,r].astype(np.int64);ec=np.r_[ec,c].astype(np.int64);ew=np.r_[ew,weights]
        encoded=er*components+ec
        order=np.argsort(encoded);encoded=encoded[order]
        unique,starts=np.unique(encoded,return_index=True)
        ew=np.minimum.reduceat(ew[order],starts);er=unique//components;ec=unique%components
        for _ in range(256):
            updated=delta.copy()
            np.minimum.at(updated,er,delta[ec]+ew)
            change=float(np.max(np.abs(updated-delta)))
            delta=updated
            if change<1e-11:break
        else:return None
        delta-=delta[0]
    return None


def solve(first,second,verbose=False,grid=None,pricing='tree'):
    started=time.perf_counter()
    x,a=first;y,b=second
    n,m=len(x),len(y)
    k=min(24,n,m)
    _,cols=cKDTree(y).query(x,k=k,workers=1)
    _,rows=cKDTree(x).query(y,k=k,workers=1)
    fi,fj=feasible_chain(a,b)
    keys=np.unique(np.concatenate((np.repeat(np.arange(n),k)*m+cols.ravel(),rows.ravel()*m+np.repeat(np.arange(m),k),fi*m+fj)))
    previous_value=None
    for iteration in range(80):
        ri,ci=keys//m,keys%m
        costs=np.einsum('ij,ij->i',x[ri]-y[ci],x[ri]-y[ci])
        matrix=coo_matrix((costs,(ri,ci)),shape=(n,m))
        g,info=ot.emd(a,b,matrix,log=True,numItermax=20000000)
        if info['result_code']!=1 or info['warning']:
            raise RuntimeError(f"Restricted exact OT failed: {info['result_code']}, {info['warning']}")
        value=float(info['cost'])
        repair=None
        if previous_value is not None and abs(previous_value-value)<1e-7:
            repair=repair_dual(g,x,y,a,b,info['u'],info['v'],value)
        previous_value=value
        if pricing=='tree':
            violation,extra,corrections=tree_price(x,y,info['u'],info['v'])
        else:
            violation,extra,corrections=price(x,y,info['u'],info['v']) if grid is None else lattice_price(x,y,info['u'],info['v'],*grid)
        certified_dual=float(a@info['u']+b@info['v']-a@corrections)
        certified_gap=abs(value-certified_dual)
        if repair is not None:
            certified_gap=repair['gap'];violation=repair['violation']
        if verbose:
            print(f'SPARSE iter={iteration} edges={len(keys)} cost={value:.10f} violation={violation:.4g} gap={certified_gap:.4g} elapsed={time.perf_counter()-started:.1f}',flush=True)
        if certified_gap<=1e-10:
            g=g.tocsr()
            residual=max(float(np.max(np.abs(np.asarray(g.sum(1)).ravel()-a))),float(np.max(np.abs(np.asarray(g.sum(0)).ravel()-b))))
            gap=certified_gap
            assert residual<1e-9 and gap<1e-8 and value>=-1e-10
            return {'w2':math.sqrt(max(value,0.)),'w2_squared':max(value,0.),'reference_occupied_bins':n,'generated_occupied_bins':m,
                    'solver_result_code':1,'marginal_residual':residual,'primal_dual_gap':gap,'transport_nonzeros':int(g.nnz),
                    'seconds':time.perf_counter()-started,'column_generation_iterations':iteration+1,'priced_edges':int(n*m),
                    'max_dual_violation_before_feasibility_correction':violation,'solver':'exact sparse network simplex with global dual certificate','pricing':pricing,
                    'component_dual_repair':repair}
        old_count=len(keys)
        keys=np.union1d(keys,extra)
        assert len(keys)>old_count, (violation,len(extra))
    raise RuntimeError('Column generation did not converge within 80 rounds')


if __name__=='__main__':
    import run_evaluation as run
    rng=np.random.default_rng(123)
    checks=[]
    for dimension in (3,4):
        x,y=rng.normal(size=(125,dimension)),rng.normal(size=(143,dimension))
        a,b=rng.random(125),rng.random(143)
        a,b=a/a.sum(),b/b.sum()
        result=solve((x,a),(y,b),verbose=True)
        dense=float(ot.emd2(a,b,cdist(x,y,'sqeuclidean')))
        assert abs(dense-result['w2_squared'])<1e-10
        checks.append({'dimension':dimension,'dense_difference':abs(dense-result['w2_squared']),**result})
    run.save(run.OUT/'sparse_solver_verification.json',{'status':'passed','checks':checks})
    with np.load(run.OUT/'clouds/ks4d/sigma10.npz') as z:
        clouds={k:z[k].copy() for k in z.files if k!='meta'}
    measures,origin,widths=run.measures_for(clouds)
    # Verify the complete-lattice pricing against brute force on a subset,
    # retaining all target bins and arbitrary dual potentials.
    x,a=measures['reference'];y,b=measures['seed88_mf']
    x=x[::100];y=y[::100];u=rng.normal(size=len(x));v=rng.normal(size=len(y))
    direct=price(x,y,u,v)
    lattice=lattice_price(x,y,u,v,origin,widths)
    tree=tree_price(x,y,u,v)
    np.testing.assert_allclose(direct[2],lattice[2],atol=1e-11,rtol=0)
    np.testing.assert_allclose(direct[2],tree[2],atol=1e-11,rtol=0)
    run.save(run.OUT/'lattice_pricing_verification.json',{'status':'passed','maximum_difference':float(np.max(np.abs(direct[2]-lattice[2]))),'tree_difference':float(np.max(np.abs(direct[2]-tree[2])))})
    result=solve(measures['reference'],measures['seed88_mf'],verbose=True,grid=(origin,widths))
    run.save(run.OUT/'sparse_tree_probe.json',result)
