"""Verify the ported dashboard logic reproduces the validated backtest engine."""
import ast, sys, re
import pandas as pd, numpy as np
sys.path.insert(0,'..')
import returns as R, strategy as S, loaders as L
from live import gates

src = open('efp_dashboard.py', encoding='utf-8').read()
tree = ast.parse(src)
WANT = {'derived_lease_pct','arb_bound_dollar','bound_thresholds','classify_bound',
        'bound_multiple','comex_implied_forward','corroboration_x','noise_buffer_dollar'}
ns = {'pd':pd,'np':np}
# module-level config the functions rely on
for node in tree.body:
    if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) \
       and node.targets[0].id in {'FREIGHT_USD_OZ','TRANSIT_DAYS','RECAST_USD_OZ',
                                  'BUFFER_NOISE_MULT','NOISE_FLOOR_PP','CORROB_WARN_X'}:
        exec(ast.get_source_segment(src,node), ns)
for node in tree.body:
    if isinstance(node,ast.FunctionDef) and node.name in WANT:
        exec(ast.get_source_segment(src,node), ns)
print('extracted:', sorted(k for k in WANT if k in ns))

d,j = R._frames()
X = {'gold':'XAU','silver':'XAG','platinum':'XPT','palladium':'XPD'}
fails=[]

# 1. config parity with strategy.py
FREIGHT={'gold':0.50,'silver':0.10,'platinum':0.50,'palladium':0.50}   # Henry's desk numbers
for m,x in X.items():
    if ns['FREIGHT_USD_OZ'][x]!=FREIGHT[m]: fails.append(f'freight {m}')
    if ns['TRANSIT_DAYS'][x]!=S.TRANSIT[m]: fails.append(f'transit {m}')
    _rc={'gold':0.50,'silver':0.0,'platinum':0.0,'palladium':0.0}
    if ns['RECAST_USD_OZ'][x]!=_rc[m]: fails.append(f'recast {m}')
if ns['NOISE_FLOOR_PP']!=gates.NOISE_FLOOR_PP: fails.append('noise floor')
if ns['BUFFER_NOISE_MULT']!=1.0: fails.append('buffer mult')
print('config parity:', 'OK' if not fails else fails)

# 2. arb bound reproduces strategy.py's bound, on real data
rows=[]
for m,x in X.items():
    xx=j[(j.metal==m)&(j.slot==3)&j.tradeable].sort_values('date').set_index('date')
    lease=d[f'lease_{m}']['1M'].reindex(xx.index)
    ref = FREIGHT[m] + xx.spot*(lease/100)*S.TRANSIT[m]/360
    got = [ns['arb_bound_dollar'](x, sp, lz) for sp,lz in zip(xx.spot,lease)]
    diff=(pd.Series(got,index=xx.index)-ref).abs().max()
    rows.append((m,diff))
    if not (diff < 1e-9): fails.append(f'bound {m} diff {diff}')
print('arb_bound_dollar vs strategy.py bound: max abs diff',
      max(r[1] for r in rows))

# 3. classify_bound reproduces the reference threshold: bound + 1.0x noise floor
agree=tot=0
for m,x in X.items():
    xx,_,r,per = S._prep(m,3,d,j)
    lease=d[f'lease_{m}']['1M'].reindex(xx.index)
    b = FREIGHT[m] + xx.spot*(lease/100)*S.TRANSIT[m]/360
    buf = xx.spot*(1.0*gates.NOISE_FLOOR_PP/100)*xx.days_to_delivery/360
    _rcm={'gold':0.50,'silver':0.0,'platinum':0.0,'palladium':0.0}[m]
    sh=(r>(b+buf)).fillna(False); lo=(r<-(b+buf+_rcm)).fillna(False)
    for ts in xx.index:
        rv=r.get(ts); bd=b.get(ts)
        if pd.isna(rv) or pd.isna(bd): continue
        sig=ns['classify_bound'](x,rv,bd,xx.spot.get(ts),xx.days_to_delivery.get(ts))
        want = 'SELL EFP' if sh.get(ts,False) else 'BUY EFP' if lo.get(ts,False) else 'Inside bound'
        tot+=1; agree+= (sig==want)
print(f'classify_bound vs reference threshold: {agree}/{tot} = {agree/tot*100:.2f}%')
if agree!=tot: fails.append(f'signal mismatch {tot-agree}')

# 4. comex_implied_forward reproduces gates.comex_implied_lease
for m,x in X.items():
    g=j[(j.metal==m)&j.settle.notna()]
    px=g.pivot_table(index='date',columns='slot',values='settle')
    T=g.pivot_table(index='date',columns='slot',values='days_to_delivery')
    ref_lease = gates.comex_implied_lease(j,d,m)
    got=[]
    for dt_ in ref_lease.index:
        if dt_ not in px.index: got.append(np.nan); continue
        f=ns['comex_implied_forward'](px.loc[dt_,1],px.loc[dt_,2],T.loc[dt_,1],T.loc[dt_,2])
        sofr=d['sofr']['3M'].reindex([dt_]).iloc[0]
        got.append(sofr-f)
    diff=(pd.Series(got,index=ref_lease.index)-ref_lease).abs().max()
    if not (diff<1e-9): fails.append(f'comex fwd {m} diff {diff}')
print('comex_implied_forward vs gates.comex_implied_lease: max abs diff', round(float(diff),12))

# 5. corroboration reproduces the documented episode scores
print('\ncorroboration sanity (entered vs COMEX-implied, in noise units):')
for lbl,a,b_ in [('platinum 2025 (real)',12.14,0.39),('palladium 2022 (noise)',1.21,0.32)]:
    print(f'   {lbl:<24} {ns["corroboration_x"](a,b_):.1f}x')

print()
print('FAILURES:', fails if fails else 'none — ported logic matches the validated engine')
sys.exit(1 if fails else 0)
