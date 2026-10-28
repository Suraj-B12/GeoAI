import sys, json
import pandas as pd, numpy as np
from scipy.stats import spearmanr
sid=sys.argv[1]
d=pd.read_csv(f'out/bestframe_{sid}.csv')
rs=[];rf=[];rows=[]
for k,g in d.groupby('k'):
    if len(g)<3: continue
    rs.append(spearmanr(g.sharp,g.win_pos)[0]); rf.append(spearmanr(g.fair_sharp,g.win_pos)[0])
    ok=g[g.blur_px<=4] if (g.blur_px<=4).any() else g
    code=ok.loc[ok.sharp.idxmax()]; fair=g.loc[g.fair_sharp.idxmax()]; near=g.loc[g.win_pos.idxmin()]
    # nearest-sharp rule: among frames with win_pos<=0.35, sharpest
    nz=g[g.win_pos<=g.win_pos.min()+0.35]; ns=nz.loc[nz.sharp.idxmax()]
    rows.append({'k':k,'n':len(g),'code_win':code.win_pos,'fair_win':fair.win_pos,'near_win':near.win_pos,'ns_win':ns.win_pos,
      'code_fair_rel':code.fair_sharp/fair.fair_sharp,'near_fair_rel':near.fair_sharp/fair.fair_sharp,'ns_fair_rel':ns.fair_sharp/fair.fair_sharp,
      'code_vlm_rows':code.vlm_rows,'near_vlm_rows':near.vlm_rows,'ns_vlm_rows':ns.vlm_rows,
      'code_gsd_c_vlm':code.gsd_c_vlm,'near_gsd_c_vlm':near.gsd_c_vlm,'ns_gsd_c_vlm':ns.gsd_c_vlm,
      'same_code_fair':int(code.frame==fair.frame),'code_blur':code.blur_px,'near_blur':near.blur_px,'ns_blur':ns.blur_px})
R=pd.DataFrame(rows)
out={'session':sid,'segments':len(R),'candidates':len(d),
 'within_seg_spearman_code_sharp_vs_winpos_median':float(np.nanmedian(rs)),
 'within_seg_spearman_fair_sharp_vs_winpos_median':float(np.nanmedian(rf)),
 'share_code_choice_equals_fair_choice':float(R.same_code_fair.mean()),
 'medians':R.median().round(3).to_dict(),
 'share_near_fair_ge_0.9':float((R.near_fair_rel>=0.9).mean()),'share_ns_fair_ge_0.9':float((R.ns_fair_rel>=0.9).mean()),
 'share_code_fair_ge_0.9':float((R.code_fair_rel>=0.9).mean()),
 'vlm_rows_gain_ns_over_code_median':float((R.ns_vlm_rows/R.code_vlm_rows).median()),
 'gsd_c_gain_ns_over_code_median':float((R.code_gsd_c_vlm/R.ns_gsd_c_vlm).median()),
 'ns_blur_over4':float((R.ns_blur>4).mean()),'code_blur_over4':float((R.code_blur>4).mean())}
R.to_csv(f'out/bestframe_summary_{sid}.csv',index=False)
json.dump(out,open(f'out/bestframe_summary_{sid}.json','w'),indent=1)
print(json.dumps(out,indent=1))
