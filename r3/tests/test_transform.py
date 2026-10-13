"""Unit test: MultiOmicsPreprocessor.transform() applies fitted params with NO refit (no leakage)."""
import os, sys, pickle
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT,"r2")); sys.path.insert(0, ROOT); os.chdir(ROOT)
import nbdefs as nb

def test_transform_no_refit():
    pre=pickle.load(open("r3/cache_single_archive/preprocessor.pkl","rb"))
    # snapshot fitted params
    before={m:(pre.scalers[m].mean_.copy(), np.array(pre.feature_names[m])) for m in pre.feature_names}
    rng=np.random.default_rng(0); fake={}
    for m in nb.config.modalities:
        names=list(pre.get_feature_names(m)); cols=names[:int(0.8*len(names))]+["EXTRA_%d"%i for i in range(3)]
        df=pd.DataFrame(rng.normal(size=(15,len(cols))),columns=cols); df.iloc[::4,0]=np.nan; fake[m]=df
    out=pre.transform(fake)
    for m in nb.config.modalities:
        names=list(pre.get_feature_names(m))
        assert out[m].shape==(15,len(names)), f"{m} wrong shape"
        assert np.isfinite(out[m]).all(), f"{m} has NaN/Inf"
    # prove NO refit: fitted params unchanged after transform
    for m in pre.feature_names:
        assert np.allclose(before[m][0], pre.scalers[m].mean_), f"{m} scaler mutated -> refit occurred!"
        assert (before[m][1]==np.array(pre.feature_names[m])).all(), f"{m} feature names mutated!"
    print("PASS: transform() applied fitted params with no refit (shapes+finite ok, params unchanged).")

if __name__=="__main__":
    test_transform_no_refit()
