"""Native-space resolution, continuous fidelity and nontrivial coherence assays."""
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, r2_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler


def participation(z):
    centered = np.asarray(z, float)-np.mean(z, axis=0)
    eigenvalues = np.linalg.eigvalsh(centered.T@centered/len(centered)).clip(0)
    total = eigenvalues.sum()
    return dict(effective_rank=float(total**2/np.square(eigenvalues).sum()) if total>0 else 0.,
                variance=float(total), pair_rms=float(np.sqrt(2*total)))


def boundary_coherence(z, labels, pair_distance, n, mask):
    """Matched-distance AUROC for a reference-label change across a spatial bond."""
    a, b = z[:n], z[n:]
    distance = np.linalg.norm(a-b, axis=1)
    changed = labels[:n] != labels[n:]
    eligible = np.flatnonzero(mask)
    if len(eligible)<100:
        return dict(auc=None, pairs=len(eligible), bins=0, same_label_normalized_distance=None)
    edges = np.unique(np.quantile(pair_distance[eligible], np.linspace(0, 1, 6)))
    bin_id = np.searchsorted(edges[1:-1], pair_distance, side='right')
    scores, weights = [], []
    for j in range(max(1, len(edges)-1)):
        ids = eligible[bin_id[eligible] == j]
        counts = np.bincount(changed[ids].astype(int), minlength=2)
        if counts.min() < 10:
            continue
        scores.append(roc_auc_score(changed[ids], distance[ids])); weights.append(len(ids))
    scale = participation(a[eligible])['pair_rms']
    same = eligible[~changed[eligible]]
    return dict(auc=float(np.average(scores, weights=weights)) if scores else None,
        pairs=len(eligible), changed_pairs=int(changed[eligible].sum()), bins=len(scores),
        same_label_normalized_distance=float(distance[same].mean()/scale) if scale>0 and len(same) else None)


def classification(z, y, split, classes):
    train, test = split==0, split==1
    counts = {str(c): dict(train=int((train & (y==c)).sum()), test=int((test & (y==c)).sum())) for c in classes}
    supported = [c for c in classes if counts[str(c)]['train'] >= 20]
    train &= np.isin(y, supported)
    test_supported = test & np.isin(y, supported)
    result = dict(counts=counts, supported_classes=list(map(int, supported)),
                  test_coverage=float(test_supported.sum()/test.sum()), balanced_accuracy=None, ap={})
    if len(supported)<2 or len(np.unique(y[test_supported]))<2:
        return result
    scaler = StandardScaler().fit(z[train])
    model = LogisticRegression(C=1., class_weight='balanced', max_iter=2000, random_state=0)
    model.fit(scaler.transform(z[train]), y[train])
    probs = model.predict_proba(scaler.transform(z[test]))
    prediction = model.classes_[probs.argmax(1)]
    keep = np.isin(y[test], supported)
    result['balanced_accuracy'] = float(balanced_accuracy_score(y[test][keep], prediction[keep]))
    result['confusion'] = confusion_matrix(y[test][keep], prediction[keep], labels=model.classes_).tolist()
    for c in classes:
        key = str(c)
        if c in model.classes_ and 20 <= (y[test]==c).sum() < test.sum():
            result['ap'][key] = float(average_precision_score(y[test]==c, probs[:, list(model.classes_).index(c)]))
        else:
            result['ap'][key] = None
    return result


def fidelity(z, target, density, split, eligible):
    train = (split==0) & eligible; test = (split==1) & eligible
    result = dict(train_count=int(train.sum()), test_count=int(test.sum()),
                  r2=None, density_r2=None, conditional_r2_gain=None)
    if min(train.sum(), test.sum())<40:
        return result
    # Target scaling gives each nonconstant descriptor equal weight in the fit.
    y_scaler = StandardScaler().fit(target[train])
    active = y_scaler.var_ > 1e-12
    if not active.any():
        return result
    y = y_scaler.transform(target)[:, active]
    scores = {}
    for name, x in [('z', z), ('density', density[:, None]), ('joint', np.c_[density, z])]:
        scaler = StandardScaler().fit(x[train]); x = scaler.transform(x)
        model = Ridge(alpha=10.).fit(x[train], y[train])
        scores[name] = np.atleast_1d(r2_score(y[test], model.predict(x[test]), multioutput='raw_values'))
    result.update(r2=scores['z'].tolist(), density_r2=scores['density'].tolist(),
        active_columns=np.flatnonzero(active).tolist(),
        conditional_r2_gain=float(np.mean(scores['joint']-scores['density'])),
        mean_r2=float(np.mean(scores['z'])))
    return result


def perturbation(z, perturbed):
    scale = participation(z)['pair_rms']
    distance = np.linalg.norm(perturbed-z, axis=1)
    return dict(pair_rms=scale, normalized_median=float(np.median(distance)/scale) if scale>0 else None,
                normalized_p95=float(np.quantile(distance, .95)/scale) if scale>0 else None)
