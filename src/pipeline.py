"""
pipeline.py - ML failure triage on the Loghub HDFS_v1 dataset.

Run:  python src/pipeline.py --data <folder with HDFS.log and preprocessed/>

Four steps, each written to results/:
  1. Failure clustering (unsupervised NLP): mask variable fields, TF-IDF,
     k-means with k chosen by silhouette score.
  2. Failure prediction (supervised): per-session event counts ->
     Logistic Regression tuned with GridSearchCV.
  3. Session anomaly detection (unsupervised): Isolation Forest trained
     without labels on the same features, then checked against the labels.
  4. Time-series anomaly detection: per-minute event counts -> Isolation
     Forest flags unusual minutes; checked against failed-session activity.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

sys.path.insert(0, os.path.dirname(__file__))
from logtools import LINE_RE, block_ids, mask_message

SEED = 42
RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')


def scores(y_true, y_pred):
    return {'precision': round(precision_score(y_true, y_pred), 4),
            'recall': round(recall_score(y_true, y_pred), 4),
            'f1': round(f1_score(y_true, y_pred), 4)}


# ── Step 0: one streaming pass over the raw log ─────────────────────────────

def scan_raw_log(log_path, failed_blocks):
    """Read HDFS.log line by line (constant memory) and collect:
    masked-message counts, which messages touch failed sessions, and
    per-minute counts of each message type."""
    msg_total = Counter()
    msg_failed = Counter()
    per_minute = defaultdict(Counter)
    minute_failed = Counter()
    minute_total = Counter()
    unparsed = 0
    with open(log_path, 'r', encoding='utf-8', errors='replace') as fh:
        for raw in fh:
            m = LINE_RE.match(raw)
            if m is None:
                unparsed += 1
                continue
            masked = mask_message(m['message'])
            minute = m['date'] + m['time'][:4]   # yymmddHHMM, cheaper than datetime per line
            msg_total[masked] += 1
            per_minute[minute][masked] += 1
            minute_total[minute] += 1
            if any(b in failed_blocks for b in block_ids(m['message'])):
                msg_failed[masked] += 1
                minute_failed[minute] += 1
    return msg_total, msg_failed, per_minute, minute_total, minute_failed, unparsed


# ── Step 1: failure clustering ──────────────────────────────────────────────

def cluster_messages(msg_total, msg_failed):
    messages = list(msg_total)
    vec = TfidfVectorizer(token_pattern=r'[A-Za-z<>]+', lowercase=True)
    X = vec.fit_transform(messages)
    best = None
    for k in range(4, 13):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        s = silhouette_score(X, km.labels_)
        if best is None or s > best[1]:
            best = (k, s, km)
    k, sil, km = best
    terms = np.array(vec.get_feature_names_out())
    clusters = []
    for c in range(k):
        idx = [i for i, lab in enumerate(km.labels_) if lab == c]
        total = sum(msg_total[messages[i]] for i in idx)
        failed = sum(msg_failed[messages[i]] for i in idx)
        top = terms[np.argsort(km.cluster_centers_[c])[::-1][:5]].tolist()
        example = max((messages[i] for i in idx), key=lambda m: msg_total[m])
        clusters.append({'cluster': c, 'message_types': len(idx), 'lines': total,
                         'failed_session_share': round(failed / total, 4) if total else 0,
                         'top_terms': top, 'example': example})
    clusters.sort(key=lambda d: -d['failed_session_share'])
    return {'distinct_message_types': len(messages), 'k': k,
            'silhouette': round(float(sil), 4), 'clusters': clusters}


# ── Steps 2 and 3: session-level prediction and anomaly detection ───────────

def session_models(matrix_csv):
    df = pd.read_csv(matrix_csv)
    y = (df['Label'].astype(str).str.lower().isin(['fail', 'anomaly', '1'])).astype(int).values
    X = df[[c for c in df.columns if c.startswith('E')]].values.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    # log1p squashes huge counts so one noisy event type cannot dominate
    model = make_pipeline(FunctionTransformer(np.log1p),
                          LogisticRegression(max_iter=2000))
    grid = GridSearchCV(model,
                        {'logisticregression__C': [0.01, 0.1, 1, 10],
                         'logisticregression__class_weight': [None, 'balanced']},
                        scoring='f1', cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                        n_jobs=1)
    grid.fit(X_tr, y_tr)
    pred = grid.predict(X_te)
    prob = grid.predict_proba(X_te)[:, 1]
    lr = {'best_params': grid.best_params_, 'cv_f1': round(grid.best_score_, 4),
          'test': scores(y_te, pred), 'test_roc_auc': round(roc_auc_score(y_te, prob), 4),
          'confusion_matrix': confusion_matrix(y_te, pred).tolist()}

    # Isolation Forest never sees labels; contamination set to the train
    # failure rate, which assumes we know roughly how often failures happen.
    iso = IsolationForest(n_estimators=200, contamination=float(y_tr.mean()),
                          random_state=SEED, n_jobs=1).fit(np.log1p(X_tr))
    iso_pred = (iso.predict(np.log1p(X_te)) == -1).astype(int)
    iso_score = -iso.score_samples(np.log1p(X_te))
    iforest = {'test': scores(y_te, iso_pred), 'test_roc_auc': round(roc_auc_score(y_te, iso_score), 4),
               'confusion_matrix': confusion_matrix(y_te, iso_pred).tolist()}

    info = {'sessions': int(len(y)), 'failed_sessions': int(y.sum()),
            'failure_rate': round(float(y.mean()), 4), 'event_types': int(X.shape[1]),
            'train': int(len(y_tr)), 'test': int(len(y_te))}
    return info, lr, iforest


# ── Step 4: time-series anomaly detection ───────────────────────────────────

def timeseries_anomalies(per_minute, minute_total, minute_failed):
    minutes = sorted(per_minute)
    types = sorted({m for c in per_minute.values() for m in c})
    X = np.array([[per_minute[t][m] for m in types] for t in minutes], dtype=float)
    share = np.array([minute_failed[t] / minute_total[t] for t in minutes])
    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=SEED).fit(np.log1p(X))
    flagged = iso.predict(np.log1p(X)) == -1
    top = np.quantile(share, 0.98)
    hot = share >= top
    res = {'minutes': len(minutes), 'flagged_minutes': int(flagged.sum()),
           'mean_failed_share_flagged': round(float(share[flagged].mean()), 4),
           'mean_failed_share_other': round(float(share[~flagged].mean()), 4),
           'hot_minutes_top2pct': int(hot.sum()),
           'hot_minutes_caught': int((hot & flagged).sum())}
    series = pd.DataFrame({'minute': pd.to_datetime(minutes, format='%y%m%d%H%M'), 'lines': [minute_total[t] for t in minutes],
                           'failed_share': share, 'flagged': flagged})
    return res, series


def plots(lr, iforest, series, clusters):
    sns.set_theme(style='whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, (name, r) in zip(axes, [('Logistic Regression', lr), ('Isolation Forest', iforest)]):
        sns.heatmap(np.array(r['confusion_matrix']), annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['normal', 'fail'], yticklabels=['normal', 'fail'], ax=ax)
        ax.set_title(name); ax.set_xlabel('predicted'); ax.set_ylabel('actual')
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'confusion_matrices.png'), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.4))
    sns.lineplot(data=series, x='minute', y='lines', ax=ax, linewidth=0.6, color='#4c72b0')
    f = series[series['flagged']]
    ax.scatter(f['minute'], f['lines'], color='#c44e52', s=12, label='flagged minute', zorder=3)
    ax.set_title('Log lines per minute with Isolation Forest flags'); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'timeseries_flags.png'), dpi=130); plt.close(fig)

    c = pd.DataFrame(clusters['clusters'])
    c['label'] = c['top_terms'].apply(lambda t: ' '.join(t[:3]))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    sns.barplot(data=c, y='label', x='failed_session_share', ax=ax, color='#dd8452')
    ax.set_title('Share of lines from failed sessions, per message cluster'); ax.set_ylabel('')
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'cluster_failure_share.png'), dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='folder containing HDFS.log and preprocessed/')
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()

    labels = pd.read_csv(os.path.join(a.data, 'preprocessed', 'anomaly_label.csv'))
    failed = set(labels.loc[labels['Label'].str.lower() == 'anomaly', 'BlockId'])

    print('scanning raw log ...', flush=True)
    msg_total, msg_failed, per_minute, minute_total, minute_failed, unparsed = \
        scan_raw_log(os.path.join(a.data, 'HDFS.log'), failed)
    print(f'  {sum(msg_total.values()):,} lines parsed, {unparsed} unparsed ({time.time()-t0:.0f}s)', flush=True)

    clusters = cluster_messages(msg_total, msg_failed)
    print('clustering done', flush=True)
    info, lr, iforest = session_models(os.path.join(a.data, 'preprocessed', 'Event_occurrence_matrix.csv'))
    print('session models done', flush=True)
    ts, series = timeseries_anomalies(per_minute, minute_total, minute_failed)
    print('time series done', flush=True)

    out = {'dataset': info, 'lines_parsed': sum(msg_total.values()), 'lines_unparsed': unparsed,
           'clustering': clusters, 'logistic_regression': lr, 'isolation_forest_sessions': iforest,
           'timeseries': ts, 'runtime_s': round(time.time() - t0, 1)}
    with open(os.path.join(RESULTS, 'metrics.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2, default=str)
    plots(lr, iforest, series, clusters)
    print(json.dumps({k: out[k] for k in ('dataset', 'logistic_regression', 'isolation_forest_sessions', 'timeseries')},
                     indent=2, default=str))


if __name__ == '__main__':
    main()
