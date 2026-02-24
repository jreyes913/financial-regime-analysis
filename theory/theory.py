import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.insert(0, os.getcwd())
import time
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from datetime import datetime, timedelta
from src.data_loader import DataLoader
from dotenv import dotenv_values

config = dotenv_values(".env", encoding="utf-8-sig")
api_key = config["TIINGO_KEY"]

SYMBOLS = [
    "KO","PG","JNJ","MCD","WMT",
    "AAPL","MSFT","NVDA","AMZN",
    "JPM","GS","BAC",
    "XOM","CVX",
    "TSLA","META",
]

end_date   = datetime.today().date()
start_date = (datetime.today() - timedelta(days=365.25 * 5)).date()

TREND_COL    = "trend"
TREND_STATES = 2
VOL_STATES   = 2
TREND_LABELS = {0: "Down", 1: "Up"}
VOL_LABELS   = {0: "Low",  1: "High"}

TECH_ORDER  = ["H(D)", "L(D)", "H(U)", "L(U)"]
QUANT_ORDER = ["D(H)", "U(H)", "D(L)", "U(L)"]
SEMANTIC    = {("H(D)", "D(H)"), ("L(D)", "D(L)"), ("H(U)", "U(H)"), ("L(U)", "U(L)")}


class HMMLayer:
    def __init__(self):
        self.means=None
        self.variances=None
        self.states=None
        self.transmat=None
        self.cutoffs=None
        self.n=None

    def print(self, title=None, labels=None):
        n=self.n
        total=len(self.states)
        labels=labels or {i:str(i) for i in range(n)}
        cols=[labels[i] for i in range(n)]
        w=20+14*n

        if title:
            print("\n  " + "-"*w)
            print("  " + ("  "+title).center(w))
        print("  " + "-"*w)
        print("  " + "".ljust(20) + "".join(f"{c:>14}" for c in cols))
        print("  " + "-"*w)

        counts=[int((self.states==i).sum()) for i in range(n)]

        print("  " + f"{'Count':20}" + "".join(f"{c:>14}" for c in counts))
        print("  " + f"{'Share':20}" + "".join(f"{c/total:>14.2%}" for c in counts))
        print("  " + f"{'Mean':20}" + "".join(f"{self.means[i]:>14.6f}" for i in range(n)))
        print("  " + f"{'Variance':20}" + "".join(f"{self.variances[i]:>14.6f}" for i in range(n)))
        print("  " + "-"*w)

        cutoff_labels=[f"{cols[i]}->{cols[i+1]}" for i in range(n-1)]
        for label,cutoff in zip(cutoff_labels,self.cutoffs):
            print("  " + f"{'Cutoff '+label:20}{cutoff:>14.6f}")

        print("  " + "-"*w)
        print("  " + f"{'Transition Matrix':20}" + "".join(f"{'-> '+c:>14}" for c in cols))
        print("  " + "-"*w)

        for i,row in enumerate(self.transmat):
            print("  " + f"{'From '+cols[i]:20}" + "".join(f"{x:>14.2%}" for x in row))
        print("  " + "-"*w)


class HierarchicalHMM:
    def __init__(self, df, parent_col, child_col):
        self.df=df.copy()
        self.parent_col=parent_col
        self.child_col=child_col
        self.parent_layer=HMMLayer()
        self.child_layers={}

    def _fit_hmm(self, values, n_components):
        col=values.reshape(-1,1)
        model=GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=200,
            random_state=42
        )
        model.fit(col)

        states=model.predict(col)
        means=model.means_.ravel()
        variances=model.covars_.ravel()
        transmat=model.transmat_

        order=np.argsort(means)
        means=means[order]
        variances=variances[order]
        transmat=transmat[np.ix_(order,order)]

        remap={old:new for new,old in enumerate(order)}
        states=np.array([remap[s] for s in states])

        layer=HMMLayer()
        layer.means=means
        layer.variances=variances
        layer.transmat=transmat
        layer.states=states
        layer.cutoffs=[(means[i]+means[i+1])/2 for i in range(n_components-1)]
        layer.n=n_components
        return layer

    def _n_components(self, col):
        return TREND_STATES if col==TREND_COL else VOL_STATES

    def _labels(self, col):
        return TREND_LABELS if col==TREND_COL else VOL_LABELS

    def fit_parent(self):
        values=self.df[self.parent_col].dropna().values
        self.parent_layer=self._fit_hmm(values,self._n_components(self.parent_col))
        self.df["parent_state"]=self.parent_layer.states

    def fit_children(self,min_rows=10):
        self.child_layers={}
        self.df["child_state"]=-1

        for parent_state in sorted(self.df["parent_state"].unique()):
            subset=self.df[self.df["parent_state"]==parent_state]

            if len(subset)<min_rows:
                print(f"Skipping parent_state={parent_state}: only {len(subset)} rows")
                continue

            child_values=subset[self.child_col].dropna().values
            child_layer=self._fit_hmm(child_values,self._n_components(self.child_col))
            self.child_layers[parent_state]=child_layer

            idx=self.df.index[self.df["parent_state"]==parent_state]
            self.df.loc[idx,"child_state"]=child_layer.states

    def print_parent(self):
        self.parent_layer.print(
            title=f"Parent Layer  [{self.parent_col}]",
            labels=self._labels(self.parent_col)
        )

    def print_children(self):
        parent_labels=self._labels(self.parent_col)
        child_labels=self._labels(self.child_col)

        for parent_state,child_layer in self.child_layers.items():
            child_layer.print(
                title=f"Child Layer  [parent={parent_labels[parent_state]}]  [{self.child_col}]",
                labels=child_labels
            )


def compute_features(df):
    df["200_sma"]=df["close"].rolling(200).mean()
    df["200_std"]=df["close"].rolling(200).std()
    df["trend"]=(df["close"]-df["200_sma"])/df["200_std"]

    df["normal_open"]=np.log(df["open"]/df["close"].shift(1))
    df["normal_high"]=np.log(df["high"]/df["open"])
    df["normal_low"]=np.log(df["low"]/df["open"])
    df["normal_close"]=np.log(df["close"]/df["open"])

    df["normal_close_var"]=df["normal_close"].rolling(14).var(ddof=0)
    df["normal_open_var"]=df["normal_open"].rolling(14).var(ddof=0)

    df["rs_var"]=(
        df["normal_high"]*(df["normal_high"]-df["normal_close"]) +
        df["normal_low"]*(df["normal_low"]-df["normal_close"])
    ).rolling(14).mean()

    k=0.34/(1.34+((14+1)/(14-1)))
    df["yz_var"]=df["normal_open_var"]+k*df["normal_close_var"]+(1-k)*df["rs_var"]
    df["vol_quant"]=np.log(np.sqrt(df["yz_var"]))

    df["prev_close"]=df["close"].shift(1)

    df["tr"]=df[["high","low","prev_close"]].apply(
        lambda row: max(
            row["high"]-row["low"],
            abs(row["high"]-row["prev_close"]) if pd.notna(row["prev_close"]) else 0,
            abs(row["low"]-row["prev_close"]) if pd.notna(row["prev_close"]) else 0,
        ),
        axis=1,
    )

    df["atr"]=df["tr"].rolling(14,min_periods=1).mean()
    df["vol_technical"]=np.log(df["atr"]/df["200_sma"])

    for k in [5,10,21]:
        df[f"fwd_{k}"]=np.log(df["close"].shift(-k)/df["close"])

    df.dropna(inplace=True)
    df.reset_index(drop=True,inplace=True)
    return df


def classify(df_new,hmm):
    result=df_new.copy()

    result["parent_state"]=(result[hmm.parent_col] > hmm.parent_layer.cutoffs[0]).astype(int)
    result["child_state"]=-1

    for parent_state,child_layer in hmm.child_layers.items():
        mask=result["parent_state"]==parent_state
        result.loc[mask,"child_state"]=(
            result.loc[mask,hmm.child_col] > child_layer.cutoffs[0]
        ).astype(int)

    return result


def joint_states(df_classified,hmm):
    trend_states=df_classified["parent_state"] if hmm.parent_col==TREND_COL else df_classified["child_state"]
    vol_states=df_classified["child_state"] if hmm.parent_col==TREND_COL else df_classified["parent_state"]

    if hmm.parent_col==TREND_COL:
        return [f"{'H' if v==1 else 'L'}({'D' if t==0 else 'U'})" for t,v in zip(trend_states,vol_states)]
    else:
        return [f"{'D' if t==0 else 'U'}({'H' if v==1 else 'L'})" for t,v in zip(trend_states,vol_states)]


def build_agreement_matrix(tech_df,quant_df,tech_hmm,quant_hmm,label):
    tech_states=joint_states(tech_df,tech_hmm)
    quant_states=joint_states(quant_df,quant_hmm)

    matrix=pd.DataFrame(0.0,index=TECH_ORDER,columns=QUANT_ORDER)
    total=len(tech_states)

    for ts,qs in zip(tech_states,quant_states):
        if ts in TECH_ORDER and qs in QUANT_ORDER:
            matrix.loc[ts,qs]+=1

    matrix/=total

    w=62
    print("\n  " + "-"*w)
    print("  " + ("  Agreement Matrix - "+label).center(w))
    print("  " + "-"*w)
    print("  " + f"{'Technical / Quant':22}", *[f"{c:>9}" for c in QUANT_ORDER])
    print("  " + "-"*w)

    for row in TECH_ORDER:
        print("  " + f"{row:22}", *[f"{matrix.loc[row,c]:>9.2%}" for c in QUANT_ORDER])

    print("  " + "-"*w)

    diagonal=sum(matrix.loc[tr,qc] for tr,qc in SEMANTIC)
    print("  " + f"{'Total Agreement':22}{diagonal:>9.2%}")
    print("  " + "-"*w)

    return matrix,tech_states,quant_states


def predictive_analysis(tech_states,quant_states,df_ref,label):
    records=[]

    for i,(ts,qs) in enumerate(zip(tech_states,quant_states)):
        if ts not in TECH_ORDER or qs not in QUANT_ORDER:
            continue

        row={"tech":ts,"quant":qs,"agreement":(ts,qs) in SEMANTIC}

        for k in [5,10,21]:
            col=f"fwd_{k}"
            if col in df_ref.columns:
                row[col]=df_ref.iloc[i][col]

        records.append(row)

    results=pd.DataFrame(records)

    w=62
    print("\n  " + "-"*w)
    print("  " + ("  Predictive Analysis - "+label).center(w))
    print("  " + "-"*w)
    print("  " + f"{'Cell':22}{'N':>6}{'Fwd 5d':>10}{'Fwd 10d':>10}{'Fwd 21d':>10}")
    print("  " + "-"*w)

    for ts in TECH_ORDER:
        for qs in QUANT_ORDER:
            subset=results[(results["tech"]==ts)&(results["quant"]==qs)]
            if len(subset)==0:
                continue

            tag="Y" if (ts,qs) in SEMANTIC else "N"
            fwds=[f"{subset[f'fwd_{k}'].mean():>10.2%}" for k in [5,10,21]]
            print("  " + f"{tag+' '+ts+' / '+qs:22}{len(subset):>6}" + "".join(fwds))

    print("  " + "-"*w)

    agree=results[results["agreement"]==True]
    disagree=results[results["agreement"]==False]

    print("  " + f"{'Agreement avg fwd 21d':30}{agree['fwd_21'].mean():>10.2%}")
    print("  " + f"{'Disagreement avg fwd 21d':30}{disagree['fwd_21'].mean():>10.2%}")
    print("  " + "-"*w)

    return records


all_results=[]

for symbol in SYMBOLS:
    print("\n" + "="*62)
    print("  " + symbol)
    print("="*62)

    filename=f"./cache/{symbol}-{end_date}.csv"

    if not os.path.exists(filename):
        print("File does not exist - fetching from API...")
        with DataLoader(api_key=api_key) as loader:
            raw=loader.load_ticker(symbol=symbol,start=start_date,end=end_date)
            raw.to_csv(filename)
        print("Sleeping 145s to respect API limits...")
        time.sleep(145)
    else:
        print("File exists - loading from cache.")

    df=pd.read_csv(filename)
    df=compute_features(df)

    split_idx=int(len(df)*0.8)
    df_train=df.iloc[:split_idx].copy()
    df_test=df.iloc[split_idx:].copy()

    print(f"Train rows : {len(df_train)}  |  Test rows : {len(df_test)}")

    tech_hmm=HierarchicalHMM(df=df_train,parent_col="trend",child_col="vol_technical")
    quant_hmm=HierarchicalHMM(df=df_train,parent_col="vol_quant",child_col="trend")

    tech_hmm.fit_parent()
    tech_hmm.fit_children()
    tech_hmm.print_parent()
    tech_hmm.print_children()

    quant_hmm.fit_parent()
    quant_hmm.fit_children()
    quant_hmm.print_parent()
    quant_hmm.print_children()

    df_test_tech=classify(df_test,tech_hmm)
    df_test_quant=classify(df_test,quant_hmm)

    train_tech_states=joint_states(tech_hmm.df,tech_hmm)
    train_quant_states=joint_states(quant_hmm.df,quant_hmm)

    build_agreement_matrix(tech_hmm.df,quant_hmm.df,tech_hmm,quant_hmm,f"{symbol} - Train Set")
    build_agreement_matrix(df_test_tech,df_test_quant,tech_hmm,quant_hmm,f"{symbol} - Test Set")

    train_records=predictive_analysis(train_tech_states,train_quant_states,df_train,f"{symbol} - Train Set")
    test_records=predictive_analysis(
        joint_states(df_test_tech,tech_hmm),
        joint_states(df_test_quant,quant_hmm),
        df_test,
        f"{symbol} - Test Set"
    )

    for split,records in [("train",train_records),("test",test_records)]:
        for r in records:
            all_results.append({
                "symbol":symbol,
                "split":split,
                "tech":r["tech"],
                "quant":r["quant"],
                "agreement":r["agreement"],
                "fwd_5":r.get("fwd_5",np.nan),
                "fwd_10":r.get("fwd_10",np.nan),
                "fwd_21":r.get("fwd_21",np.nan),
            })


summary=pd.DataFrame(all_results)
test_summary=summary[summary["split"]=="test"]

grouped=(
    test_summary
    .groupby(["tech","quant","agreement"])
    .agg(total_n=("fwd_21","count"),mean_fwd21=("fwd_21","mean"))
    .reset_index()
    .sort_values("mean_fwd21",ascending=False)
)

w=72
print("\n" + "="*w)
print("  " + "Cross-Ticker Summary - Test Set".center(w))
print("="*w)
print("  " + f"{'Cell':25}{'Agree':>8}{'N':>8}{'Fwd 21d':>10}")
print("  " + "-"*54)

for _,row in grouped.iterrows():
    tag="Y" if row["agreement"] else "N"
    print("  " + f"{tag+' '+row['tech']+' / '+row['quant']:25}{'Yes' if row['agreement'] else 'No':>8}{int(row['total_n']):>8}{row['mean_fwd21']:>10.2%}")

print("  " + "-"*54)

agree_avg=test_summary[test_summary["agreement"]==True]["fwd_21"].mean()
disagree_avg=test_summary[test_summary["agreement"]==False]["fwd_21"].mean()

print("  " + f"{'Agreement avg fwd 21d':33}{agree_avg:>10.2%}")
print("  " + f"{'Disagreement avg fwd 21d':33}{disagree_avg:>10.2%}")
print("  " + "-"*54)

print("\n" + "="*w)
print("  " + "Cross-Ticker Summary - By Cell Type".center(w))
print("="*w)

cell_grouped=(
    test_summary
    .groupby(["tech","quant"])
    .agg(
        total_n=("fwd_21","count"),
        mean_fwd_5=("fwd_5","mean"),
        mean_fwd_10=("fwd_10","mean"),
        mean_fwd_21=("fwd_21","mean"),
    )
    .reset_index()
    .sort_values("mean_fwd_21",ascending=False)
)

print("  " + f"{'Cell':22}{'Agree':>7}{'N':>7}{'Fwd 5d':>10}{'Fwd 10d':>10}{'Fwd 21d':>10}")
print("  " + "-"*68)

for _,row in cell_grouped.iterrows():
    ts=row["tech"]
    qs=row["quant"]
    tag="Y" if (ts,qs) in SEMANTIC else "N"

    print(
        "  " +
        f"{tag+' '+ts+' / '+qs:22}"
        f"{'Yes' if (ts,qs) in SEMANTIC else 'No':>7}"
        f"{int(row['total_n']):>7}"
        f"{row['mean_fwd_5']:>10.2%}"
        f"{row['mean_fwd_10']:>10.2%}"
        f"{row['mean_fwd_21']:>10.2%}"
    )

print("  " + "-"*68)