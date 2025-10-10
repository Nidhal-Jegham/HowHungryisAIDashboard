import os
import re
import pandas as pd
import numpy as np
from datetime import datetime


def harmonize_benchmark_names(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for c in df.columns:
        col = c.strip()
        col_norm = (
            col.replace("𝜏", "Tau")
               .replace("τ", "Tau")
               .replace("Tau2", "Tau²")
               .replace("Tau 2", "Tau²")
               .replace("Tau^2", "Tau²")
        )
        if re.search(r"Terminal[- ]?Bench", col_norm, re.I):
            if "Agentic Terminal Use" in col_norm or "AgenticCoding" in col_norm:
                rename_map[c] = "Terminal-Bench Hard (Agentic Coding & Terminal Use)"
        elif re.search(r"Tau²[- ]?Bench", col_norm, re.I):
            rename_map[c] = "Tau²-Bench Telecom (Agentic Tool Use)"
        elif re.search(r"Artificial Analysis", col_norm, re.I):
            rename_map[c] = "Artificial Analysis Intelligence Index"
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"Renamed columns: {rename_map}")
    return df


DATA_DIR = "./output"  

pattern = re.compile(r"artificialanalysis_environmental_(\d{4}-\d{2}-\d{2})\.csv$")

files_with_dates = [f for f in os.listdir(DATA_DIR) if pattern.match(f)]
if not files_with_dates:
    raise FileNotFoundError(f"No files matching pattern in {DATA_DIR}")

files_with_dates.sort(key=lambda x: datetime.strptime(pattern.search(x).group(1), "%Y-%m-%d"))

dfs = {}
for fname in files_with_dates:
    date_str = pattern.search(fname).group(1)
    fp = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(fp)
    df = harmonize_benchmark_names(df)
    dfs[date_str] = df
    print(f"Loaded {fname} → df_{date_str.replace('-', '_')}  ({df.shape[0]} rows, {df.shape[1]} cols)")

for date_str, df in dfs.items():
    globals()[f"df_{date_str.replace('-', '_')}"] = df

print(f"\nLoaded {len(dfs)} dated datasets successfully.\n")

schemas = {date: sorted(df.columns.tolist()) for date, df in dfs.items()}
first_date = list(schemas.keys())[0]
first_schema = schemas[first_date]
differences = {}
for date, cols in schemas.items():
    if cols != first_schema:
        diff = sorted(set(cols).symmetric_difference(first_schema))
        differences[date] = diff

if not differences:
    print("All datasets have identical columns.")
else:
    print("Schema differences found:")
    for date, diff in differences.items():
        print(f"  {date}: {diff}")

aligned = []
for date_str, df in dfs.items():
    df = df.copy()
    df["SnapshotDate"] = pd.to_datetime(date_str)
    aligned.append(df)
df_all = pd.concat(aligned, ignore_index=True)
print(f"\nConcatenated df_all shape: {df_all.shape}")

def tag_ctx(model_col, ctx_col, pattern_model):
    is_non_azure = model_col.str.contains(pattern_model, case=False, na=False)
    already_tagged = model_col.str.contains(r"\[\s*\d+\s*k\s*\]", case=False, na=False)
    has_ctx = ctx_col.isin(["128k", "64k"])
    mask = is_non_azure & ~already_tagged & has_ctx
    return mask

mask_r1 = tag_ctx(df_all["Model"], df_all.get("ContextWindow", pd.Series(index=df_all.index, dtype=str)), r"^DeepSeek\s*R1\s*\(DeepSeek\)")
df_all.loc[mask_r1, "Model"] = df_all.loc[mask_r1].apply(lambda r: f"{r['Model']} [{r['ContextWindow']}]", axis=1)

mask_v3 = tag_ctx(df_all["Model"], df_all.get("ContextWindow", pd.Series(index=df_all.index, dtype=str)), r"^DeepSeek\s*V3\s*\(DeepSeek\)")
df_all.loc[mask_v3, "Model"] = df_all.loc[mask_v3].apply(lambda r: f"{r['Model']} [{r['ContextWindow']}]", axis=1)

keep = [
    "SnapshotDate","Model","ContextWindow","API ID","length","Query Length",
    "MedianTokens/s","P5Tokens/s","P25Tokens/s","P75Tokens/s","P95Tokens/s",
    "MedianFirst Chunk (s)","First AnswerToken (s)",
    "P5First Chunk (s)","P25First Chunk (s)","P75First Chunk (s)","P95First Chunk (s)",
    "Hardware","Host","GPUs Power Draw","Non-GPUs Power Draw",
    "Min GPU Power Utilization","Max GPU Power Utilization","Non-GPU Power Utilization",
    "PUE","WUE (Site)","WUE (Source)","CIF","Company","Size",
    "Mean Max Energy (Wh)","Std Max Energy (Wh)",
    "Mean Min Energy (Wh)","Std Min Energy (Wh)",
    "Mean Combined Energy (Wh)","Std Combined Energy (Wh)",
    "Mean Max Carbon (gCO2e)","Std Max Carbon (gCO2e)",
    "Mean Min Carbon (gCO2e)","Std Min Carbon (gCO2e)",
    "Mean Combined Carbon (gCO2e)","Std Combined Carbon (gCO2e)",
    "Mean Max Water (Site, mL)","Std Max Water (Site, mL)",
    "Mean Min Water (Site, mL)","Std Min Water (Site, mL)",
    "Mean Combined Water (Site, mL)","Std Combined Water (Site, mL)",
    "Mean Max Water (Source, mL)","Std Max Water (Source, mL)",
    "Mean Min Water (Source, mL)","Std Min Water (Source, mL)",
    "Mean Combined Water (Source, mL)","Std Combined Water (Source, mL)",
    "Mean Max Water (Site & Source, mL)","Std Max Water (Site & Source, mL)",
    "Mean Min Water (Site & Source, mL)","Std Min Water (Site & Source, mL)",
    "Mean Combined Water (Site & Source, mL)","Std Combined Water (Site & Source, mL)",
]
present = [c for c in keep if c in df_all.columns]
df_all = df_all[present].copy()
df_all = df_all.sort_values(["SnapshotDate","Model"], kind="mergesort").reset_index(drop=True)

group_keys = [
    "Model", "Query Length", "API ID", "Company", "Hardware", "Host",
    "GPUs Power Draw", "Non-GPUs Power Draw", "Min GPU Power Utilization",
    "Max GPU Power Utilization", "Non-GPU Power Utilization", "PUE",
    "WUE (Site)", "WUE (Source)", "CIF", "Size"
]

model_avg = (
    df_all.groupby(group_keys)
    .agg({
        "Mean Max Energy (Wh)": ["mean", "std"],
        "Std Max Energy (Wh)": ["mean"],
        "Mean Min Energy (Wh)": ["mean", "std"],
        "Std Min Energy (Wh)": ["mean"],
        "Mean Combined Energy (Wh)": ["mean", "std"],
        "Std Combined Energy (Wh)": ["mean"],
        "Mean Max Carbon (gCO2e)": ["mean", "std"],
        "Std Max Carbon (gCO2e)": ["mean"],
        "Mean Min Carbon (gCO2e)": ["mean", "std"],
        "Std Min Carbon (gCO2e)": ["mean"],
        "Mean Combined Carbon (gCO2e)": ["mean", "std"],
        "Std Combined Carbon (gCO2e)": ["mean"],
        "Mean Max Water (Site & Source, mL)": ["mean", "std"],
        "Std Max Water (Site & Source, mL)": ["mean"],
        "Mean Min Water (Site & Source, mL)": ["mean", "std"],
        "Std Min Water (Site & Source, mL)": ["mean"],
        "Mean Combined Water (Site & Source, mL)": ["mean", "std"],
        "Std Combined Water (Site & Source, mL)": ["mean"],
    })
)
model_avg.columns = [" ".join(col).strip() for col in model_avg.columns]
model_avg = model_avg.reset_index()

def pooled_std(group: pd.DataFrame, mean_col: str, std_col: str) -> float:
    mu = group[mean_col].dropna()
    si = group[std_col].dropna()
    if len(mu) < 2 or len(si) == 0:
        return np.nan
    within = np.mean(si**2)
    between = np.var(mu, ddof=1)
    return float(np.sqrt(within + between))

pooled_specs = [
    ("Pooled Std Max Energy (Wh)", "Mean Max Energy (Wh)", "Std Max Energy (Wh)"),
    ("Pooled Std Min Energy (Wh)", "Mean Min Energy (Wh)", "Std Min Energy (Wh)"),
    ("Pooled Std Average Energy (Wh)", "Mean Combined Energy (Wh)", "Std Combined Energy (Wh)"),
    ("Pooled Std Max Carbon (gCO2e)", "Mean Max Carbon (gCO2e)", "Std Max Carbon (gCO2e)"),
    ("Pooled Std Min Carbon (gCO2e)", "Mean Min Carbon (gCO2e)", "Std Min Carbon (gCO2e)"),
    ("Pooled Std Average Carbon (gCO2e)", "Mean Combined Carbon (gCO2e)", "Std Combined Carbon (gCO2e)"),
    ("Pooled Std Max Water (mL)", "Mean Max Water (Site & Source, mL)", "Std Max Water (Site & Source, mL)"),
    ("Pooled Std Min Water (mL)", "Mean Min Water (Site & Source, mL)", "Std Min Water (Site & Source, mL)"),
    ("Pooled Std Average Water (mL)", "Mean Combined Water (Site & Source, mL)", "Std Combined Water (Site & Source, mL)"),
]

pooled_frames = []
for out_col, mean_col, std_col in pooled_specs:
    s = (
        df_all.groupby(group_keys)
        .apply(lambda g: pooled_std(g, mean_col, std_col))
        .rename(out_col)
        .reset_index()
    )
    pooled_frames.append(s)

pooled_all = pooled_frames[0]
for s in pooled_frames[1:]:
    pooled_all = pooled_all.merge(s, on=group_keys, how="outer")

rename_map_main = {
    "Mean Max Energy (Wh) mean": "Avg Daily Max Energy (Wh)",
    "Mean Max Energy (Wh) std": "Day-to-day Std of Daily Max Energy (Wh)",
    "Std Max Energy (Wh) mean": "Avg Intra-day Std (Max Energy, Wh)",
    "Mean Min Energy (Wh) mean": "Avg Daily Min Energy (Wh)",
    "Mean Min Energy (Wh) std": "Day-to-day Std of Daily Min Energy (Wh)",
    "Std Min Energy (Wh) mean": "Avg Intra-day Std (Min Energy, Wh)",
    "Mean Combined Energy (Wh) mean": "Avg Daily Average Energy (Wh)",
    "Mean Combined Energy (Wh) std": "Day-to-day Std of Daily Average Energy (Wh)",
    "Std Combined Energy (Wh) mean": "Avg Intra-day Std (Average Energy, Wh)",
    "Mean Max Carbon (gCO2e) mean": "Avg Daily Max Carbon (gCO2e)",
    "Mean Max Carbon (gCO2e) std": "Day-to-day Std of Daily Max Carbon (gCO2e)",
    "Std Max Carbon (gCO2e) mean": "Avg Intra-day Std (Max Carbon, gCO2e)",
    "Mean Min Carbon (gCO2e) mean": "Avg Daily Min Carbon (gCO2e)",
    "Mean Min Carbon (gCO2e) std": "Day-to-day Std of Daily Min Carbon (gCO2e)",
    "Std Min Carbon (gCO2e) mean": "Avg Intra-day Std (Min Carbon, gCO2e)",
    "Mean Combined Carbon (gCO2e) mean": "Avg Daily Average Carbon (gCO2e)",
    "Mean Combined Carbon (gCO2e) std": "Day-to-day Std of Daily Average Carbon (gCO2e)",
    "Std Combined Carbon (gCO2e) mean": "Avg Intra-day Std (Average Carbon, gCO2e)",
    "Mean Max Water (Site & Source, mL) mean": "Avg Daily Max Water (mL)",
    "Mean Max Water (Site & Source, mL) std": "Day-to-day Std of Daily Max Water (mL)",
    "Std Max Water (Site & Source, mL) mean": "Avg Intra-day Std (Max Water, mL)",
    "Mean Min Water (Site & Source, mL) mean": "Avg Daily Min Water (mL)",
    "Mean Min Water (Site & Source, mL) std": "Day-to-day Std of Daily Min Water (mL)",
    "Std Min Water (Site & Source, mL) mean": "Avg Intra-day Std (Min Water, mL)",
    "Mean Combined Water (Site & Source, mL) mean": "Avg Daily Average Water (mL)",
    "Mean Combined Water (Site & Source, mL) std": "Day-to-day Std of Daily Average Water (mL)",
    "Std Combined Water (Site & Source, mL) mean": "Avg Intra-day Std (Average Water, mL)",
}

model_avg_renamed = model_avg.rename(columns=rename_map_main)

out = model_avg_renamed.merge(
    pooled_all,
    on=group_keys,
    how="left",
    validate="one_to_one"
)

preferred_order = ["Model"] + [c for c in out.columns if c != "Model"]
out = out[preferred_order].sort_values("Model").reset_index(drop=True)

out.to_csv("Monthly LLM Environmental Footprint.csv", index=False)
print('Wrote "Monthly LLM Environmental Footprint.csv"')

rename_map_day2day = {
    "Mean Max Energy (Wh)": "Daily Max Energy (Wh)",
    "Std Max Energy (Wh)": "Intra-day Std (Max Energy, Wh)",
    "Mean Min Energy (Wh)": "Daily Min Energy (Wh)",
    "Std Min Energy (Wh)": "Intra-day Std (Min Energy, Wh)",
    "Mean Combined Energy (Wh)": "Daily Average Energy (Wh)",
    "Std Combined Energy (Wh)": "Intra-day Std (Average Energy, Wh)",
    "Mean Max Carbon (gCO2e)": "Daily Max Carbon (gCO2e)",
    "Std Max Carbon (gCO2e)": "Intra-day Std (Max Carbon, gCO2e)",
    "Mean Min Carbon (gCO2e)": "Daily Min Carbon (gCO2e)",
    "Std Min Carbon (gCO2e)": "Intra-day Std (Min Carbon, gCO2e)",
    "Mean Combined Carbon (gCO2e)": "Daily Average Carbon (gCO2e)",
    "Std Combined Carbon (gCO2e)": "Intra-day Std (Average Carbon, gCO2e)",
    "Mean Max Water (Site & Source, mL)": "Daily Max Water (mL)",
    "Std Max Water (Site & Source, mL)": "Intra-day Std (Max Water, mL)",
    "Mean Min Water (Site & Source, mL)": "Daily Min Water (mL)",
    "Std Min Water (Site & Source, mL)": "Intra-day Std (Min Water, mL)",
    "Mean Combined Water (Site & Source, mL)": "Daily Average Water (mL)",
    "Std Combined Water (Site & Source, mL)": "Intra-day Std (Average Water, mL)",
}
df_all_day2day = df_all.rename(columns=rename_map_day2day)
df_all_day2day.to_csv("Monthly LLM Environmental Footprint Day2Day.csv", index=False)
print('Wrote "Monthly LLM Environmental Footprint Day2Day.csv"')
