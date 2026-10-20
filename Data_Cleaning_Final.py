import re
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


# header cleanup
# the scraper changes out headers with random spacing/newlines depending on the day,
# so we map every variant we've seen back to one canonical name

CANON = {
    "API ID": ["API ID", "API\nID", "APIID", "API-ID", "API  ID", "API  \nID", "API / ID"],
    "Model": ["Model"],
    "API Provider": ["API Provider", "APIProvider"],
    "Function Calling": ["Function Calling", "FunctionCalling"],
    "JSON Mode": ["JSON Mode", "JSONMode"],
    "License": ["License"],
    "OpenAI Compatible": ["OpenAI Compatible", "OpenAICompatible"],
    "ContextWindow": ["ContextWindow", "Context Window"],
    "Footnotes": ["Footnotes"],
    "FurtherAnalysis": ["FurtherAnalysis", "Further Analysis"],
    "Artificial Analysis Intelligence Index": [
        "Artificial Analysis Intelligence Index", "Artificial AnalysisIntelligence Index",
    ],
    "BlendedUSD/1M Tokens": ["BlendedUSD/1M Tokens", "Blended USD/1M Tokens"],
    "Input PriceUSD/1M Tokens": ["Input PriceUSD/1M Tokens", "InputPriceUSD/1M Tokens", "Input Price USD/1M Tokens"],
    "Output PriceUSD/1M Tokens": ["Output PriceUSD/1M Tokens", "OutputPriceUSD/1M Tokens", "Output Price USD/1M Tokens"],
    "MedianTokens/s": ["MedianTokens/s", "Median Tokens/s", "Median Token/s"],
    "P5Tokens/s": ["P5Tokens/s", "P5 Tokens/s"],
    "P25Tokens/s": ["P25Tokens/s", "P25 Tokens/s"],
    "P75Tokens/s": ["P75Tokens/s", "P75 Tokens/s"],
    "P95Tokens/s": ["P95Tokens/s", "P95 Tokens/s"],
    "MedianFirst Chunk (s)": ["MedianFirst Chunk (s)", "Median First Chunk (s)", "MedianFirstChunk (s)"],
    "First AnswerToken (s)": ["First AnswerToken (s)", "FirstAnswerToken (s)", "First Answer Token (s)"],
    "P5First Chunk (s)": ["P5First Chunk (s)", "P5 First Chunk (s)", "P5FirstChunk (s)"],
    "P25First Chunk (s)": ["P25First Chunk (s)", "P25 First Chunk (s)", "P25FirstChunk (s)"],
    "P75First Chunk (s)": ["P75First Chunk (s)", "P75 First Chunk (s)", "P75FirstChunk (s)"],
    "P95First Chunk (s)": ["P95First Chunk (s)", "P95 First Chunk (s)", "P95FirstChunk (s)"],
    "TotalResponse (s)": ["TotalResponse (s)", "Total Response (s)"],
    "ReasoningTime (s)": ["ReasoningTime (s)", "Reasoning Time (s)"],
    "MMLU-Pro (Reasoning & Knowledge)": [
        "MMLU-Pro(Reasoning &Knowledge)", "MMLU-Pro(Reasoning & Knowledge)",
        "MMLU Pro (Reasoning & Knowledge)", "MMLU-Pro",
    ],
    "GPQA Diamond (Scientific Reasoning)": [
        "GPQA Diamond(ScientificReasoning)", "GPQA Diamond(Scientific Reasoning)", "GPQA Diamond", "GPQA-Diamond",
    ],
    "Humanity's Last Exam (Reasoning & Knowledge)": [
        "Humanity's LastExam(Reasoning & Knowledge)", "Humanity's Last Exam(Reasoning & Knowledge)",
        "Humanitys Last Exam (Reasoning & Knowledge)",
    ],
    "LiveCodeBench (Coding)": ["LiveCodeBench(Coding)", "LiveCodeBench"],
    "SciCode (Coding)": ["SciCode(Coding)", "SciCode"],
    "HumanEval (Coding)": ["HumanEval(Coding)", "HumanEval"],
    "AIME 2025 (Competition Math)": ["AIME 2025(CompetitionMath)", "AIME 2025"],
    "AIME 2024 (Competition Math)": ["AIME 2024(CompetitionMath)", "AIME 2024"],
    "Math 500 (Competition Math)": ["Math 500(CompetitionMath)", "Math 500"],
    "IFBench (Instruction Following)": ["IFBench(InstructionFollowing)", "IFBench"],
    "Terminal-Bench Hard (Agentic Coding & Terminal Use)": [
        "Terminal-BenchHard (AgenticCoding & Terminal Use)",
        "Terminal-Bench Hard (Agentic Coding & Terminal Use)", "Terminal Bench Hard",
    ],
    "Tau2-Bench Telecom (Agentic Tool Use)": [
        "𝜏²-BenchTelecom(Agentic Tool Use)", "Tau^2-BenchTelecom(Agentic Tool Use)",
        "Tau2-Bench Telecom (Agentic Tool Use)", "Tau2-BenchTelecom",
    ],
    "AA-LCR (Long Context Reasoning)": [
        "AA-LCR(LongContext Reasoning)", "AA-LCR (LongContext Reasoning)", "AA-LCR", "AA-LCR (Long Context Reasoning)",
    ],
    "Chatbot Arena": ["ChatbotArena", "Chatbot Arena"],
    "length": ["length"],
    "Query Length": ["Query Length", "QueryLength"],
}


def _norm(s):
    s = s.lower().replace("\xa0", " ").replace("\u200b", "")
    return re.sub(r"[^a-z0-9]", "", s)


CANON_LOOKUP = {_norm(v): canon for canon, variants in CANON.items() for v in variants}


def canonicalize_headers(df):
    fixed = []
    for c in df.columns:
        key = _norm(c)
        if key in CANON_LOOKUP:
            fixed.append(CANON_LOOKUP[key])
        else:
            c2 = c.replace("\xa0", " ").replace("\u200b", "").replace("–", "-")
            fixed.append(re.sub(r"\s+", " ", c2).strip())
    df.columns = fixed

    if "API ID" not in df.columns and {"API", "ID"}.issubset(df.columns):
        df["API ID"] = df["API"].astype(str) + " " + df["ID"].astype(str)
        df.drop(columns=["API", "ID"], inplace=True)

    return df



RUNS = [
    ("./data/artificialanalysis_cleanshort.csv", "Short", 300),
    ("./data/artificialanalysis_cleanmedium.csv", "Medium", 1000),
    ("./data/artificialanalysis_cleanlong.csv", "Long", 1500),
]

frames = []
for path, label, out_tokens in RUNS:
    d = pd.read_csv(path)
    d["length"] = label
    d["Query Length"] = out_tokens
    d = canonicalize_headers(d)
    if {"API ID", "Model"}.issubset(d.columns):
        d = d.drop_duplicates(subset=["API ID", "Model"])
    else:
        print(f"[{label}] missing API ID or Model. Columns: {list(d.columns)}")
    frames.append(d)

df = pd.concat(frames, ignore_index=True)


# model groups by provider/deployment

OPENAI_NEW = {
    "gpt-5-mini-2025-08-07", "gpt-5-nano-2025-08-07", "gpt-5-2025-08-07",
    "o3-2025-04-16", "o4-mini-2025-04-16", "o3-mini", "o1-2024-12-17", "o1-mini-2024-09-12",
    "gpt-4.1-mini-2025-04-14", "gpt-4.1-2025-04-14", "chatgpt-4o-latest", "gpt-4o-2024-11-20",
    "gpt-4.1-nano-2025-04-14", "gpt-4o", "o3-pro-2025-06-10", "o1-preview-2024-09-12", "gpt-4o-2024-08-06",
}
OPENAI_OLD = {"gpt-4o-mini-2024-07-18", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4"}

CLAUDE = {
    "claude-opus-4-20250514", "claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-5-haiku-20241022",
    "claude-3-haiku-20240307", "claude-opus-4-1-20250805", "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250929",
}

GOOGLE = {"gemini-2.5-pro", "google/gemini-2.5-flash"}

LLAMA = {
    "us.meta.llama4-maverick-17b-instruct-v1:0", "us.meta.llama4-scout-17b-instruct-v1:0",
    "us.meta.llama3-3-70b-instruct-v1:0", "us.meta.llama3-2-90b-instruct-v1:0",
    "meta.llama3-70b-instruct-v1:0", "meta.llama3-1-8b-instruct-v1:0", "meta.llama3-8b-instruct-v1:0",
    "us.meta.llama3-2-3b-instruct-v1:0", "us.meta.llama3-2-1b-instruct-v1:0",
    "us.meta.llama3-2-11b-instruct-v1:0", "us.meta.llama3-1-405b-instruct-v1:0",
    "us.meta.llama3-1-70b-instruct-v1:0", "meta.llama3-1-405b-instruct-v1:0", "meta.llama3-1-70b-instruct-v1:0",
}

DEEPSEEK = {"deepseek-reasoner", "deepseek-chat"}
DEEPSEEK_AZURE = {"DeepSeek-V3-0324", "DeepSeek-R1-0528"}
GROK = {"grok-4-0709", "grok-3-mini-beta", "grok-3-mini-fast-beta", "grok-3-beta", "grok-3-fast-beta"}
MISTRAL_AZURE = {"mistral-medium-2505", "Mistral-Large-2411", "Mistral-small", "Mistral-Nemo"}
MISTRAL_AWS = {"mistral.mistral-large-2407-v1:0"}
MOONSHOT_AWS = {"moonshot.kimi-k2-thinking"}


# datacenter multipliers per host: PUE, WUE site (L/kWh), WUE source (L/kWh), CIF (kgCO2e/kWh)
AZURE_ENV = (1.12, 0.3, 4.35, 0.34)
AWS_ENV = (1.14, 0.18, 5.12, 0.3)
GOOGLE_ENV = (1.09, 0.3, 1.1, 0.231)
DEEPSEEK_ENV = (1.27, 1.2, 6.016, 0.6)
XAI_ENV = (1.5, 0.36, 3.142, 0.385)

PROVIDERS = [
    (OPENAI_NEW, "OpenAI", "DGX H200/H100", "Azure", AZURE_ENV),
    (OPENAI_OLD, "OpenAI", "DGX A100", "Azure", AZURE_ENV),
    (CLAUDE, "Anthropic", "DGX H200/H100", "Anthropic", AWS_ENV),
    (GOOGLE, "Google", "TPU V6e", "Google", GOOGLE_ENV),
    (LLAMA, "Meta", "DGX H200/H100", "AWS", AWS_ENV),
    (DEEPSEEK, "DeepSeek", "DGX H800", "DeepSeek", DEEPSEEK_ENV),
    (DEEPSEEK_AZURE, "DeepSeek", "DGX H200/H100", "Azure", AZURE_ENV),
    (GROK, "xAI", "DGX H200/H100", "xAI", XAI_ENV),
    (MISTRAL_AZURE, "Mistral AI", "DGX H200/H100", "Azure", AZURE_ENV),
    (MISTRAL_AWS, "Mistral AI", "DGX H200/H100", "AWS", AWS_ENV),
    (MOONSHOT_AWS, "Moonshot", "DGX H200/H100", "AWS", AWS_ENV),
]

LARGE = {
    "moonshot.kimi-k2-thinking", "claude-opus-4-1-20250805", "claude-haiku-4-5-20251001",
    "claude-sonnet-4-20250514", "claude-sonnet-4-5-20250929", "claude-opus-4-20250514",
    "mistral.mistral-large-2407-v1:0", "gemini-2.5-pro", "google/gemini-2.5-flash",
    "mistral-medium-2505", "Mistral-Large-2411", "DeepSeek-V3-0324", "gpt-5-2025-08-07",
    "DeepSeek-R1-0528", "deepseek-reasoner", "deepseek-chat", "us.meta.llama3-2-90b-instruct-v1:0",
    "us.meta.llama3-1-405b-instruct-v1:0", "meta.llama3-1-405b-instruct-v1:0",
    "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229",
    "claude-3-5-haiku-20241022", "claude-3-haiku-20240307", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4",
    "o3-2025-04-16", "o1-2024-12-17", "gpt-4.1-2025-04-14", "chatgpt-4o-latest", "gpt-4o-2024-11-20",
    "gpt-4o", "o3-pro-2025-06-10", "o1-preview-2024-09-12", "gpt-4o-2024-08-06",
    "grok-4-0709", "grok-3-beta", "grok-3-fast-beta",
}
MEDIUM = {
    "gpt-5-mini-2025-08-07", "o4-mini-2025-04-16", "o3-mini", "o1-mini-2024-09-12",
    "gpt-4.1-mini-2025-04-14", "gpt-4o-mini-2024-07-18", "us.meta.llama3-3-70b-instruct-v1:0",
    "meta.llama3-70b-instruct-v1:0", "us.meta.llama3-1-70b-instruct-v1:0", "meta.llama3-1-70b-instruct-v1:0",
    "grok-3-mini-beta", "grok-3-mini-fast-beta",
}
SMALL = {"Mistral-small", "Mistral-Nemo", "gpt-5-nano-2025-08-07", "gpt-4.1-nano-2025-04-14"}
MICRO = {
    "us.meta.llama4-maverick-17b-instruct-v1:0", "us.meta.llama4-scout-17b-instruct-v1:0",
    "meta.llama3-1-8b-instruct-v1:0", "meta.llama3-8b-instruct-v1:0", "us.meta.llama3-2-11b-instruct-v1:0",
}
NANO = {"us.meta.llama3-2-3b-instruct-v1:0", "us.meta.llama3-2-1b-instruct-v1:0"}

SIZES = [("Large", LARGE), ("Medium", MEDIUM), ("Small", SMALL), ("Micro", MICRO), ("Nano", NANO)]

POWER = {
    "DGX H200/H100": (5.6, 4.6),
    "DGX H800": (5.6, 4.6),
    "DGX A100": (3.2, 3.3),
    "TPU V6e": (1.2, 0.96),
}

HW_FAMILY = {"DGX H200/H100": "hopper", "DGX H800": "hopper", "DGX A100": "a100", "TPU V6e": "tpu"}
UTILIZATION = {
    ("Large", "hopper"): (0.055, 0.075, 0.0625),
    ("Medium", "hopper"): (0.03, 0.045, 0.03125),
    ("Small", "hopper"): (0.017, 0.025, 0.016),
    ("Micro", "hopper"): (0.0075, 0.0125, 0.0087),
    ("Nano", "hopper"): (0.0055, 0.01, 0.0087),
    ("Large", "tpu"): (0.1, 0.1125, 0.1),
    ("Large", "a100"): (0.125, 0.15, 0.0625),
    ("Medium", "a100"): (0.0625, 0.07, 0.0312),
}


all_ids = set().union(*(p[0] for p in PROVIDERS))
df_selected = df[df["API ID"].isin(all_ids)].copy()

LAT_COLS = ["P5First Chunk (s)", "P25First Chunk (s)", "MedianFirst Chunk (s)", "P75First Chunk (s)", "P95First Chunk (s)"]
TPS_COLS = ["P5Tokens/s", "P25Tokens/s", "MedianTokens/s", "P75Tokens/s", "P95Tokens/s"]
for c in LAT_COLS + TPS_COLS + ["First AnswerToken (s)"]:
    df_selected[c] = pd.to_numeric(df_selected[c], errors="coerce")


# reasoning models only report first chunk before thinking, so we push the latency
# quantiles out by the thinking time (first answer token minus first chunk)

REASONING_IDS = ["moonshot.kimi-k2-thinking", "deepseek-reasoner", "DeepSeek-R1-0528"]
REASONING_SPREAD = {
    "P5First Chunk (s)": 0.1,
    "P25First Chunk (s)": 0.5,
    "MedianFirst Chunk (s)": 1.0,
    "P75First Chunk (s)": 1.5,
    "P95First Chunk (s)": 1.9,
}
for api in REASONING_IDS:
    m = df_selected["API ID"] == api
    reasoning_time = df_selected.loc[m, "First AnswerToken (s)"] - df_selected.loc[m, "MedianFirst Chunk (s)"]
    for col, k in REASONING_SPREAD.items():
        df_selected.loc[m, col] += reasoning_time * k


RENAME = {
    "DeepSeek-R1-0528": "DeepSeek R1 (Azure)",
    "deepseek-chat": "DeepSeek V3 (DeepSeek)",
    "DeepSeek-V3-0324": "DeepSeek V3 (Azure)",
    "mistral.mistral-large-2407-v1:0": "Mistral Large 2 (AWS)",
    "Mistral-Large-2411": "Mistral Large 2 (Azure)",
}
for api, name in RENAME.items():
    df_selected.loc[df_selected["API ID"] == api, "Model"] = name

df_selected.drop(columns=[
    "API Provider", "Function Calling", "JSON Mode", "License", "OpenAI Compatible", "Footnotes",
    "BlendedUSD/1M Tokens", "Input PriceUSD/1M Tokens", "Output PriceUSD/1M Tokens",
    "TotalResponse (s)", "ReasoningTime (s)", "FurtherAnalysis",
], inplace=True, errors="ignore")
df_selected.reset_index(drop=True, inplace=True)



def lookup_provider(api):
    for ids, company, hardware, host, env in PROVIDERS:
        if api in ids:
            return company, hardware, host, env
    return None, None, None, (None, None, None, None)


def lookup_size(api):
    for name, ids in SIZES:
        if api in ids:
            return name
    return None


def model_meta(api):
    company, hardware, host, env = lookup_provider(api)
    size = lookup_size(api)
    gpu_kw, other_kw = POWER.get(hardware, (None, None))
    util = UTILIZATION.get((size, HW_FAMILY.get(hardware)), (None, None, None))
    return pd.Series({
        "Hardware": hardware,
        "Host": host,
        "GPUs Power Draw": gpu_kw,
        "Non-GPUs Power Draw": other_kw,
        "Min GPU Power Utilization": util[0],
        "Max GPU Power Utilization": util[1],
        "Non-GPU Power Utilization": util[2],
        "PUE": env[0],
        "WUE (Site)": env[1],
        "WUE (Source)": env[2],
        "CIF": env[3],
        "Company": company,
        "Size": size,
    })


df_selected = pd.concat([df_selected, df_selected["API ID"].apply(model_meta)], axis=1)


# monte carlo

def _phi(x):
    # standard normal cdf via the abramowitz-stegun erf approximation
    x = np.asarray(x, dtype=float)
    sign = np.sign(x)
    x = np.abs(x) / np.sqrt(2.0)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    t = 1.0 / (1.0 + 0.3275911 * x)
    erf = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return 0.5 * (1.0 + sign * erf)


QS = [0.05, 0.25, 0.50, 0.75, 0.95]

METRIC_LABELS = [
    ("energy", "Energy (Wh)"),
    ("carbon", "Carbon (gCO2e)"),
    ("water_site", "Water (Site, mL)"),
    ("water_source", "Water (Source, mL)"),
    ("water_comb", "Water (Site & Source, mL)"),
]
VARIANTS = ["Max", "Min", "Combined"]
MC_COLS = [
    f"{stat} {v} {label}"
    for _, label in METRIC_LABELS
    for v in VARIANTS
    for stat in ("Mean", "Std")
]


def compute_environmental_metrics(row, N=10000, rho=-0.30, w_max=0.50, seed=0):
    # same seed for every row on purpose, so differences between models come from
    # their inputs and not from sampling noise
    empty = pd.Series({c: None for c in MC_COLS})
    try:
        output_tokens = float(row["Query Length"])
        lat = [float(row[c]) for c in LAT_COLS]
        tps = [float(row[c]) for c in TPS_COLS]
        gpu_power = float(row["GPUs Power Draw"])
        non_gpu_power = float(row["Non-GPUs Power Draw"])
        max_gpu_util = float(row["Max GPU Power Utilization"])
        min_gpu_util = float(row["Min GPU Power Utilization"])
        non_gpu_util = float(row["Non-GPU Power Utilization"])
        pue = float(row["PUE"])
        cif = float(row["CIF"])
        wue_site = float(row["WUE (Site)"])
        wue_source = float(row["WUE (Source)"])
    except (KeyError, TypeError, ValueError):
        return empty

    lat = np.maximum.accumulate(lat)
    tps = np.maximum.accumulate(tps)

    # we do the monte carlo sampling here
    # draw correlated normals (latency and throughput are negatively correlated, rho),
    # push them through the normal cdf to get uniforms, then map those uniforms onto
    # each metric's empirical quantiles with linear interpolation (gaussian copula basically)
    # note the uniforms get clipped to [p5, p95] so the tails are capped at the reported p5/p95
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(N, 2))
    chol = np.array([[1.0, 0.0], [rho, np.sqrt(max(1 - rho**2, 1e-12))]])
    u = _phi(z @ chol.T)

    L = np.interp(np.clip(u[:, 0], QS[0], QS[-1]), QS, lat)
    R = np.maximum(np.interp(np.clip(u[:, 1], QS[0], QS[-1]), QS, tps), 1e-9)

    # time per query in hours = time to first token + time to generate the output
    time_h = (L + output_tokens / R) / 3600.0

    # kW for the node at low and high utilization
    power = {
        "Min": gpu_power * min_gpu_util + non_gpu_power * non_gpu_util,
        "Max": gpu_power * max_gpu_util + non_gpu_power * non_gpu_util,
    }

    samples = {}
    for v in ("Max", "Min"):
        energy = time_h * power[v] * pue  # kWh at the facility level
        site = (energy / pue) * wue_site  # site water only counts IT energy
        source = energy * wue_source
        samples[v] = {
            "energy": energy,
            "carbon": energy * cif,
            "water_site": site,
            "water_source": source,
            "water_comb": site + source,
        }

    # combined is just a weighted blend of the min and max scenarios
    samples["Combined"] = {
        k: w_max * samples["Max"][k] + (1 - w_max) * samples["Min"][k] for k in samples["Max"]
    }

    # x1000 converts kWh -> Wh, kg -> g, L -> mL
    out = {}
    for key, label in METRIC_LABELS:
        for v in VARIANTS:
            x = samples[v][key] * 1000
            out[f"Mean {v} {label}"] = float(np.mean(x))
            out[f"Std {v} {label}"] = float(np.std(x, ddof=1))
    return pd.Series(out)


df_environmental = pd.concat(
    [df_selected, df_selected.apply(compute_environmental_metrics, axis=1)[MC_COLS]], axis=1
)

df_environmental.loc[df_environmental["Model"] == "DeepSeek R1 0528 (May '25)", "Model"] = "DeepSeek R1 (May 2025)"
df_environmental.loc[df_environmental["Model"] == "DeepSeek V3 0324 (Mar '25)", "Model"] = "DeepSeek V3 (Mar '25)"


# scale up to billions of prompts
# per prompt Wh * 1e9 prompts / 1e6 = MWh, so it's just *1000 per billion (same logic for g -> t and mL -> kL)

SCALES = [1, 50, 100]
for n in SCALES:
    df_environmental[f"Energy Consumption of {n} Billion Prompts (MWh)"] = df_environmental["Mean Combined Energy (Wh)"] * n * 1000
    df_environmental[f"Carbon Emissions of {n} Billion Prompts (TonsCO2e)"] = df_environmental["Mean Combined Carbon (gCO2e)"] * n * 1000
    df_environmental[f"Water Consumption of {n} Billion Prompts (Kiloliter)"] = df_environmental["Mean Combined Water (Site & Source, mL)"] * n * 1000

# real world equivalents
# household 1.095 MWh, university 1202 MWh, person drinking water 1.2 kL/yr,
# olympic pool 2500 kL, gasoline car 4.6 t/yr, atlantic flight 504 t
for n in SCALES:
    energy = df_environmental[f"Energy Consumption of {n} Billion Prompts (MWh)"]
    water = df_environmental[f"Water Consumption of {n} Billion Prompts (Kiloliter)"]
    carbon = df_environmental[f"Carbon Emissions of {n} Billion Prompts (TonsCO2e)"]

    df_environmental[f"Household Energy Equiv. – {n}B Prompts (MWh)"] = energy / 1.0950
    df_environmental[f"University Energy Equiv. – {n}B Prompts (MWh)"] = energy / 1202
    df_environmental[f"People Annual Drinking Water Equiv. – {n}B Prompts (kL)"] = water / 1.2
    df_environmental[f"Olympic Swimming Pools Equiv. – {n}B Prompts (kL)"] = water / 2500
    df_environmental[f"Gasoline Car Equiv. – {n}B Prompts (TonsCO2e)"] = carbon / 4.6
    df_environmental[f"Atlantic Flight Equiv. – {n}B Prompts (TonsCO2e)"] = carbon / 504


# save the latest file plus a dated snapshot so we keep history

df_environmental.to_csv("artificialanalysis_environmental.csv", index=False)

snapshot_date = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
df_snapshot = df_environmental.copy()
df_snapshot.insert(0, "SnapshotDate", snapshot_date)
df_snapshot.to_csv(f"artificialanalysis_environmental_{snapshot_date}.csv", index=False)
