
# In[1]:


import pandas as pd 
import re

import numpy as np 
from datetime import datetime
from zoneinfo import ZoneInfo
import os

# In[2]:

CANON = {
    "API ID": [
        "API ID","API\nID","APIID","API-ID","API  ID","API  \nID","API / ID"
    ],
    "Model": ["Model"],
    "API Provider": ["API Provider","APIProvider"],
    "Function Calling": ["Function Calling","FunctionCalling"],
    "JSON Mode": ["JSON Mode","JSONMode"],
    "License": ["License"],
    "OpenAI Compatible": ["OpenAI Compatible","OpenAICompatible"],
    "ContextWindow": ["ContextWindow","Context Window"],
    "Footnotes": ["Footnotes"],
    "FurtherAnalysis": ["FurtherAnalysis","Further Analysis"],
    "Artificial Analysis Intelligence Index": [
        "Artificial Analysis Intelligence Index","Artificial AnalysisIntelligence Index"
    ],

    "BlendedUSD/1M Tokens": ["BlendedUSD/1M Tokens","Blended USD/1M Tokens"],
    "Input PriceUSD/1M Tokens": [
        "Input PriceUSD/1M Tokens","InputPriceUSD/1M Tokens","Input Price USD/1M Tokens"
    ],
    "Output PriceUSD/1M Tokens": [
        "Output PriceUSD/1M Tokens","OutputPriceUSD/1M Tokens","Output Price USD/1M Tokens"
    ],

    "MedianTokens/s": ["MedianTokens/s","Median Tokens/s","Median Token/s"],
    "P5Tokens/s": ["P5Tokens/s","P5 Tokens/s"],
    "P25Tokens/s": ["P25Tokens/s","P25 Tokens/s"],
    "P75Tokens/s": ["P75Tokens/s","P75 Tokens/s"],
    "P95Tokens/s": ["P95Tokens/s","P95 Tokens/s"],

    "MedianFirst Chunk (s)": ["MedianFirst Chunk (s)","Median First Chunk (s)","MedianFirstChunk (s)"],
    "First AnswerToken (s)": ["First AnswerToken (s)","FirstAnswerToken (s)","First Answer Token (s)"],
    "P5First Chunk (s)": ["P5First Chunk (s)","P5 First Chunk (s)","P5FirstChunk (s)"],
    "P25First Chunk (s)": ["P25First Chunk (s)","P25 First Chunk (s)","P25FirstChunk (s)"],
    "P75First Chunk (s)": ["P75First Chunk (s)","P75 First Chunk (s)","P75FirstChunk (s)"],
    "P95First Chunk (s)": ["P95First Chunk (s)","P95 First Chunk (s)","P95FirstChunk (s)"],
    "TotalResponse (s)": ["TotalResponse (s)","Total Response (s)"],
    "ReasoningTime (s)": ["ReasoningTime (s)","Reasoning Time (s)"],

    "MMLU-Pro (Reasoning & Knowledge)": [
        "MMLU-Pro(Reasoning &Knowledge)","MMLU-Pro(Reasoning & Knowledge)","MMLU Pro (Reasoning & Knowledge)","MMLU-Pro"
    ],
    "GPQA Diamond (Scientific Reasoning)": [
        "GPQA Diamond(ScientificReasoning)","GPQA Diamond(Scientific Reasoning)","GPQA Diamond","GPQA-Diamond"
    ],
    "Humanity's Last Exam (Reasoning & Knowledge)": [
        "Humanity's LastExam(Reasoning & Knowledge)","Humanity's Last Exam(Reasoning & Knowledge)",
        "Humanitys Last Exam (Reasoning & Knowledge)"
    ],

    "LiveCodeBench (Coding)": ["LiveCodeBench(Coding)","LiveCodeBench"],
    "SciCode (Coding)": ["SciCode(Coding)","SciCode"],
    "HumanEval (Coding)": ["HumanEval(Coding)","HumanEval"],

    "AIME 2025 (Competition Math)": ["AIME 2025(CompetitionMath)","AIME 2025"],
    "AIME 2024 (Competition Math)": ["AIME 2024(CompetitionMath)","AIME 2024"],
    "Math 500 (Competition Math)": ["Math 500(CompetitionMath)","Math 500"],

    "IFBench (Instruction Following)": ["IFBench(InstructionFollowing)","IFBench"],

    "Terminal-Bench Hard (Agentic Coding & Terminal Use)": [
        "Terminal-BenchHard (AgenticCoding & Terminal Use)","Terminal-Bench Hard (Agentic Coding & Terminal Use)","Terminal Bench Hard"
    ],
    "Tau2-Bench Telecom (Agentic Tool Use)": [
        "𝜏²-BenchTelecom(Agentic Tool Use)","Tau^2-BenchTelecom(Agentic Tool Use)",
        "Tau2-Bench Telecom (Agentic Tool Use)","Tau2-BenchTelecom"
    ],
    "AA-LCR (Long Context Reasoning)": [
        "AA-LCR(LongContext Reasoning)","AA-LCR (LongContext Reasoning)","AA-LCR","AA-LCR (Long Context Reasoning)"
    ],

    "Chatbot Arena": ["ChatbotArena","Chatbot Arena"],

    "length": ["length"],
    "Query Length": ["Query Length","QueryLength"],
}

def _norm(s: str) -> str:
    s = s.lower().replace('\xa0', ' ').replace('\u200b', '')
    s = re.sub(r'\s+', ' ', s).strip()
    return re.sub(r'[^a-z0-9]', '', s)

CANON_LOOKUP = {}
for canon, variants in CANON.items():
    for v in variants:
        CANON_LOOKUP[_norm(v)] = canon

def canonicalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    fixed = []
    for c in df.columns:
        key = _norm(c)
        if key in CANON_LOOKUP:
            fixed.append(CANON_LOOKUP[key])
        else:
            c2 = (c.replace('\xa0',' ').replace('\u200b','').replace('–','-'))
            c2 = re.sub(r'\s+', ' ', c2).strip()
            fixed.append(c2)
    df.columns = fixed

    if "API ID" not in df.columns and "API" in df.columns and "ID" in df.columns:
        df["API ID"] = df["API"].astype(str) + " " + df["ID"].astype(str)
        df.drop(columns=[c for c in ["API","ID"] if c in df.columns], inplace=True)

    return df

def resolve_columns(df: pd.DataFrame, targets: list[str]) -> list[str]:
    """Return actual present names corresponding to targets, using canonical keys."""
    table = {_norm(c): c for c in df.columns}
    resolved = []
    for t in targets:
        k = _norm(t)
        if k in table:
            resolved.append(table[k])
        else:
            cands = [c for kk,c in table.items() if k in kk or kk in k]
            if len(cands) == 1:
                resolved.append(cands[0])
    return resolved

def drop_targets(df: pd.DataFrame, targets: list[str]) -> None:
    cols = resolve_columns(df, targets)
    if cols:
        df.drop(columns=cols, inplace=True, errors="ignore")

def resolve_columns(df: pd.DataFrame, targets: list[str]) -> list[str]:
    """Return actual present names corresponding to targets, using canonical keys."""
    table = {_norm(c): c for c in df.columns}
    resolved = []
    for t in targets:
        k = _norm(t)
        if k in table:
            resolved.append(table[k])
        else:
            cands = [c for kk,c in table.items() if k in kk or kk in k]
            if len(cands) == 1:
                resolved.append(cands[0])
    return resolved

def drop_targets(df: pd.DataFrame, targets: list[str]) -> None:
    cols = resolve_columns(df, targets)
    if cols:
        df.drop(columns=cols, inplace=True, errors="ignore")



df_short= pd.read_csv('./data/artificialanalysis_cleanshort.csv')
df_medium= pd.read_csv('./data/artificialanalysis_cleanmedium.csv')
df_long= pd.read_csv('./data/artificialanalysis_cleanlong.csv')

# In[3]:


df_short['length'] = 'Short'
df_medium['length'] = 'Medium'  
df_long['length'] = 'Long'

df_short['Query Length'] = '300'
df_medium['Query Length'] = '1000'  
df_long['Query Length'] = '1500'

# In[4]:

df_short  = canonicalize_headers(df_short)
df_medium = canonicalize_headers(df_medium)
df_long   = canonicalize_headers(df_long)

for name, d in [("short", df_short), ("medium", df_medium), ("long", df_long)]:
    if {"API ID","Model"}.issubset(d.columns):
        d.drop_duplicates(subset=["API ID","Model"], inplace=True)
    else:
        print(f"[{name}] missing API ID or Model. Columns: {list(d.columns)}")







# In[5]:


df_merged = pd.concat([df_short, df_medium, df_long], ignore_index=True)

# In[6]:


df_merged

# In[7]:


df=df_merged.copy()

# In[8]:


df.columns

# In[9]:


api_id= [
  "mistral.mistral-large-2407-v1:0",
      "gemini-2.5-pro",
      "google/gemini-2.5-flash",
      "mistral-medium-2505",
  "Mistral-Large-2411",
  "Mistral-small",
  "Mistral-Nemo",
    "DeepSeek-V3-0324",
    "gpt-5-mini-2025-08-07",
    "gpt-5-nano-2025-08-07",
    "gpt-5-2025-08-07",
    "DeepSeek-R1-0528",
    "o3-2025-04-16",
    "o4-mini-2025-04-16",
    "o3-mini",
    "o1-2024-12-17",
    "o1-mini-2024-09-12",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-2025-04-14",
    "chatgpt-4o-latest",
    "gpt-4o-2024-11-20",
    "gpt-4.1-nano-2025-04-14",
    "gpt-4o",
    "gpt-4o-mini-2024-07-18",
    "o3-pro-2025-06-10",
    "o1-preview-2024-09-12",
    "gpt-4o-2024-08-06",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "gpt-4",  
    "claude-opus-4-20250514",
    "claude-3-7-sonnet-20250219",
    "claude-sonnet-4-20250514",
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
    "claude-3-5-haiku-20241022",
     "claude-3-haiku-20240307",
    "us.meta.llama4-maverick-17b-instruct-v1:0",
    "us.meta.llama4-scout-17b-instruct-v1:0",
    "us.meta.llama3-3-70b-instruct-v1:0",
    "us.meta.llama3-2-90b-instruct-v1:0",
    "meta.llama3-70b-instruct-v1:0",
    "meta.llama3-1-8b-instruct-v1:0",
    "meta.llama3-8b-instruct-v1:0",
    "us.meta.llama3-2-3b-instruct-v1:0",
    "us.meta.llama3-2-1b-instruct-v1:0",
    "us.meta.llama3-2-11b-instruct-v1:0",
    "us.meta.llama3-1-405b-instruct-v1:0",
    "us.meta.llama3-1-70b-instruct-v1:0",
    "meta.llama3-1-405b-instruct-v1:0",
    "meta.llama3-1-70b-instruct-v1:0",
    "deepseek-reasoner",
    "deepseek-chat",
    "grok-4-0709",
    "grok-3-mini-beta",
    "grok-3-mini-fast-beta",
    "grok-3-beta",
    "grok-3-fast-beta",
    "grok-3-mini-beta",
    "grok-3-mini-fast-beta",
        "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-3-7-sonnet-20250219",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
    "claude-3-5-haiku-20241022",
     "claude-3-haiku-20240307"
]


# In[10]:

MISTRAL_API_ID_AZURE=[      "mistral-medium-2505",
  "Mistral-Large-2411",
  "Mistral-small",
  "Mistral-Nemo",
            ]

MISTRAL_API_ID_AWS=[     
 "mistral.mistral-large-2407-v1:0"
            ]

    
GROK_API_ID=[  "grok-4-0709",
    "grok-3-mini-beta",
    "grok-3-mini-fast-beta",
    "grok-3-beta",
    "grok-3-fast-beta",
    "grok-3-mini-beta",
    "grok-3-mini-fast-beta"]

# In[11]:


OpenAI_API_ID_NEW = [
    "gpt-5-mini-2025-08-07",
    "gpt-5-nano-2025-08-07",
    "gpt-5-2025-08-07",
    "o3-2025-04-16",
    "o4-mini-2025-04-16",
    "o3-mini",
    "o1-2024-12-17",
    "o1-mini-2024-09-12",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-2025-04-14",
    "chatgpt-4o-latest",
    "gpt-4o-2024-11-20",
    "gpt-4.1-nano-2025-04-14",
    "gpt-4o",
    "o3-pro-2025-06-10",
    "o1-preview-2024-09-12",
    "gpt-4o-2024-08-06",
    ]
OpenAI_API_ID_OLD = [    
    "gpt-4o-mini-2024-07-18",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "gpt-4"]

# In[12]:


CLAUDE_API_ID = ["claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",

    "claude-3-opus-20240229",
    "claude-3-5-haiku-20241022",
    "claude-3-haiku-20240307"
]

GOOGLE_API_ID= [      "gemini-2.5-pro",
      "google/gemini-2.5-flash"]

# In[13]:


LLama_API_ID = [

      "us.meta.llama4-maverick-17b-instruct-v1:0",
    "us.meta.llama4-scout-17b-instruct-v1:0",
    "us.meta.llama3-3-70b-instruct-v1:0",
    "us.meta.llama3-2-90b-instruct-v1:0",
    "meta.llama3-70b-instruct-v1:0",
    "meta.llama3-1-8b-instruct-v1:0",
    "meta.llama3-8b-instruct-v1:0",
    "us.meta.llama3-2-3b-instruct-v1:0",
    "us.meta.llama3-2-1b-instruct-v1:0",
    "us.meta.llama3-2-11b-instruct-v1:0",
    "us.meta.llama3-1-405b-instruct-v1:0",
    "us.meta.llama3-1-70b-instruct-v1:0",
    "meta.llama3-1-405b-instruct-v1:0",
    "meta.llama3-1-70b-instruct-v1:0",
    ]

# In[14]:


DEEPSEEK_API_ID = ["deepseek-reasoner",
    "deepseek-chat"]

DEEPSEEK_API_Microsoft_Azure = ["DeepSeek-V3-0324","DeepSeek-R1-0528"]

# In[15]:


LARGE_API_ID= [
  "mistral.mistral-large-2407-v1:0",
    "gemini-2.5-pro",
    "google/gemini-2.5-flash",
      "mistral-medium-2505",
  "Mistral-Large-2411",
    "DeepSeek-V3-0324",
    "gpt-5-2025-08-07",
    "DeepSeek-R1-0528",
    "deepseek-reasoner",
    "deepseek-chat",
    "us.meta.llama3-2-90b-instruct-v1:0",
    "us.meta.llama3-1-405b-instruct-v1:0",
    "meta.llama3-1-405b-instruct-v1:0",

    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
    "claude-3-5-haiku-20241022",
     "claude-3-haiku-20240307",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "gpt-4",
    "o3-2025-04-16",
    "o1-2024-12-17",
    "gpt-4.1-2025-04-14",
    "chatgpt-4o-latest",
    "gpt-4o-2024-11-20",
    "gpt-4o",
    "o3-pro-2025-06-10",
    "o1-preview-2024-09-12",
    "gpt-4o-2024-08-06",
        "grok-4-0709",
   
    "grok-3-beta",
    "grok-3-fast-beta",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-3-7-sonnet-20250219",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
    "claude-3-5-haiku-20241022",
     "claude-3-haiku-20240307"

 ]



# In[16]:


MEDIUM_API_ID = [
    "gpt-5-mini-2025-08-07",
    "o4-mini-2025-04-16",
    "o3-mini",
    "o1-mini-2024-09-12",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4o-mini-2024-07-18",
    "us.meta.llama3-3-70b-instruct-v1:0",
    "meta.llama3-70b-instruct-v1:0",
    "us.meta.llama3-1-70b-instruct-v1:0",
    "meta.llama3-1-70b-instruct-v1:0",
    "grok-3-mini-beta",
    "grok-3-mini-fast-beta",
    "grok-3-mini-beta",
    "grok-3-mini-fast-beta"]

    

# In[17]:


SMALL_API_ID = [
      "Mistral-small",
  "Mistral-Nemo",
    "gpt-5-nano-2025-08-07",
    "gpt-4.1-nano-2025-04-14",
    
]

# In[18]:


MICRO_API_ID = [
    "us.meta.llama4-maverick-17b-instruct-v1:0",
    "us.meta.llama4-scout-17b-instruct-v1:0",
    "meta.llama3-1-8b-instruct-v1:0",
    "meta.llama3-8b-instruct-v1:0",
    "us.meta.llama3-2-11b-instruct-v1:0"]


# In[19]:


NANO_API_ID = [
    "us.meta.llama3-2-3b-instruct-v1:0",
    "us.meta.llama3-2-1b-instruct-v1:0",
    ]

# In[20]:


df.columns

# In[21]:



# In[22]:




# In[23]:


# In[24]:



# In[25]:


df_selected = df[df['API ID'].isin(api_id)]

# In[26]:



# In[52]:




# In[28]:




# In[29]:


df_selected.columns


mask = df_selected['API ID'] == "deepseek-reasoner"
reasoning_time= df_selected.loc[mask, 'First AnswerToken (s)']-df_selected.loc[mask, 'MedianFirst Chunk (s)']

df_selected.loc[mask, 'MedianFirst Chunk (s)'] +=reasoning_time


df_selected.loc[mask, 'P5First Chunk (s)'] +=reasoning_time*0.1
df_selected.loc[mask, 'P25First Chunk (s)'] +=reasoning_time*0.5
df_selected.loc[mask, 'P75First Chunk (s)'] +=reasoning_time*1.5
df_selected.loc[mask, 'P95First Chunk (s)']= df_selected.loc[mask, 'P95First Chunk (s)'].astype(float)
df_selected.loc[mask, 'P95First Chunk (s)'] +=reasoning_time*1.9
df_selected.loc[mask, 'Model'] = df_selected.loc[mask].apply(
    lambda row: f"DeepSeek R1 (DeepSeek) [{row['ContextWindow']}]"
    if pd.notna(row['ContextWindow'])
    else "DeepSeek R1 (DeepSeek)",
    axis=1
)



mask = df_selected['API ID'] == "DeepSeek-R1-0528"
reasoning_time= df_selected.loc[mask, 'First AnswerToken (s)']-df_selected.loc[mask, 'MedianFirst Chunk (s)']

df_selected.loc[mask, 'MedianFirst Chunk (s)'] +=reasoning_time


df_selected.loc[mask, 'P5First Chunk (s)'] +=reasoning_time*0.1
df_selected.loc[mask, 'P25First Chunk (s)'] +=reasoning_time*0.5
df_selected.loc[mask, 'P75First Chunk (s)'] +=reasoning_time*1.5
df_selected.loc[mask, 'P95First Chunk (s)']= df_selected.loc[mask, 'P95First Chunk (s)'].astype(float)
df_selected.loc[mask, 'P95First Chunk (s)'] +=reasoning_time*1.9
df_selected.loc[mask, 'Model'] = "DeepSeek R1 (Azure)"


mask = df_selected['API ID'] == "deepseek-chat"
df_selected.loc[mask, 'Model'] = "DeepSeek V3 (DeepSeek)"

mask = df_selected['API ID'] == "DeepSeek-V3-0324"
df_selected.loc[mask, 'Model'] = "DeepSeek V3 (Azure)"


mask = df_selected['API ID'] == "mistral.mistral-large-2407-v1:0"
df_selected.loc[mask, 'Model'] = "Mistral Large 2 (AWS)"

mask = df_selected['API ID'] == "Mistral-Large-2411"
df_selected.loc[mask, 'Model'] = "Mistral Large 2 (Azure)"
# In[35]:


df_selected.drop(columns=['API Provider',"Function Calling",'JSON Mode', 'License', 'OpenAI Compatible',"Footnotes",
       'MMLU-Pro (Reasoning & Knowledge)',
       'GPQA Diamond (Scientific Reasoning)',
       "Humanity's Last Exam (Reasoning & Knowledge)",
       'LiveCodeBench (Coding)', 'SciCode (Coding)',
       'IFBench (Instruction Following)', 'AIME 2025 (Competition Math)',
        'BlendedUSD/1M Tokens', 'Input PriceUSD/1M Tokens',
       'Output PriceUSD/1M Tokens','TotalResponse (s)',
       'ReasoningTime (s)', 'FurtherAnalysis'], inplace=True)

# In[36]:


df_selected.reset_index(drop=True, inplace=True)

# In[37]:


df_selected.shape

# In[38]:


def get_hardware_host(api):
    if api in OpenAI_API_ID_NEW:
        return "DGX H200/H100", "Azure"
    elif api in OpenAI_API_ID_OLD:
        return "DGX A100", "Azure"
    elif api in DEEPSEEK_API_ID:
        return "DGX H800", "DeepSeek"
    elif api in DEEPSEEK_API_Microsoft_Azure:
        return "DGX H200/H100", "Azure"
    elif api in CLAUDE_API_ID:
        return "DGX H200/H100", "Anthropic"
    elif api in LLama_API_ID:
        return "DGX H200/H100", "AWS"
    elif api in GROK_API_ID:
        return "DGX H200/H100", "xAI"
    elif api in MISTRAL_API_ID_AZURE:
        return "DGX H200/H100", "Azure"
    elif api in MISTRAL_API_ID_AWS:
        return "DGX H200/H100", "AWS"
    elif api in GOOGLE_API_ID:
          return "TPU V6e", "Google"
    else:
        return None, None  

df_selected[['Hardware', 'Host']] = df_selected['API ID'].apply(
    lambda x: pd.Series(get_hardware_host(x))
)


# In[39]:


def get_power_consumption(hardware):
    if hardware in ["DGX H200/H100", "DGX H800"]:
        return 5.6, 4.6
    elif hardware in ["DGX A100"]:
        return 3.2, 3.3
    elif hardware in ["TPU V6e"]:
          return 1.2, 0.96
    else:
        return None, None 

df_selected[['GPUs Power Draw', 'Non-GPUs Power Draw']] = df_selected['Hardware'].apply(
    lambda x: pd.Series(get_power_consumption(x))
)


# In[40]:



def determine_utilization(row):
    if row['API ID'] in LARGE_API_ID and row['Hardware'] in ["DGX H200/H100", "DGX H800"]:
        return pd.Series([0.055, 0.075, 0.0625])
    if row['API ID'] in LARGE_API_ID and row['Hardware'] in ["TPU V6e"]:
        return pd.Series([0.1, 0.1125, 0.1])
    elif row['API ID'] in MEDIUM_API_ID and row['Hardware'] in ["DGX H200/H100", "DGX H800"]:
        return pd.Series([0.03, 0.045, 0.03125])
    elif row['API ID'] in SMALL_API_ID and row['Hardware'] in ["DGX H200/H100", "DGX H800"]:
        return pd.Series([0.017, 0.025, 0.016])
    elif row['API ID'] in MICRO_API_ID and row['Hardware'] in ["DGX H200/H100", "DGX H800"]:
        return pd.Series([0.0075, 0.0125, 0.0087])
    elif row['API ID'] in NANO_API_ID and row['Hardware'] in ["DGX H200/H100", "DGX H800"]:
        return pd.Series([0.0055, 0.01, 0.0087])
    elif row['API ID'] in LARGE_API_ID and row['Hardware'] == "DGX A100":
        return pd.Series([0.125, 0.15, 0.0625])
    elif row['API ID'] in MEDIUM_API_ID and row['Hardware'] == "DGX A100":
        return pd.Series([0.0625, 0.07, 0.0312])
    else:
        return pd.Series([None, None, None])

df_selected[["Min GPU Power Utilization", "Max GPU Power Utilization", "Non-GPU Power Utilization"]] = df_selected.apply(determine_utilization, axis=1)


# In[41]:


def get_environmental_multipliers(api):
    if api in OpenAI_API_ID_NEW:
        return 1.12,0.3,3.142,0.3528
    if api in MISTRAL_API_ID_AZURE:
        return 1.12,0.3,3.142,0.3528
    if api in MISTRAL_API_ID_AWS:
        return 1.14,0.18,3.142,0.385
    if api in GOOGLE_API_ID:
          return 1.09, 0.3, 1.1, 0.231
    elif api in OpenAI_API_ID_OLD:
        return 1.12,0.3,3.142,0.3528
    elif api in DEEPSEEK_API_ID:
        return 1.27,1.2,6.016,0.6
    elif api in DEEPSEEK_API_Microsoft_Azure:
        return 1.12,0.3,3.142,0.3528
    elif api in CLAUDE_API_ID:
        return 1.14,0.18,3.142,0.385
    elif api in LLama_API_ID:
        return 1.14,0.18,3.142,0.385
    elif api in GROK_API_ID:
        return 1.5, 0.36,3.142,0.385
    else:
        return None, None  

df_selected[['PUE', 'WUE (Site)', "WUE (Source)", 'CIF']] = df_selected['API ID'].apply(
    lambda x: pd.Series(get_environmental_multipliers(x))
)


# In[42]:


def get_company(api):
    if api in OpenAI_API_ID_NEW:
        return "OpenAI"
    elif api in OpenAI_API_ID_OLD:
        return "OpenAI"
    elif api in DEEPSEEK_API_ID:
        return "DeepSeek"
    elif api in DEEPSEEK_API_Microsoft_Azure:
        return "DeepSeek"
    elif api in CLAUDE_API_ID:
        return "Anthropic"
    elif api in LLama_API_ID:
        return "Meta"
    elif api in GROK_API_ID:
        return "xAI"
    elif api in MISTRAL_API_ID_AWS:
        return "Mistral AI"
    elif api in MISTRAL_API_ID_AZURE:
        return "Mistral AI"
    elif api in GOOGLE_API_ID:
          return "Google"
    else:
        return None  
        
df_selected["Company"] = df_selected['API ID'].apply(
    lambda x: pd.Series(get_company(x))
)


# In[43]:


def get_size(api):
    if api in LARGE_API_ID:
        return "Large"
    elif api in MEDIUM_API_ID:
        return "Medium"
    elif api in SMALL_API_ID:
        return "Small"
    elif api in MICRO_API_ID:
        return "Micro"
    elif api in NANO_API_ID:
        return "Nano"
    else:
        return None, None  
        
df_selected["Size"] = df_selected['API ID'].apply(
    lambda x: pd.Series(get_size(x))
)


# In[44]:


df_selected.columns

# In[45]:






def compute_environmental_metrics(row):
    try:
        output_tokens = float(row["Query Length"])
    except (KeyError, TypeError, ValueError):
        return pd.Series([None] * 30)

    latency_cols = ['P5First Chunk (s)', 'P25First Chunk (s)', 'MedianFirst Chunk (s)',
                    'P75First Chunk (s)', 'P95First Chunk (s)']
    tps_cols = ['P5Tokens/s', 'P25Tokens/s', 'MedianTokens/s', 'P75Tokens/s', 'P95Tokens/s']

    energy_max_vals, energy_min_vals = [], []
    carbon_max_vals, carbon_min_vals = [], []

    water_site_max_vals, water_site_min_vals = [], []       
    water_source_max_vals, water_source_min_vals = [], []   
    water_comb_max_vals, water_comb_min_vals = [], []       

    for latency_col in latency_cols:
        for tps_col in tps_cols:
            try:
                latency = float(row[latency_col])
                tps = float(row[tps_col])
                if tps <= 0:
                    continue
            except (ValueError, ZeroDivisionError, TypeError):
                continue

            base_time = (latency + (output_tokens / tps)) / 3600.0

            gpu_power = float(row['GPUs Power Draw'])
            non_gpu_power = float(row['Non-GPUs Power Draw'])
            max_gpu_util = float(row['Max GPU Power Utilization'])
            min_gpu_util = float(row['Min GPU Power Utilization'])
            non_gpu_util = float(row['Non-GPU Power Utilization'])
            pue = float(row['PUE'])
            cif = float(row['CIF'])
            wue_site = float(row['WUE (Site)'])       
            wue_source = float(row['WUE (Source)'])  

            power_draw_max = (gpu_power * max_gpu_util) + (non_gpu_power * non_gpu_util)  
            energy_max = base_time * power_draw_max * pue   
            carbon_max = energy_max * cif                   

            water_source_max = energy_max * wue_source      
            water_site_max = (energy_max / pue) * wue_site  
            water_combined_max = water_source_max + water_site_max

            energy_max_vals.append(energy_max)
            carbon_max_vals.append(carbon_max)
            water_source_max_vals.append(water_source_max)
            water_site_max_vals.append(water_site_max)
            water_comb_max_vals.append(water_combined_max)

            power_draw_min = (gpu_power * min_gpu_util) + (non_gpu_power * non_gpu_util)  
            energy_min = base_time * power_draw_min * pue   
            carbon_min = energy_min * cif                   

            water_source_min = energy_min * wue_source      
            water_site_min = (energy_min / pue) * wue_site  
            water_combined_min = water_source_min + water_site_min

            energy_min_vals.append(energy_min)
            carbon_min_vals.append(carbon_min)
            water_source_min_vals.append(water_source_min)
            water_site_min_vals.append(water_site_min)
            water_comb_min_vals.append(water_combined_min)

    if not energy_max_vals and not energy_min_vals:
        return pd.Series([None] * 30)

    energy_comb = energy_max_vals + energy_min_vals
    carbon_comb = carbon_max_vals + carbon_min_vals

    water_site_comb = water_site_max_vals + water_site_min_vals
    water_source_comb = water_source_max_vals + water_source_min_vals
    water_comb = water_comb_max_vals + water_comb_min_vals

    m = lambda x: np.mean(x) if len(x) else None
    s = lambda x: np.std(x) if len(x) else None

    return pd.Series([
        m(energy_max_vals), s(energy_max_vals),
        m(energy_min_vals), s(energy_min_vals),
        m(energy_comb),     s(energy_comb),

        m(carbon_max_vals), s(carbon_max_vals),
        m(carbon_min_vals), s(carbon_min_vals),
        m(carbon_comb),     s(carbon_comb),

        m(water_site_max_vals), s(water_site_max_vals),
        m(water_site_min_vals), s(water_site_min_vals),
        m(water_site_comb),     s(water_site_comb),

        m(water_source_max_vals), s(water_source_max_vals),
        m(water_source_min_vals), s(water_source_min_vals),
        m(water_source_comb),     s(water_source_comb),

        m(water_comb_max_vals), s(water_comb_max_vals),
        m(water_comb_min_vals), s(water_comb_min_vals),
        m(water_comb),          s(water_comb),
    ])

df_environmental = df_selected.copy()

df_environmental[[
    'Mean Max Energy (Wh)', 'Std Max Energy (Wh)',
    'Mean Min Energy (Wh)', 'Std Min Energy (Wh)',
    'Mean Combined Energy (Wh)', 'Std Combined Energy (Wh)',

    'Mean Max Carbon (gCO2e)', 'Std Max Carbon (gCO2e)',
    'Mean Min Carbon (gCO2e)', 'Std Min Carbon (gCO2e)',
    'Mean Combined Carbon (gCO2e)', 'Std Combined Carbon (gCO2e)',

    'Mean Max Water (Site, mL)', 'Std Max Water (Site, mL)',
    'Mean Min Water (Site, mL)', 'Std Min Water (Site, mL)',
    'Mean Combined Water (Site, mL)', 'Std Combined Water (Site, mL)',

    'Mean Max Water (Source, mL)', 'Std Max Water (Source, mL)',
    'Mean Min Water (Source, mL)', 'Std Min Water (Source, mL)',
    'Mean Combined Water (Source, mL)', 'Std Combined Water (Source, mL)',

    'Mean Max Water (Site & Source, mL)', 'Std Max Water (Site & Source, mL)',
    'Mean Min Water (Site & Source, mL)', 'Std Min Water (Site & Source, mL)',
    'Mean Combined Water (Site & Source, mL)', 'Std Combined Water (Site & Source, mL)',
]] = df_environmental.apply(compute_environmental_metrics, axis=1)

df_environmental[[
    'Mean Max Energy (Wh)', 'Std Max Energy (Wh)',
    'Mean Min Energy (Wh)', 'Std Min Energy (Wh)',
    'Mean Combined Energy (Wh)', 'Std Combined Energy (Wh)',

    # Carbon
    'Mean Max Carbon (gCO2e)', 'Std Max Carbon (gCO2e)',
    'Mean Min Carbon (gCO2e)', 'Std Min Carbon (gCO2e)',
    'Mean Combined Carbon (gCO2e)', 'Std Combined Carbon (gCO2e)',

    # Water - Scope 1 (Site)
    'Mean Max Water (Site, mL)', 'Std Max Water (Site, mL)',
    'Mean Min Water (Site, mL)', 'Std Min Water (Site, mL)',
    'Mean Combined Water (Site, mL)', 'Std Combined Water (Site, mL)',

    # Water - Scope 2 (Source)
    'Mean Max Water (Source, mL)', 'Std Max Water (Source, mL)',
    'Mean Min Water (Source, mL)', 'Std Min Water (Source, mL)',
    'Mean Combined Water (Source, mL)', 'Std Combined Water (Source, mL)',

    # Water - Site & Source (combined)
    'Mean Max Water (Site & Source, mL)', 'Std Max Water (Site & Source, mL)',
    'Mean Min Water (Site & Source, mL)', 'Std Min Water (Site & Source, mL)',
    'Mean Combined Water (Site & Source, mL)', 'Std Combined Water (Site & Source, mL)',
]]=df_environmental[[
    # Energy
    'Mean Max Energy (Wh)', 'Std Max Energy (Wh)',
    'Mean Min Energy (Wh)', 'Std Min Energy (Wh)',
    'Mean Combined Energy (Wh)', 'Std Combined Energy (Wh)',

    # Carbon
    'Mean Max Carbon (gCO2e)', 'Std Max Carbon (gCO2e)',
    'Mean Min Carbon (gCO2e)', 'Std Min Carbon (gCO2e)',
    'Mean Combined Carbon (gCO2e)', 'Std Combined Carbon (gCO2e)',

    # Water - Scope 1 (Site)
    'Mean Max Water (Site, mL)', 'Std Max Water (Site, mL)',
    'Mean Min Water (Site, mL)', 'Std Min Water (Site, mL)',
    'Mean Combined Water (Site, mL)', 'Std Combined Water (Site, mL)',

    # Water - Scope 2 (Source)
    'Mean Max Water (Source, mL)', 'Std Max Water (Source, mL)',
    'Mean Min Water (Source, mL)', 'Std Min Water (Source, mL)',
    'Mean Combined Water (Source, mL)', 'Std Combined Water (Source, mL)',

    # Water - Site & Source (combined)
    'Mean Max Water (Site & Source, mL)', 'Std Max Water (Site & Source, mL)',
    'Mean Min Water (Site & Source, mL)', 'Std Min Water (Site & Source, mL)',
    'Mean Combined Water (Site & Source, mL)', 'Std Combined Water (Site & Source, mL)',
]]*1000


# In[47]:


df_selected.loc[df_selected['Model']=="DeepSeek R1 0528 (May '25)", 'P5First Chunk (s)']

# In[48]:


df_environmental.loc[df_environmental['Model']=="DeepSeek R1 0528 (May '25)", 'Model'] = 'DeepSeek R1 (May 2025)'
df_environmental.loc[df_environmental['Model']=="DeepSeek V3 0324 (Mar '25)", 'Model'] = "DeepSeek V3 (Mar '25)"

df_environmental["Energy Consumption of 1 Billion Prompts (MWh)"] = df_environmental['Mean Combined Energy (Wh)']*1000
df_environmental["Carbon Emissions of 1 Billion Prompts (TonsCO2e)"] = df_environmental['Mean Combined Carbon (gCO2e)']*1000
df_environmental["Water Consumption of 1 Billion Prompts (Kiloliter)"] = df_environmental['Mean Combined Water (Site & Source, mL)']*1000

df_environmental["Energy Consumption of 50 Billion Prompts (MWh)"] = df_environmental['Mean Combined Energy (Wh)']*50*1000
df_environmental["Carbon Emissions of 50 Billion Prompts (TonsCO2e)"] = df_environmental['Mean Combined Carbon (gCO2e)']*50*1000
df_environmental["Water Consumption of 50 Billion Prompts (Kiloliter)"] = df_environmental['Mean Combined Water (Site & Source, mL)']*50*1000

df_environmental["Energy Consumption of 100 Billion Prompts (MWh)"] = df_environmental['Mean Combined Energy (Wh)']*100*100
df_environmental["Carbon Emissions of 100 Billion Prompts (TonsCO2e)"] = df_environmental['Mean Combined Carbon (gCO2e)']*100*1000
df_environmental["Water Consumption of 100 Billion Prompts (Kiloliter)"] = df_environmental['Mean Combined Water (Site & Source, mL)']*100*1000

df_environmental['Household Energy Equiv. – 1B Prompts (MWh)'] = df_environmental["Energy Consumption of 1 Billion Prompts (MWh)"]/1.0950
df_environmental["University Energy Equiv. – 1B Prompts (MWh)"] = df_environmental["Energy Consumption of 1 Billion Prompts (MWh)"]/1202
df_environmental['Household Energy Equiv. – 50B Prompts (MWh)'] = df_environmental["Energy Consumption of 50 Billion Prompts (MWh)"]/1.0950
df_environmental['Household Energy Equiv. – 100B Prompts (MWh)'] = df_environmental["Energy Consumption of 100 Billion Prompts (MWh)"]/1.0950
df_environmental["University Energy Equiv. – 50B Prompts (MWh)"] = df_environmental["Energy Consumption of 50 Billion Prompts (MWh)"]/1202
df_environmental["University Energy Equiv. – 100B Prompts (MWh)"] = df_environmental["Energy Consumption of 100 Billion Prompts (MWh)"]/1202


df_environmental['People Annual Drinking Water Equiv. – 1B Prompts (kL)'] = df_environmental["Water Consumption of 1 Billion Prompts (Kiloliter)"]/1.2
df_environmental['People Annual Drinking Water Equiv. – 50B Prompts (kL)'] = df_environmental["Water Consumption of 50 Billion Prompts (Kiloliter)"]/1.2
df_environmental['People Annual Drinking Water Equiv. – 100B Prompts (kL)'] = df_environmental["Water Consumption of 100 Billion Prompts (Kiloliter)"]/1.2
df_environmental["Olympic Swimming Pools Equiv. – 1B Prompts (kL)"] = df_environmental["Water Consumption of 1 Billion Prompts (Kiloliter)"]/2500
df_environmental["Olympic Swimming Pools Equiv. – 50B Prompts (kL)"] = df_environmental["Water Consumption of 50 Billion Prompts (Kiloliter)"]/2500
df_environmental["Olympic Swimming Pools Equiv. – 100B Prompts (kL)"] = df_environmental["Water Consumption of 100 Billion Prompts (Kiloliter)"]/2500

df_environmental["Gasoline Car Equiv. – 1B Prompts (TonsCO2e)"] = df_environmental["Carbon Emissions of 1 Billion Prompts (TonsCO2e)"]/4.6
df_environmental["Gasoline Car Equiv. – 50B Prompts (TonsCO2e)"] = df_environmental["Carbon Emissions of 50 Billion Prompts (TonsCO2e)"]/4.6
df_environmental["Gasoline Car Equiv. – 100B Prompts (TonsCO2e)"] = df_environmental["Carbon Emissions of 100 Billion Prompts (TonsCO2e)"]/4.6
df_environmental["Atlantic Flight Equiv. – 1B Prompts (TonsCO2e)"] = df_environmental["Carbon Emissions of 1 Billion Prompts (TonsCO2e)"]/60
df_environmental["Atlantic Flight Equiv. – 50B Prompts (TonsCO2e)"] = df_environmental["Carbon Emissions of 50 Billion Prompts (TonsCO2e)"]/60
df_environmental["Atlantic Flight Equiv. – 100B Prompts (TonsCO2e)"] = df_environmental["Carbon Emissions of 100 Billion Prompts (TonsCO2e)"]/60

# In[49]:


df_environmental.to_csv('artificialanalysis_environmental.csv', index=False)

snapshot_date = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
dated_fname = f"artificialanalysis_environmental_{snapshot_date}.csv"

df_snapshot = df_environmental.copy()
df_snapshot.insert(0, 'SnapshotDate', snapshot_date)
df_snapshot.to_csv(dated_fname, index=False)





# In[ ]:




































