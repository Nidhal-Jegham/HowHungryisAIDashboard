#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 

# In[2]:


df_short= pd.read_csv('./data/artificialanalysis_cleanshort.csv')
df_medium= pd.read_csv('./data/artificialanalysis_cleanmedium.csv')
df_long= pd.read_csv('./data/artificialanalysis_cleanlong.csv')

# In[3]:


df_short['length'] = 'Short'
df_medium['length'] = 'Medium'  
df_long['length'] = 'Long'

# In[4]:


df_short.drop_duplicates(subset= "API ID",inplace=True)
df_medium.drop_duplicates(subset= "API ID",inplace=True)
df_long.drop_duplicates(subset= "API ID",inplace=True)

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
    "DeepSeek-V3-0324",
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
    "claude-3-5-sonnet-20240620",
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
    "claude-3-5-sonnet-20240620",
    "claude-3-haiku-20240307"
]


# In[10]:


GROK_API_ID=[  "grok-4-0709",
    "grok-3-mini-beta",
    "grok-3-mini-fast-beta",
    "grok-3-beta",
    "grok-3-fast-beta",
    "grok-3-mini-beta",
    "grok-3-mini-fast-beta"]

# In[11]:


OpenAI_API_ID_NEW = ["o3-2025-04-16",
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

# In[13]:


LLama_API_ID = ["us.meta.llama4-maverick-17b-instruct-v1:0",
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

DEEPSEEK_API_Microsoft_Azure = ["DeepSeek-R1-0528","DeepSeek-V3-0324"]

# In[15]:


LARGE_API_ID= [
    "DeepSeek-V3-0324",
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
    "claude-3-5-sonnet-20240620",
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
    "claude-3-5-sonnet-20240620",
    "claude-3-haiku-20240307"

 ]



# In[16]:


MEDIUM_API_ID = [
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
df_selected.loc[mask, 'Model'] = "DeepSeek R1 (DeepSeek)"

mask = df_selected['API ID'] == "DeepSeek-R1-0528"
reasoning_time= df_selected.loc[mask, 'First AnswerToken (s)']-df_selected.loc[mask, 'MedianFirst Chunk (s)']

df_selected.loc[mask, 'MedianFirst Chunk (s)'] +=reasoning_time


df_selected.loc[mask, 'P5First Chunk (s)'] +=reasoning_time*0.1
df_selected.loc[mask, 'P25First Chunk (s)'] +=reasoning_time*0.5
df_selected.loc[mask, 'P75First Chunk (s)'] +=reasoning_time*1.5
df_selected.loc[mask, 'P95First Chunk (s)']= df_selected.loc[mask, 'P95First Chunk (s)'].astype(float)
df_selected.loc[mask, 'P95First Chunk (s)'] +=reasoning_time*1.9
df_selected.loc[mask, 'Model'] = "DeepSeek R1 (Microsoft Azure)"


# In[35]:


df_selected.drop(columns=['API Provider',"Function Calling",'JSON Mode', 'License', 'OpenAI Compatible',"Footnotes",
       'MMLU-Pro (Reasoning & Knowledge)',
       'GPQA Diamond (Scientific Reasoning)',
       "Humanity's Last Exam (Reasoning & Knowledge)",
       'LiveCodeBench (Coding)', 'SciCode (Coding)',
       'IFBench (Instruction Following)', 'AIME 2025 (Competition Math)',
       'Chatbot Arena', 'BlendedUSD/1M Tokens', 'Input PriceUSD/1M Tokens',
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
        return "DGX H800", "Deepseek"
    elif api in DEEPSEEK_API_Microsoft_Azure:
        return "DGX H200/H100", "Azure"
    elif api in CLAUDE_API_ID:
        return "DGX H200/H100", "Anthropic"
    elif api in LLama_API_ID:
        return "DGX H200/H100", "AWS"
    elif api in GROK_API_ID:
        return "DGX H200/H100", "xAI"
    else:
        return None, None  # default fallback

# Apply to each row
df_selected[['Hardware', 'Host']] = df_selected['API ID'].apply(
    lambda x: pd.Series(get_hardware_host(x))
)


# In[39]:


def get_power_consumption(hardware):
    if hardware in ["DGX H200/H100", "DGX H800"]:
        return 5.6, 4.6
    elif hardware in ["DGX A100"]:
        return 3.2, 3.3
    else:
        return None, None 

df_selected[['GPUs Power Draw', 'Non-GPUs Power Draw']] = df_selected['Hardware'].apply(
    lambda x: pd.Series(get_power_consumption(x))
)


# In[40]:


import pandas as pd

def determine_utilization(row):
    if row['API ID'] in LARGE_API_ID and row['Hardware'] in ["DGX H200/H100", "DGX H800"]:
        return pd.Series([0.055, 0.075, 0.0625])
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

# Apply row-wise and assign to columns
df_selected[["Min GPU Power Utilization", "Max GPU Power Utilization", "Non-GPU Power Utilization"]] = df_selected.apply(determine_utilization, axis=1)


# In[41]:


def get_environmental_multipliers(api):
    if api in OpenAI_API_ID_NEW:
        return 1.12,0.3,3.142,0.3528
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
        return None, None  # default fallback

# Apply to each row
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
    else:
        return None  # default fallback

# Apply to each row
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
        return None, None  # default fallback

# Apply to each row
df_selected["Size"] = df_selected['API ID'].apply(
    lambda x: pd.Series(get_size(x))
)


# In[44]:


df_selected.columns

# In[45]:




def compute_environmental_metrics(row):
    output_tokens = 300  
    latency_cols = ['P5First Chunk (s)', 'P25First Chunk (s)', 'MedianFirst Chunk (s)',
                    'P75First Chunk (s)', 'P95First Chunk (s)']
    tps_cols = ['P5Tokens/s', 'P25Tokens/s', 'MedianTokens/s', 'P75Tokens/s', 'P95Tokens/s']

    max_energy_values = []
    min_energy_values = []

    for latency_col in latency_cols:
        for tps_col in tps_cols:
            try:
                latency = float(row[latency_col])
                tps = float(row[tps_col])
                if tps == 0:
                    continue
            except (ValueError, ZeroDivisionError, TypeError):
                continue

            base_time = (latency + (output_tokens / tps)) / 3600

            gpu_power = row['GPUs Power Draw']
            non_gpu_power = row['Non-GPUs Power Draw']
            max_gpu_util = row['Max GPU Power Utilization']
            min_gpu_util = row['Min GPU Power Utilization']
            non_gpu_util = row['Non-GPU Power Utilization']
            pue = row['PUE']
            cif = row['CIF']
            wue_site = row['WUE (Site)']
            wue_source = row['WUE (Source)']

            # Max case
            power_draw_max = (gpu_power * max_gpu_util) + (non_gpu_power * non_gpu_util)
            energy_max = base_time * power_draw_max * pue
            carbon_max = energy_max * cif
            water_max = (energy_max * wue_source) + ((energy_max / pue) * wue_site)

            # Min case
            power_draw_min = (gpu_power * min_gpu_util) + (non_gpu_power * non_gpu_util)
            energy_min = base_time * power_draw_min * pue
            carbon_min = energy_min * cif
            water_min = (energy_min * wue_source) + ((energy_min / pue) * wue_site)

            max_energy_values.append((energy_max, carbon_max, water_max))
            min_energy_values.append((energy_min, carbon_min, water_min))

    if not max_energy_values or not min_energy_values:
        return pd.Series([None] * 18)

    energy_max_vals, carbon_max_vals, water_max_vals = zip(*max_energy_values)
    energy_min_vals, carbon_min_vals, water_min_vals = zip(*min_energy_values)

    combined_energy_vals = energy_max_vals + energy_min_vals
    combined_carbon_vals = carbon_max_vals + carbon_min_vals
    combined_water_vals = water_max_vals + water_min_vals

    return pd.Series([
        np.mean(energy_max_vals), np.std(energy_max_vals),
        np.mean(energy_min_vals), np.std(energy_min_vals),
        np.mean(combined_energy_vals), np.std(combined_energy_vals),
        np.mean(carbon_max_vals), np.std(carbon_max_vals),
        np.mean(carbon_min_vals), np.std(carbon_min_vals),
        np.mean(combined_carbon_vals), np.std(combined_carbon_vals),
        np.mean(water_max_vals), np.std(water_max_vals),
        np.mean(water_min_vals), np.std(water_min_vals),
        np.mean(combined_water_vals), np.std(combined_water_vals),

    ])

df_environmental = df_selected.copy()

df_environmental[[
    'Mean Max Energy (kWh)', 'Std Max Energy (kWh)',
    'Mean Min Energy (kWh)', 'Std Min Energy (kWh)',
    'Mean Combined Energy (kWh)', 'Std Combined Energy (kWh)',
    'Mean Max Carbon (kg/CO2)', 'Std Max Carbon (kg/CO2)',
    'Mean Min Carbon (kg/CO2)', 'Std Min Carbon (kg/CO2)',
    'Mean Combined Carbon (kg/CO2)', 'Std Combined Carbon (kg/CO2)',
    'Mean Max Water (L)', 'Std Max Water (L)',
    'Mean Min Water (L)', 'Std Min Water (L)',
    'Mean Combined Water (L)', 'Std Combined Water (L)',
]] = df_environmental.apply(compute_environmental_metrics, axis=1)


# In[46]:


df_environmental[[
    'Mean Max Energy (kWh)', 'Std Max Energy (kWh)',
    'Mean Min Energy (kWh)', 'Std Min Energy (kWh)',
    'Mean Combined Energy (kWh)', 'Std Combined Energy (kWh)',
    'Mean Max Carbon (kg/CO2)', 'Std Max Carbon (kg/CO2)',
    'Mean Min Carbon (kg/CO2)', 'Std Min Carbon (kg/CO2)',
    'Mean Combined Carbon (kg/CO2)', 'Std Combined Carbon (kg/CO2)',
    'Mean Max Water (L)', 'Std Max Water (L)',
    'Mean Min Water (L)', 'Std Min Water (L)',
    'Mean Combined Water (L)', 'Std Combined Water (L)',
]]= df_environmental[[
    'Mean Max Energy (kWh)', 'Std Max Energy (kWh)',
    'Mean Min Energy (kWh)', 'Std Min Energy (kWh)',
    'Mean Combined Energy (kWh)', 'Std Combined Energy (kWh)',
    'Mean Max Carbon (kg/CO2)', 'Std Max Carbon (kg/CO2)',
    'Mean Min Carbon (kg/CO2)', 'Std Min Carbon (kg/CO2)',
    'Mean Combined Carbon (kg/CO2)', 'Std Combined Carbon (kg/CO2)',
    'Mean Max Water (L)', 'Std Max Water (L)',
    'Mean Min Water (L)', 'Std Min Water (L)',
    'Mean Combined Water (L)', 'Std Combined Water (L)',
]]*1000  # Convert kg to gCO2e and kWh to Wh and L to mL

# In[47]:


df_selected.loc[df_selected['Model']=="DeepSeek R1 0528 (May '25)", 'P5First Chunk (s)']

# In[48]:


df_environmental.loc[df_environmental['Model']=="DeepSeek R1 0528 (May '25)", 'Model'] = 'DeepSeek R1 (May 2025)'
df_environmental.loc[df_environmental['Model']=="DeepSeek V3 0324 (Mar '25)", 'Model'] = "DeepSeek V3 (Mar '25)"

# In[49]:


df_environmental.to_csv('artificialanalysis_environmental.csv', index=False)

# In[50]:


df_environmental.columns

# In[ ]:






