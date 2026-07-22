import pandas as pd

COLS_NEEDED = [
    'opinion_id', 'cluster_id', 'docket_id',
    'cluster_date_filed', 'signer_role',
    'judge_name', 'judge_id', 'fjc_id',
    'disposition_unified', 'case_type_unified',
    'plaintiff_type_unified', 'defendant_type_unified',
    'judge_name_source', 'judge_id_source',
    'disposition_unified_source', 'case_type_unified_source',
    'plaintiff_type_unified_source', 'defendant_type_unified_source',
]

df = pd.read_csv('dc_opinions.csv', usecols=COLS_NEEDED)

# Build era variable from cluster_date_filed
year = pd.to_datetime(df['cluster_date_filed'], errors='coerce').dt.year
df['era'] = pd.cut(
    year,
    bins=[-float('inf'), 1979, 1989, 2019, float('inf')],
    labels=['pre-1980', '1980-1989', '1990-2019', '2020+'],
)

# Treat NaN signer_role as 'unclear' for stratification; keep original column intact
df['signer_role'] = df['signer_role'].fillna('unclear')

# Keep only the three target eras
df = df[df['era'].isin(['pre-1980', '1980-1989', '1990-2019'])].copy()

# Draw stratified sample: 30 per cell, or all rows if fewer than 30
cells = []
for (era, role), group in df.groupby(['era', 'signer_role'], observed=True):
    n = min(30, len(group))
    cells.append(group.sample(n=n, random_state=43))

sample = (
    pd.concat(cells)
    .sample(frac=1, random_state=43)
    .reset_index(drop=True)
)

# Rename pipeline prediction columns with p_ prefix
sample = sample.rename(columns={
    'judge_name':                    'p_judge_name',
    'judge_id':                      'p_judge_id',
    'fjc_id':                        'p_fjc_id',
    'signer_role':                   'p_signer_role',
    'disposition_unified':           'p_disposition',
    'case_type_unified':             'p_case_type',
    'plaintiff_type_unified':        'p_plaintiff_type',
    'defendant_type_unified':        'p_defendant_type',
    'judge_name_source':             'p_judge_name_source',
    'judge_id_source':               'p_judge_id_source',
    'disposition_unified_source':    'p_disposition_source',
    'case_type_unified_source':      'p_case_type_source',
    'plaintiff_type_unified_source': 'p_plaintiff_type_source',
    'defendant_type_unified_source': 'p_defendant_type_source',
})

OUTPUT_COLS = [
    'opinion_id', 'cluster_id', 'docket_id', 'era', 'p_signer_role',
    'p_judge_name', 'p_judge_id', 'p_fjc_id',
    'p_disposition', 'p_case_type', 'p_plaintiff_type', 'p_defendant_type',
    'p_judge_name_source', 'p_judge_id_source',
    'p_disposition_source', 'p_case_type_source',
    'p_plaintiff_type_source', 'p_defendant_type_source',
]
sample[OUTPUT_COLS].to_csv('gabriel_validation_sample.csv', index=False)
print(f"Saved gabriel_validation_sample.csv — {len(sample)} rows")
print(sample.groupby(['era', 'p_signer_role']).size().to_string())

# Create labels template: opinion_id + empty gt_ columns
labels = sample[['opinion_id']].copy()
for col in ['gt_judge_name', 'gt_signer_role', 'gt_disposition', 'gt_case_type', 'gt_notes']:
    labels[col] = ''
labels.to_csv('gabriel_validation_labels.csv', index=False)
print(f"\nSaved gabriel_validation_labels.csv — {len(labels)} rows")

# Print the allowed case_type vocabulary for reference
vocab = sorted(pd.read_csv('dc_opinions.csv', usecols=['case_type_unified'])['case_type_unified'].dropna().unique())
print(f"\ncase_type_unified vocabulary ({len(vocab)} values):")
print(vocab)
