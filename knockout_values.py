import sys
import pandas as pd
import numpy as np

def knockout_values(df, missing_pct=None, na_value=None, exclude=None):
    '''
    Set some values from the Pandas dataframe `df` (controlled by missing_pct) 
    to "N/A" to simulate missing data for testing an imputation pipeline or 
    model's sensitivity to missing values.
        df: Pandas DataFrame containing the complete dataset
        missing_pct: percentage of values to remove.  May be 'None' to make no
            changes, a floating-point values to drop values across `df` uniformly,
            or a dictionary of "colname: pct" pairs defining the missing percent
            for each column individually (unlisted columns will not be changed).
        na_value: Value to use for missing data.  Defaults to 'nan' for float values,
            -1 for integer values, and "NA" for strings.  
            Expects a dict of "type: value" pairs defining the sentinel value to use 
            for each type.  Omit or set to None to use defaults.
        exclude: name of column or list of column names to exclude (response variables,
            for example).  These columns will not be modified.
        returns: copy of `df` with some values missing, and a bool DataFrame indicating
            which values were set to NA.
    '''
    float_type = lambda x: str(x).startswith('float') or str(x).startswith('int')
    int_type   = lambda x: str(x).startswith('int')
    float_like = lambda x: isinstance(x, (np.floating, float, np.integer, int)) or str(x).startswith('float') or str(x).startswith('int')
    int_like   = lambda x: isinstance(x, (np.integer, int)) or str(x).startswith('int')
    default_na = lambda x: -1 if int_type(x) else (np.nan if float_type(x) else 'NA')
    is_pct     = lambda x: x >= 0.0 and x <= 1.0
    assert (missing_pct is None) or float_like(missing_pct) or isinstance(missing_pct, dict), 'missing_pct must be None, float, or dict.'
    types = df.dtypes.to_dict()
    exclude = set([]) if exclude is None else (set(exclude) if not type(exclude) == type('str') else set([exclude]))
    # Set appropriate NA values for every column
    _na_value = {c: default_na(types[c]) for c in types}
    if isinstance(na_value, dict):
        for c in _na_value:
            if c in na_value:
                _na_value[c] = na_value[c]
    na_value = _na_value
    # Set missing percent for every column
    _missing_pct = {c: None for c in df.columns}
    if float_like(missing_pct):
        assert is_pct(missing_pct), 'missing_pct must be in range [0,1] if numeric.'
        missing_pct = float(missing_pct)
        _missing_pct = {c: missing_pct for c in list(df.columns)}
    elif isinstance(missing_pct, dict):
        for c in missing_pct:
            _missing_pct[c] = missing_pct[c]
            if float_like(_missing_pct[c]):
                _missing_pct[c] = float(_missing_pct[c])
                assert is_pct(_missing_pct[c]), 'missing_pct must be in range [0,1] if numeric.'
    missing_pct = _missing_pct
    # Copy the dataframe and set up masks
    masked_df = df.copy() # working dataframe we will modify (the original is unchanged)
    mask      = pd.DataFrame(np.zeros_like(df.values, dtype=bool), columns=df.columns) # boolean mask of values we modify
    n_rows    = masked_df.shape[0]
    for c in mask.columns:
        if c in exclude:
            continue  # skip any excluded columns
        pct = missing_pct[c]
        if pct is not None:
            mask_idx = np.random.choice(n_rows, int(np.round(n_rows*pct)), replace=False)
            mask[c][mask_idx] = True
            masked_df[c] = masked_df[c].where(~mask[c], other=na_value[c]) # where replaces when bool is False; we want opposite, so we use '~'
    return (masked_df, mask)

if __name__ == '__main__':
    print("This script is not intended to be run stand-alone.")
    sys.exit(1)
