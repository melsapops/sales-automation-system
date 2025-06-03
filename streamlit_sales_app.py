# Fill missing industry with 'Unknown'
        if 'industry' in df.columns:
            df['industry'] = df['industry'].fillna('Unknown')