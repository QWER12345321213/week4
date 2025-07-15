
def encode_ordinal(data, columns):
    encoders = {col: {val: i for i, val in enumerate(sorted(set(data[col])))} for col in columns}
    encoded = [[encoders[col].get(row[col], -1) for col in columns] for _, row in data.iterrows()]
    return encoded

def encode_onehot(data, columns):
    categories = {col: sorted(set(data[col])) for col in columns}
    encoded = []
    for _, row in data.iterrows():
        row_encoded = []
        for col in columns:
            vec = [0] * len(categories[col])
            if row[col] in categories[col]:
                idx = categories[col].index(row[col])
                vec[idx] = 1
            row_encoded.extend(vec)
        encoded.append(row_encoded)
    return encoded

def get_feature(train_df, test_df, encoding_method, encoding_columns):
    if encoding_method == 'ordinal':
        return encode_ordinal(train_df, encoding_columns), encode_ordinal(test_df, encoding_columns)
    elif encoding_method == 'one-hot':
        return encode_onehot(train_df, encoding_columns), encode_onehot(test_df, encoding_columns)
