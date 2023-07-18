import numpy as np

IMG_EXTENT = [54.866, 56.193872, -20.5849, -21.6499]


def get_arrays_cols(df):
    arrays_cols = []
    for c in df.columns:
        if type(df[c].iloc[0]) == np.ndarray:
            arrays_cols.append(c)
    return arrays_cols


def pad(df):
    """
    Pad the data in order to make its size compatible with the unet
    Input : dataframe
    Output : dataframe with padding
    """
    df_out = df.copy()
    arrays_cols = get_arrays_cols(df_out)
    for c in arrays_cols:
        for i in range(len(df_out)):
            df_out[c][i] = np.pad(df_out[c][i], ((5,5), (2,3)), mode='reflect')
    return df_out


def crop(df):
    """
    Crop the data
    Input : dataframe
    Output : cropped dataframe
    """
    df_out = df.copy()
    for c in df_out.columns:
        if type(df_out[c][0]) == np.ndarray:
            for i in range(len(df_out)):
                df_out[c][i] = df_out[c][i][5:-5, 2:-3]
    return df_out


def param_to_array(arrays_serie):
    """
    Transforms a pandas series into a big numpy array of shape B x H x W
    """
    array = np.zeros((len(arrays_serie), arrays_serie[0].shape[0], arrays_serie[0].shape[1]), dtype=np.float32)
    for i in range(len(arrays_serie)):
        array[i, :, :] = arrays_serie[i]
    return array


def df_to_array(df):
    """
    transforms a pandas dataframe into a big numpy array of shape B x H x W x C
    """
    arrays_cols = get_arrays_cols(df)
            
    array = np.zeros((len(df), df[arrays_cols[0]].iloc[0].shape[0], df[arrays_cols[0]].iloc[0].shape[1], len(arrays_cols)), dtype=np.float32)

    for i in range(len(df)):
        for i_c, c in enumerate(arrays_cols):
            array[i, :, :, i_c] = df[c].iloc[i]
    return array


def get_ind_terre_mer_500m():
    filepath = '/cnrm/recyf/Data/users/danjoul/dataset/static_G9KP_SURFIND.TERREMER.npy'
    return np.load(filepath)


def get_shape_500m():
    field_500m = np.load('/cnrm/recyf/Data/users/danjoul/dataset/data_train/G9KP_2021-01-01T00:00:00Z_rr.npy')
    return field_500m[:, :, 0].shape


def get_shape_2km5(resample='c'):
    if resample == 'c':
        field_2km5 = np.load('/cnrm/recyf/Data/users/danjoul/dataset/data_train/oper_c_2021-01-01T00:00:00Z_rr.npy')
        return field_2km5[:, :, 0].shape
    else:
        field_2km5 = np.load('/cnrm/recyf/Data/users/danjoul/dataset/data_train/oper_r_2021-01-01T00:00:00Z_rr.npy')
        return field_2km5[:, :, 0].shape