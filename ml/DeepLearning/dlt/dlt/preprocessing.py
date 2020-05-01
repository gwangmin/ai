from sklearn.preprocessing import MinMaxScaler

def preprocess(xy, minmax=False):
    '''
    Data preprocessing

    xy: x data, y data
    minmax: whether apply minmaxScaler
    '''
    if minmax:
        scaler = MinMaxScaler()
        xy = scaler.fit_transform(xy)
    return xy
