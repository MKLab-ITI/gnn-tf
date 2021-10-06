def batches(data, batch_size=None):
    n = len(list(data.values())[0])
    if batch_size is None:
        batch_size = n
    batch_start = 0
    while batch_start < n:
        batch_end = min(batch_start+batch_size, n)
        yield {key: data[key][batch_start:batch_end] for key in data}
        batch_start = batch_end
