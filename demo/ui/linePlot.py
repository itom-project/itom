# open a figure
fig = figure()

# create a randomly filled 1d data object
dObj = dataObject.randN([1, 100])

# plot the data as line plot
fig.plot(dObj)
