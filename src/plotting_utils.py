import matplotlib.pyplot as plt


def plot_trajectories(time_grid, trajectories, cnt=None, y_label=None, title=None, plot_mean=False):
    if cnt is None:
        cnt = trajectories.shape[0]
    if title is None:
        title = y_label
    plt.ticklabel_format(style='plain', useOffset=False)
    plt.plot(time_grid, trajectories[:cnt].T)
    plt.grid()
    plt.xlabel("$t$")
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
    if plot_mean:
        plt.plot(time_grid, trajectories.mean(axis=0), label="Mean", color='blue')
        plt.fill_between(time_grid,
                         trajectories.mean(axis=0) -
                         trajectories.std(axis=0),
                         trajectories.mean(axis=0) +
                         trajectories.std(axis=0),
                         label="Mean ± Std",
                         alpha=0.4,
                         color='orange')
        plt.grid()
        plt.xlabel("$t$")
        plt.ylabel(y_label)
        plt.title("Mean " + title)
        plt.show()
