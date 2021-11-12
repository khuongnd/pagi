import matplotlib.pyplot as plt

def plot_bar(names, values, file_name=None):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(names, values)
    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)


if __name__ == "__main__":
    names = ['C', 'C++', 'Java', 'Python', 'PHP']
    values = [23, 17, 35, 29, 12]
    plot_bar(names, values)