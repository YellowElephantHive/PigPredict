# coding: utf-8

# In[ ]:


from skimage import color
import matplotlib.pyplot as plt
import numpy as np


def pdShowLabColor(lab):
    if not (lab.columns.contains('L') & lab.columns.contains('a') & lab.columns.contains('b')):
        raise ValueError('input need contain colume L a b')
        return

    showTitle = lab.columns.contains('LAB')

    if showTitle:
        name = lab.loc[:, ['LAB']]
        name = name.values
        name = name.reshape(-1, )

    lab = lab.loc[:, ['L', 'a', 'b']]
    lab = lab.as_matrix()

    lab = lab.reshape(-1, 1, 3)
    rgb = color.lab2rgb(lab)

    plt.figure(figsize=(20, (int(len(lab) / 10) + 1) * 2))

    for i in range(len(rgb)):
        r = rgb[i]
        r = r.reshape(-1, 1, 3)

        ax = plt.subplot(int(len(rgb) / 10) + 1, 10, i + 1)
        if showTitle:
            ax.set_title(name[i])
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.imshow(r, aspect='auto')
    plt.show()


def npShowLabColor(lab, label=[]):
    showTitle = (len(label) != 0)
    if showTitle:
        label = label.reshape(-1, )
        if len(label) != len(lab):
            showTitle = False
            raise ValueError('lab length is not equal label')

    lab = lab.reshape(-1, 1, 3)
    rgb = color.lab2rgb(lab)

    plt.figure(figsize=(20, (int(len(lab) / 10) + 1) * 2))

    for i in range(len(rgb)):
        r = rgb[i]
        r = r.reshape(-1, 1, 3)

        ax = plt.subplot(int(len(rgb) / 10) + 1, 10, i + 1)

        if showTitle:
            ax.set_title(label[i])

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.imshow(r, aspect='auto')
    plt.show()
