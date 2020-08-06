from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import streamlit as st
from PIL import Image

from skimage import io
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

from plot_utils import plot_utils


plt.style.use("ggplot")

plt.rcParams["figure.figsize"] = (20,12)

user_image = st.sidebar.file_uploader("Choose an image (.jpg)", type = "jpg")

k = st.sidebar.slider("Number of colours in final image", min_value = 1, max_value = 256, value = 16)

if user_image is not None:
    img = io.imread(user_image)
    ax = plt.axes(xticks=[], yticks=[])
    ax.imshow(img)

    st.image(img, caption = "Image successfully uploaded...", use_column_width = True)
    # st.write(img.shape)

    img_data = (img / 255.0).reshape(-1, 3)
    # st.write(img_data.shape)

    plot = plt.figure(1, figsize = (8, 8))
    ax_1 = plot.add_subplot(121)
    ax_2 = plot.add_subplot(122)
    

    orig_img_plt = plot_utils(img_data, title='Original Image Colour Space (16,777,216 possible colours)')
    orig_img_plt.colorSpace()
    st.pyplot()
    

    kmeans = MiniBatchKMeans(k).fit(img_data)
    k_colours = kmeans.cluster_centers_[kmeans.predict(img_data)]

    k_img_plt = plot_utils(img_data, colors=k_colours, title="Reduced color space: 16 colors")
    k_img_plt.colorSpace()
    st.pyplot()

    k_img = np.reshape(k_colours, img.shape)


    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('K-means Image Compression', fontsize=40)

    
    ax1.set_title(f'Compressed Image {k} colours')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.imshow(k_img)
    
    ax2.set_title('Original Image (16,777,216 colours)')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.imshow(img)
    
    plt.subplots_adjust(top=0.85)
    

    st.pyplot()


    
