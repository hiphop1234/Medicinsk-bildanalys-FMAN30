# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 18:22:49 2025

@author: tanja
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Modified function drawshape where shape is complex
def drawshape_comp(shape, conList, col):
    shape = np.asarray(shape).ravel()

    for i in range(len(conList)):
        if conList[i, 2] == 0:
            pts = shape
        else:
            pts = np.concatenate([shape, shape[:1]])

        plt.plot(pts.real, pts.imag, col)
        
# Resampling using interpolation
def resampling(shape, num_points = 14):    
    contour = np.concatenate([shape, shape[:1]])
    
    distance = np.abs(np.diff(contour))
    sum_arc = np.concatenate([[0], np.cumsum(distance)])
    tot_len = sum_arc[-1]
    
    redone = np.linspace(0, tot_len, num_points + 1)[:-1]
    
    new_shape = np.interp(redone, sum_arc, contour.real) + 1j * np.interp(redone, sum_arc, contour.imag)
    
    return new_shape

if __name__ == "__main__":
    
    # PART 1
    # Load DMSA images and manual segmentation
    man_seg_dict = loadmat(r"C:\Users\tanja\OneDrive\Skrivbord\Nuvarande\Medicinsk bildanalys\Assignment 2\data\man_seg.mat")
    dmsa_images_dict = loadmat(r"C:\Users\tanja\OneDrive\Skrivbord\Nuvarande\Medicinsk bildanalys\Assignment 2\data\dmsa_images.mat")

    man_seg = man_seg_dict["man_seg"].ravel()
    dmsa_images = dmsa_images_dict["dmsa_images"]

    # Extract x- and y-coordinates
    Xmcoord = np.real(man_seg);
    Ymcoord = np.imag(man_seg);

    # Choose patient and look at image
    pat_nbr = 0;
    
    # Plot left kidney segmentation
    plt.figure()
    plt.title("Unsampled")
    plt.imshow(dmsa_images[:, :, pat_nbr], cmap = 'gray', origin = 'lower')
    plt.gca().set_aspect('equal')
    plt.show()
    
    conList = np.array([[1, man_seg.size, 1]])
    drawshape_comp(man_seg, conList,'.-r')
    
   # Plot resampled left kidney segmentation using interpolation
    resampled = resampling(man_seg,14)
    resampled_closed = np.concatenate([resampled, resampled[:1]])
    
    plt.figure()
    plt.title("Sampled")
    plt.imshow(dmsa_images[:, :, pat_nbr], cmap = "gray", origin = "lower")
    plt.plot(resampled_closed.real, resampled_closed.imag, "r.-")  # closed contour
    plt.gca().set_aspect("equal")
    plt.show()
    