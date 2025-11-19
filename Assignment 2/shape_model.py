# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 18:44:44 2025

@author: tanja
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def drawshape(shape, col="r-", close=True):
    
    #shape = np.asarray(shape).ravel()
    if close:
        pts = np.concatenate([shape, shape[:1]])
    else:
        pts = shape

    plt.plot(pts.real, pts.imag, col)

# From Assignment 1
def estimate_s_R_t(P, Q):
    # Centroids
    p_mean = P.mean(axis=0)
    q_mean = Q.mean(axis=0)

    # Centered coordinates
    Pc = P - p_mean
    Qc = Q - q_mean

    # Cross-covariance
    S = Pc.T @ Qc
    U, Sigma, Vt = np.linalg.svd(S)

    # Rotation
    R = Vt.T @ U.T

    # Reflection fix
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Scale
    s = Sigma.sum() / np.sum(Pc**2)

    # Translation
    t = q_mean - s * (R @ p_mean)

    return s, R, t


def similarity_transform(shape, s, R, t):
    P = np.column_stack([shape.real, shape.imag])
    transformed = s * (P @ R.T) + t
    
    return transformed[:, 0] + 1j * transformed[:, 1]


def procrustes_alignment(shapes, iterations=5, tol=1e-6):
    n_points, M = shapes.shape

    # Step 1: use the first shape as reference and align all shapes to it
    ref = shapes[:, 0]

    aligned = np.empty_like(shapes, dtype=np.complex128)
    for k in range(M):
        P = np.column_stack([shapes[:, k].real, shapes[:, k].imag])
        Q = np.column_stack([ref.real, ref.imag])
        s, R, t = estimate_s_R_t(P, Q)
        aligned[:, k] = similarity_transform(shapes[:, k], s, R, t)

    # Step 2: initial mean shape in the same coordinate system as ref
    mean_shape = aligned.mean(axis=1)

    for idx in range(iterations):
        #old_mean = mean_shape.copy()

        # Step 3: align current mean shape to the first shape (ref)
        P = np.column_stack([mean_shape.real, mean_shape.imag])
        Q = np.column_stack([ref.real, ref.imag])
        s, R, t = estimate_s_R_t(P, Q)
        mean_shape = similarity_transform(mean_shape, s, R, t)

        # Step 4: align all shapes to this updated mean shape
        for k in range(M):
            P = np.column_stack([aligned[:, k].real, aligned[:, k].imag])
            Q = np.column_stack([mean_shape.real, mean_shape.imag])
            s, R, t = estimate_s_R_t(P, Q)
            aligned[:, k] = similarity_transform(aligned[:, k], s, R, t)

        # Step 5: update mean shape
        mean_shape = aligned.mean(axis=1)

        # Step 6: convergence check
        #num = np.linalg.norm(mean_shape - old_mean)
        #den = np.linalg.norm(old_mean)
        #rel_change = num / den
        #if rel_change < tol:
        #    print(idx + 1)      # converges very quickly, already after 3 iterations
        #    break

    return aligned, mean_shape


def pca_model(aligned_shapes):
    n_points, M = aligned_shapes.shape

    X = aligned_shapes.real
    Y = aligned_shapes.imag
    Z = np.vstack([X, Y])

    mean_vec = Z.mean(axis=1, keepdims=True)
    Zc = Z - mean_vec
    C = Zc @ Zc.T / (M - 1)

    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    return eigvals, eigvecs, mean_vec

if __name__ == "__main__":
    # Load models and images
    models_dict = loadmat(r"C:\Users\tanja\OneDrive\Skrivbord\Nuvarande\Medicinsk bildanalys\Assignment 2\data\models.mat")
    models = models_dict["models"]

    dmsa_images_dict = loadmat(r"C:\Users\tanja\OneDrive\Skrivbord\Nuvarande\Medicinsk bildanalys\Assignment 2\data\dmsa_images.mat")
    dmsa_images = dmsa_images_dict["dmsa_images"]

    shapes = models.copy()

    # Procrustes alignment
    aligned_shapes, mean_shape = procrustes_alignment(shapes)

    # PCA
    eigvals, eigvecs, mean_vec = pca_model(aligned_shapes)

    # Overlay mean kidney shape on one patient image
    pat_nbr = 15
    target = models[:, pat_nbr]  # target shape in image coordinate system

    # Estimate similarity transform that maps mean_shape -> target
    P = np.column_stack([mean_shape.real, mean_shape.imag])
    Q = np.column_stack([target.real, target.imag])
    s, R, t = estimate_s_R_t(P, Q)

    # Put mean shape into this pose
    mean_on_image = similarity_transform(mean_shape, s, R, t)

    plt.figure()
    plt.imshow(dmsa_images[:, :, pat_nbr], cmap="gray", origin="lower")
    drawshape(mean_on_image, "r.-", close=True)
    plt.title(f"Mean kidney shape overlaid (patient {pat_nbr + 1})")
    plt.gca().set_aspect("equal")
    plt.show()

    # Eigenvalues
    plt.figure()
    plt.plot(np.arange(1, len(eigvals) + 1), eigvals, "o-")
    plt.xlabel("Mode")
    plt.ylabel("Eigenvalue")
    plt.title("Eigenvalues of the shape model")
    plt.grid(True)
    plt.show()

    # Cumulative variance explained
    var_explained = eigvals / eigvals.sum()
    cumvar = np.cumsum(var_explained)

    plt.figure()
    plt.plot(np.arange(1, len(cumvar) + 1), cumvar, "o-")
    plt.xlabel("Number of modes")
    plt.ylabel("Cumulative variance explained")
    plt.title("Cumulative Variance Explained by PCA Modes")
    #plt.ylim([0, 1.05])
    plt.grid(True)
    plt.show()

    # Visualise first 3 modes for pat_nbr
    n_points, M = aligned_shapes.shape
    
    x_mean = mean_vec[:n_points, 0]
    y_mean = mean_vec[n_points:, 0]
    mean_shape_pca = x_mean + 1j * y_mean
    
    for m in range(3):
        # Standard deviation movement along mode m
        b = 2 * np.sqrt(eigvals[m])      # 2 std deviations
        mode_vec = eigvecs[:, m]
    
        # Create plus and minus shapes
        plus = mean_vec[:, 0] + b * mode_vec
        minus = mean_vec[:, 0] - b * mode_vec
    
        # Back to complex shapes
        x_plus, y_plus = plus[:n_points], plus[n_points:]
        x_minus, y_minus = minus[:n_points], minus[n_points:]
        shape_plus = x_plus + 1j * y_plus
        shape_minus = x_minus + 1j * y_minus
    
        # Use the same (s, R, t) from the mean shape for the mode
        plus_img = similarity_transform(shape_plus, s, R, t)
        minus_img = similarity_transform(shape_minus, s, R, t)
    
        plt.figure()
        plt.imshow(dmsa_images[:, :, pat_nbr], cmap="gray", origin="lower")
        drawshape(mean_on_image, "w-", close=True)
        drawshape(plus_img, "r-", close=True)
        drawshape(minus_img, "b-", close=True)
        plt.gca().set_aspect("equal")
        plt.legend(["Mean", "+ Mode", "- Mode"])
        plt.title(f"Mode {m+1} on kidney {pat_nbr + 1}")
        plt.show()
        
        plt.figure()
        drawshape(mean_shape_pca, "k.-", close=True)

        for scale, col in [
            (1,  "r-"),   
            (2,  "g-"),
            (-1, "b-"),   
            (-2, "c-"),
        ]:
            b = scale * np.sqrt(eigvals[m])
            mode_vec = eigvecs[:, m]
            deformed = mean_vec[:, 0] + b * mode_vec
            x_def, y_def = deformed[:n_points], deformed[n_points:]
            shape_def = x_def + 1j * y_def
            drawshape(shape_def, col=col, close=True)
    
        plt.gca().set_aspect("equal")
        plt.legend(["Mean", "+1σ", "+2σ", "-1σ", "-2σ"])
        plt.title(f"Mode {m+1}")
        plt.show()
        