import numpy as np 
import cv2
import matplotlib.pyplot as plt
import argparse
import os
import glob
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def extract_blocks(image):
    h, w = image.shape
    bh = 4
    bw = 4


    pad_h = (bh - (h % bh)) % bh
    pad_w = (bw - (w % bw)) % bw

   
    padded_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)


    blocks = []
    

    for i in range(0, padded_image.shape[0], bh):
        for j in range(0, padded_image.shape[1], bw):
           
            block = padded_image[i:i + bh, j:j + bw].flatten()
            blocks.append(block)


    return np.array(blocks, dtype=np.float32), padded_image.shape
 

def reconstruct_image(blocks, padded_shape, original_shape):
    ph, pw = padded_shape  
    oh, ow = original_shape  
    bh = 4
    bw = 4
    reconstructed = np.zeros((ph, pw), dtype=np.uint8)

    idx = 0
    for i in range(0, ph, bh):
        for j in range(0, pw, bw):
            if idx < len(blocks):
                reconstructed[i:i+bh, j:j+bw] = blocks[idx].reshape((4,4))
                idx += 1
    return reconstructed[:oh, :ow]

def kmeans_codebook(training_vectors, codebook_size, max_iter=100):
    kmeans = KMeans(n_clusters=codebook_size, init='k-means++', max_iter=max_iter, n_init=1)
    kmeans.fit(training_vectors)
    return kmeans.cluster_centers_

def quantize_blocks(blocks, codebook):
    distances = cdist(blocks, codebook, metric='euclidean')  
    nearest_idx = np.argmin(distances, axis=1)
    quantized_blocks = codebook[nearest_idx]
    return quantized_blocks

def compute_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10((255.0 ** 2) / mse)
    return psnr

def vector_quantization(image_path, codebook_sizes, training_folder=None):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    print(f"Image shape: {image.shape}")
    cv2.imwrite("original_image.png", image)
    
    
    all_blocks = []
    
    if training_folder:
        print(f"Collecting training data from folder: {training_folder}")
        image_files = glob.glob(os.path.join(training_folder, "*.png"))
        
        if not image_files:
            raise ValueError(f"No .png or .tif images found in {training_folder}")
        
        
        for img_path in image_files:
            training_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if training_image is not None:
                blocks, padded_shape = extract_blocks(training_image)
                all_blocks.extend(blocks)
                print(f"Extracted {len(blocks)} blocks from {os.path.basename(img_path)}")
            else:
                print(f"Warning: Could not load image {img_path}")
        
    else:
        print("No training folder provided. Using the input image to create the codebook.")
        blocks, padded_shape = extract_blocks(image)
        all_blocks.extend(blocks)
        print(f"Extracted {len(blocks)} blocks from the input image.")
    
    
    all_blocks = np.array(all_blocks, dtype=np.float32)
    print(f"Total blocks collected for training: {all_blocks.shape[0]}")
    
    psnr_values = []  
    sizes = []         
    blocks, padded_shape = extract_blocks(image)
    
    for L in codebook_sizes:
        print(f"\nDesigning codebook with L = {L} using K-Means++...")
        codebook = kmeans_codebook(all_blocks, codebook_size=L, max_iter=100)
        quantized_blocks = quantize_blocks(blocks, codebook)
        reconstructed = reconstruct_image(quantized_blocks, padded_shape, image.shape)
        cv2.imwrite('vectorCodebook'+str(L)+"_image.png", reconstructed)
        psnr_value = compute_psnr(image, reconstructed)
        print(f"PSNR for codebook size {L}: {psnr_value:.2f} dB")
        psnr_values.append(psnr_value)
        sizes.append(L)
    
   
    plt.figure(figsize=(8,6))
    plt.plot(sizes, psnr_values, marker='o')
    plt.xlabel("Codebook Size (L)")
    plt.ylabel("PSNR (dB)")
    plt.title("Codebook Size vs. PSNR (K-Means++)")
    plt.grid(True)
    plt.savefig("codebook_vs_psnr_airplane.png")
    plt.show()
    print("Plot saved as 'codebook_vs_psnr.png'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vector Quantization")
    parser.add_argument('image_path', help='Path to the input image file')
    parser.add_argument('--training_folder', type=str, default=None)
    
    args = parser.parse_args()
    image_path = args.image_path
    training_folder = args.training_folder
    
    vector_quantization(image_path, codebook_sizes=[128, 256, 512, 1024, 2048,4096], training_folder=training_folder)
    
# Command to Run on Single Image
# --python part1.py image_path

# Command to Run on Collection
# --python part1.py image_path --training_folder="train"


