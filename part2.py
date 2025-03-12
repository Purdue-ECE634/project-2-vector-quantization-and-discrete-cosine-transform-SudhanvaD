import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import argparse
def zigzag_index(n=8):
    indices = []
    for s in range(2 * n - 1):
        diagonal = []
        
        for i in range(n):
            j = s - i
            
            if 0 <= j < n:
                diagonal.append((i, j))
        
        if s % 2 == 0:
            diagonal.reverse()
        indices.extend(diagonal)
    return indices


def block_dct(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def block_idct(dct_block):
    return idct(idct(dct_block.T, norm='ortho').T, norm='ortho')

def process_image(image, K_coeff):
    h, w = image.shape
    reconstructions = {K: np.zeros_like(image, dtype=np.float32) for K in K_coeff}
    zigzag = zigzag_index(8)
    
    
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = image[i:i+8, j:j+8].astype(np.float32)
            dct_block = block_dct(block)
            
            for K in K_coeff:
                mask = np.zeros((8, 8), dtype=np.float32)
                for idx in range(K):
                    pos = zigzag[idx]
                    mask[pos] = dct_block[pos]
                r_block = block_idct(mask)
                reconstructions[K][i:i+8, j:j+8] = r_block
    return reconstructions

def compute_psnr(original, reconstructed):
    
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10((255.0 ** 2) / mse)
    return psnr

def main():
    parser = argparse.ArgumentParser(description="Vector Quantization of a grayscale image")
    parser.add_argument('image_path', help='Path to the input image file')
    parser.add_argument('--training_folder', type=str, default=None, 
                        help='Optional: Path to a folder with training images (.png, .tif)')
    
    args = parser.parse_args()
    image_path = args.image_path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


    K_coeff = [2, 4, 8, 16, 32]
    h, w = image.shape
    image = image[:(h//8)*8, :(w//8)*8]
    reconstructions = process_image(image, K_coeff)
    psnr_values = []
    for K in K_coeff:
        psnr = compute_psnr(image, reconstructions[K])
        psnr_values.append(psnr)
        print(f"PSNR for K={K}: {psnr:.2f} dB")

if __name__ == "__main__":
    main()
# Command to run
# python part2.py image_path