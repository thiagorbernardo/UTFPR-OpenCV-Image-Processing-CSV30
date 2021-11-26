# ===============================================================================
# Trabalho 4
# Autores: Thiago Ramos e João Lucas
# Universidade Tecnológica Federal do Paraná
# ===============================================================================

import sys
import timeit
import numpy as np
import cv2
from math import *

sys.setrecursionlimit(3000)

# ===============================================================================

def calculateComponents(img):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    img2 = np.zeros((output.shape))
    img2 = img2.astype(np.uint8)

    for i in range(0, nb_components):
        if sizes[i] >= 30:
            img2[output == i + 1] = 1

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img2, connectivity=4)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    median = np.median(sizes)

    averageComponentSize = np.mean(sizes)

    components = 0
    for i in range(0, nb_components):
        if sizes[i] >= median:
            components += round(np.sum(sizes[i])/median)
        else:
            components += 1

    return components

# ===============================================================================

def adaptiveBinarization(img):
    copy = img.copy()

    copy = cv2.adaptiveThreshold(copy, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -10)

    erode = cv2.morphologyEx(copy, cv2.MORPH_ERODE, (3, 3), iterations=2)
    return erode


def main():
    for INPUT_IMAGE in ["60", "82", "114", "150", "205"]:
        img = cv2.imread(f"{INPUT_IMAGE}.bmp", cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Erro abrindo a imagem.\n")
            sys.exit()

        # Image binary
        imgBin = adaptiveBinarization(img)
        # cv2.imwrite(f"out-{INPUT_IMAGE}-binary.bmp", imgBin)

        # Image components
        components = calculateComponents(imgBin)
        print(f"Image: {INPUT_IMAGE} - Result: {components}")

        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# ===============================================================================
