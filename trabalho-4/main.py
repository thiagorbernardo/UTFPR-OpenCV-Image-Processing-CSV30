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
    print("median", np.median(sizes))

    averageComponentSize = np.mean(sizes)
    print("mean", averageComponentSize)

    components = 0
    for i in range(0, nb_components):
        if sizes[i] >= median:
            components += round(np.sum(sizes[i])/median)
        else:
            components += 1

    return [components, img2]

# ===============================================================================

def otsuBinarization(img):
    """
    1. Otsu's Binarization
    2. Tirar blobs e ruídos, procurar menores e maiores componentes para remover pintando
    3. Contagem componentes conectados e pixel de arroz
    4. Total de pixeis de arroz / contagem de componentes conectados
    5. Blob > Tamanho médio do arroz ? tamanho do blob / tamanho médio do arroz: 1, 2, 3 : 1
    """

    copy = img.copy()

    copy = cv2.adaptiveThreshold(copy, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -10)

    erode = cv2.morphologyEx(copy, cv2.MORPH_ERODE, (3, 3), iterations=2)
    return erode


def main():
    # for INPUT_IMAGE in ["60"]:#, "82", "114", "150", "205"]:
    for INPUT_IMAGE in ["60", "82", "114", "150", "205"]:
        img = cv2.imread(f"{INPUT_IMAGE}.bmp", cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Erro abrindo a imagem.\n")
            sys.exit()

        # cv2.imwrite(f"1 - bin - out-{INPUT_IMAGE}", ( 255 - img ) * 255)
        # cv2.imwrite(f"2 - bin - out-{INPUT_IMAGE}", img * 255)
        cv2.imwrite(f"out/{INPUT_IMAGE}-gray.bmp", img)
        imgBin = otsuBinarization(img)
        cv2.imwrite(f"out/{INPUT_IMAGE}-otsu.bmp", imgBin)

        [components, imgMagic] = calculateComponents(imgBin)
        print(f"{INPUT_IMAGE} - {components}")
        cv2.imwrite(f"out/{INPUT_IMAGE}-magic.bmp", imgMagic * 255)

        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# ===============================================================================
