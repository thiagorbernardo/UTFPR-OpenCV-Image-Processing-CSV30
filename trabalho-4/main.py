# ===============================================================================
# Exemplo: blur .
# -------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu
# Universidade Tecnológica Federal do Paraná
# ===============================================================================
# Trabalho 2
# Autores: Thiago Ramos e João Lucas
# ===============================================================================

import sys
import timeit
import numpy as np
import cv2
from math import *
import statistics as st

sys.setrecursionlimit(3000)

# ===============================================================================


def colorfullImg(img):
    qtd, labels = cv2.connectedComponents(1 - img, connectivity=8)
    qtd = qtd - 1

    unique, counts = np.unique(labels, return_counts=True)
    counter = dict(zip(unique, counts))


    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    # cv2.imshow('labeled.png', labeled_img)

# ===============================================================================

def magic(img):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(cv2.GaussianBlur(img, (0,0), 3), connectivity=4)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    img2 = np.zeros((output.shape))
    img2 = img2.astype(np.uint8)

    for i in range(0, nb_components):
        if sizes[i] >= 30 and sizes[i] <= 3000:
            img2[output == i + 1] = 1


    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img2, connectivity=4)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # print("std", np.std(sizes))
    print("sizes", np.unique(sizes))
    median = np.median(sizes)
    print("median", np.median(sizes))
    # error = np.std(sizes) / sqrt(nb_components)
    # print("error", error)
    averageComponentSize = np.mean(sizes)
    print("mean", averageComponentSize)

    components = 0
    for i in range(0, nb_components):
        if sizes[i] >= median:
            components += round(np.sum(sizes[i])/median)
            # components += round(np.sum(sizes[i])/(averageComponentSize - 3*error))
        else:
            components += 1

    return [components, img2]

# ===============================================================================

def resizeImg(img, multiplier):
    img = cv2.resize(
        img, (int(img.shape[1] * multiplier), int(img.shape[0] * multiplier))
    )
    return img


def otsuBinarization(img):
    """
    1. Otsu's Binarization
    2. Tirar blobs e ruídos, procurar menores e maiores componentes para remover pintando
    3. Contagem componentes conectados e pixel de arroz
    4. Total de pixeis de arroz / contagem de componentes conectados
    5. Blob > Tamanho médio do arroz ? tamanho do blob / tamanho médio do arroz: 1, 2, 3 : 1
    """

    copy = img.copy()

    copy = np.where(copy > 0.75, 1, 0)
    copy = copy.astype(np.uint8)

    # otsuImg = cv2.adaptiveThreshold(copy, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # threshold, copy = cv2.threshold(copy.astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    opening = cv2.morphologyEx(copy, cv2.MORPH_OPEN, (3, 3), iterations=3)
    return opening

    return copy


def main():
    # for INPUT_IMAGE in ["205"]:#, "82", "114", "150", "205"]:
    for INPUT_IMAGE in ["60", "82", "114", "150", "205"]:
        img = cv2.imread(f"{INPUT_IMAGE}.bmp", cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Erro abrindo a imagem.\n")
            sys.exit()

        # img = img.reshape((img.shape[0], img.shape[1], img.shape[2]))
        img = img.reshape((img.shape[0], img.shape[1], 1))
        img = img.astype(np.float32) / 255

        # cv2.imwrite(f"1 - bin - out-{INPUT_IMAGE}", ( 255 - img ) * 255)
        # cv2.imwrite(f"2 - bin - out-{INPUT_IMAGE}", img * 255)
        cv2.imwrite(f"out/gray-{INPUT_IMAGE}.bmp", img * 255)
        imgBin = otsuBinarization(img)
        cv2.imwrite(f"out/otsu-{INPUT_IMAGE}.bmp", imgBin * 255)

        [components, imgMagic] = magic(imgBin)
        print(f"{INPUT_IMAGE} - {components}")
        cv2.imwrite(f"out/magic-{INPUT_IMAGE}.bmp", imgMagic * 255)

        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# ===============================================================================
