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

sys.setrecursionlimit(3000)
# ===============================================================================

# INPUT_IMAGE = "150.bmp"

# ================================================================================

def resizeImg(img, multiplier):
    img = cv2.resize(img, (int (img.shape[1] * multiplier), int (img.shape[0] * multiplier)))
    return img

def otsuBinarization(img):
    """[summary]

    1. Otsu's Binarization
    2. Tirar blobs e ruídos, procurar menores e maiores componentes para remover pintando
    3. Contagem componentes conectados e pixel de arroz
    4. Total de pixeis de arroz / contagem de componentes conectados
    5. Blob > Tamanho médio do arroz ? tamanho do blob / tamanho médio do arroz: 1, 2, 3 : 1 
    Args:
        img ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    copy = img.copy()
    copy = cv2.resize(copy,None, fx = 2, fy = 2, interpolation = cv2.INTER_NEAREST)

    copy = np.where(copy < 65, 0, 1)
    copy = copy.astype(np.uint8)

    # otsuImg = cv2.adaptiveThreshold(copy, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


    threshold, otsuImg = cv2.threshold(copy, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # print(threshold)
    # return cv2.normalize(otsuImg, otsuImg, 0, 1, cv2.NORM_MINMAX)
    # opening = cv2.morphologyEx(otsuImg, cv2.MORPH_DILATE, (3,3), iterations=2)
    # return opening

    opening = cv2.morphologyEx(otsuImg, cv2.MORPH_DILATE, (3,3), iterations=7)
    return opening

    return otsuImg
def main():
    for INPUT_IMAGE in ["60", "82", "114", "150", "205"]:
        img = cv2.imread(f"{INPUT_IMAGE}.bmp", cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Erro abrindo a imagem.\n")
            sys.exit()

        # img = img.reshape((img.shape[0], img.shape[1], img.shape[2]))
        # img = img.reshape((img.shape[0], img.shape[1], 1))
        # img = img.astype(np.float32) / 255

        # cv2.imwrite(f"1 - bin - out-{INPUT_IMAGE}", ( 255 - img ) * 255)
        # cv2.imwrite(f"2 - bin - out-{INPUT_IMAGE}", img * 255)
        img = otsuBinarization(255 - img)
        cv2.imwrite(f"1 - Ootsu Bin - out-{INPUT_IMAGE}.bmp", img * 255)

        qtd, labels = cv2.connectedComponents(1 - img, connectivity=8)
        qtd = qtd - 1

        unique, counts = np.unique(labels, return_counts=True)
        counter = dict(zip(unique, counts))
        # print(counter)

        newDict = {
            
        }
        # for i in range(0, qtd):
        #     if(counter[i] < 700 and counter[i] > 40):
        #         newDict[i] = counter[i]

        # for i in labels:
        #     count = np.sum(i)
        #     print(count > 30*30 or count < 20*20)
        #     if(count < 30*30 and count > 20*20):
        #         newLabels.append(i)
        #     # if(count > 30*30 or count < 20*20):
        #     #     labels[i] = np.ndarray(i.shape, dtype=np.uint8)
        #     #     count = 0

        #     if(count != 0):
        #         # print(count)
        #         riceQtd += 1

        # print(newDict)
        print(INPUT_IMAGE, qtd)#, len(newDict.keys()))

        # Map component labels to hue val
        label_hue = np.uint8(179*labels/np.max(labels))
        blank_ch = 255*np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

        # cvt to BGR for display
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

        # set bg label to black
        labeled_img[label_hue==0] = 0

        # cv2.imshow('labeled.png', labeled_img)
        cv2.waitKey()


        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# ===============================================================================
