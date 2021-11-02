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

INPUT_IMAGE = "input.bmp"
KERNEL_SIZE = 11

# ================================================================================

def resizeImg(img, multiplier):
    img = cv2.resize(img, (int (img.shape[1] * multiplier), int (img.shape[0] * multiplier)))
    return img

def brightPass(img, threshold = 1.55):
    copy = img.copy()

    for i in range(copy.shape[0]):
        for j in range(copy.shape[1]):
            soma = sum(copy[i][j])
            if soma < threshold:
                copy[i][j] = (0.0, 0.0, 0.0)

    return copy


def gBlur(img, sigma):
    """Gaussian blur."""
    sm_img = img.copy()
    sm_img = resizeImg(sm_img, 0.5)

    g_blur = cv2.GaussianBlur(sm_img, (0, 0), sigma)    

    return g_blur

def bloomImg(img, mask):
    return cv2.addWeighted(img, 0.8, mask, 0.2, 0.0)

def main():
    img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_COLOR)
    if img is None:
        print("Erro abrindo a imagem.\n")
        sys.exit()

    img = img.reshape((img.shape[0], img.shape[1], img.shape[2]))
    img = img.astype(np.float32) / 255

    # 1 - Mask

    mask = brightPass(img)
    mask_blur1 = gBlur(mask, 20)
    mask_blur2 = gBlur(mask, 10)
    mask_blur3 = gBlur(mask, 5)
    mask_blur4 = gBlur(mask, 2.5)

    # mask_final = np.add(mask_blur1, mask_blur2, mask_blur3)
    mask_final = mask_blur1 + mask_blur2 + mask_blur3 + mask_blur4

    cv2.imwrite(f"001 - out-{INPUT_IMAGE}", mask * 255)
    cv2.imwrite(f"002 - out-{INPUT_IMAGE}", resizeImg(mask_blur1, 2) * 255)
    cv2.imwrite(f"003 - out-{INPUT_IMAGE}", resizeImg(mask_blur2, 2) * 255)
    cv2.imwrite(f"004 - out-{INPUT_IMAGE}", resizeImg(mask_blur3, 2) * 255)
    cv2.imwrite(f"004 - out-{INPUT_IMAGE}", resizeImg(mask_blur4, 2) * 255)
    cv2.imwrite(f"009 - out-{INPUT_IMAGE}", resizeImg(mask_final, 2) * 255)
    cv2.imwrite(f"0099 - out-{INPUT_IMAGE}", bloomImg(img, resizeImg(mask_final, 2)) * 255)

    # Gaussian Blur
    # img_gblur1 = gBlur(img, 20)

    # img_gblur2 = gBlur(img, 10)
    # out1 = cv2.resize(img_gblur2, (int (img.shape[1]), int (img.shape[0])))

    # cv2.imwrite(f"01 - out-{INPUT_IMAGE}", img_gblur1 * 255)
    # cv2.imwrite(f"02 - out-{INPUT_IMAGE}", out1 * 255)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# ===============================================================================
