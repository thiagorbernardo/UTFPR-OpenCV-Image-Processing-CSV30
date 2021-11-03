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

INPUT_IMAGE = "input1.bmp"

# ================================================================================

def resizeImg(img, multiplier):
    img = cv2.resize(img, (int (img.shape[1] * multiplier), int (img.shape[0] * multiplier)))
    return img

def brightPass(img, threshold = 1.55):
    copy = img.copy()

    #TODO -> Checar intervalo aberto
    for y in range(copy.shape[0]):
        for x in range(copy.shape[1]):
            soma = sum(copy[y][x])
            if soma < threshold:
                copy[y][x] = (0.0, 0.0, 0.0)

    return copy


def blurMask(mask, sigma, occ):
    """Blur mask."""
    sm_img = mask.copy()
    sm_img = resizeImg(sm_img, 0.5)

    gaussians = []
    means = []
    st_sigma = sigma
    for i in range(occ):
        print(st_sigma)
        gaussians.append(cv2.GaussianBlur(sm_img.copy(), (0, 0), st_sigma))

        mean_img_aux = sm_img.copy()        
        k_size = st_sigma if st_sigma % 2 != 0 else st_sigma + 1
        for j in range(10):
            mean_img_aux = cv2.blur(mean_img_aux, (k_size, k_size))
        means.append(mean_img_aux)
        st_sigma = int(st_sigma * 2)

    gaussian_mask = gaussians.pop(0)

    for gaussian_img in gaussians:
        gaussian_mask += gaussian_img

    mean_mask = means.pop(0)
    
    for mean_img in means:
        mean_mask += mean_img

    return [gaussian_mask, mean_mask]

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

    # 2 - Blur
    [gaussian_mask, mean_mask] = blurMask(mask, 1, 6)

    cv2.imwrite(f"1 - Mascara - out-{INPUT_IMAGE}", mask * 255)
    cv2.imwrite(f"2 - Mascara Gaussiana - out-{INPUT_IMAGE}", resizeImg(gaussian_mask, 2) * 255)
    cv2.imwrite(f"2 - Mascara Media - out-{INPUT_IMAGE}", resizeImg(mean_mask, 2) * 255)
    cv2.imwrite(f"3 - Bloom Gaussiano - out-{INPUT_IMAGE}", bloomImg(img, resizeImg(gaussian_mask, 2)) * 255)
    cv2.imwrite(f"3 - Bloom Media - out-{INPUT_IMAGE}", bloomImg(img, resizeImg(mean_mask, 2)) * 255)


    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# ===============================================================================
