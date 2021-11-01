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


def ingenuo(img, w, h):
    """Filtro da media ingenuo."""
    copy = img.copy()

    half_h = int(np.floor(h / 2))
    half_w = int(np.floor(w / 2))

    for y in range(half_h, (img.shape[0] - half_h)):
        for x in range(half_w, (img.shape[1] - half_w)):
            for c in range(0, img.shape[2]):
                sum = 0

                for i in range(y - half_h, y + half_h):
                    for j in range(x - half_w, x + half_w):
                        sum += img[i][j][c]
                copy[y][x][c] = sum / (h * w)

    return copy


# ================================================================================


def separavel(img, w, h):
    """Filtro da media separavel."""

    copy = img.copy()

    half_h = int(np.floor(h / 2))
    half_w = int(np.floor(w / 2))

    if w > h:
        for y in range(half_h, (img.shape[0] - half_h)):
            for x in range(half_w, (img.shape[1] - half_w)):
                for c in range(0, img.shape[2]):
                    soma = 0
                    for j in range(x - half_w, x + half_w):
                        soma += img[y][j][c]
                    copy[y][x][c] = soma / w
    else:    
        for y in range(half_h, (img.shape[0] - half_h)):
            for x in range(half_w, (img.shape[1] - half_w)):
                for c in range(0, img.shape[2]):
                    soma = 0
                    for j in range(y - half_h, y + half_h):
                        soma += img[j][x][c]
                    copy[y][x][c] = soma / h

    return copy


# ===============================================================================


def integral(img, w, h):
    """Filtro da media por imagens integrais."""
    integral = img.copy()

    for y in range(0, (img.shape[0])):
        integral[y][0] = img[y][0]
        for x in range(1, (img.shape[1])):
            for c in range(0, img.shape[2]):
                integral[y][x][c] = img[y][x][c] + integral[y][x - 1][c]

    for y in range(1, (integral.shape[0])):
        for x in range(0, (integral.shape[1])):
            for c in range(0, integral.shape[2]):
                integral[y][x][c] = integral[y][x][c] + integral[y - 1][x][c]

    integral_filtered = np.zeros(integral.shape, np.float32)

    half_h = int(np.floor(h / 2))
    half_w = int(np.floor(w / 2))

    image_height = integral_filtered.shape[0] - 1
    image_window = integral_filtered.shape[1] - 1

    for y in range(0, integral_filtered.shape[0]):
        for x in range(0, integral_filtered.shape[1]):
            for c in range(0, integral_filtered.shape[2]):
                coords = {
                    "RB": {"y": y + half_h, "x": x + half_w},
                    "RT": {"y": y - half_h, "x": x + half_w},
                    "LB": {"y": y + half_h, "x": x - half_w},
                    "LT": {"y": y - half_h, "x": x - half_w},
                }

                if coords["RB"]["y"] > image_height or coords["LB"]["y"] > image_height:
                    coords["LB"]["y"] = coords["RB"]["y"] = image_height
                if coords["RT"]["y"] < 0 or coords["LT"]["y"] < 0:
                    coords["LT"]["y"] = coords["RT"]["y"] = 0

                if coords["RB"]["x"] > image_window or coords["RT"]["x"] > image_window:
                    coords["RB"]["x"] = coords["RT"]["x"] = image_window
                if coords["LB"]["x"] < 0 or coords["LT"]["x"] < 0:
                    coords["LB"]["x"] = coords["LT"]["x"] = 0

                window_h = coords["RB"]["y"] - coords["RT"]["y"]
                window_w = coords["RB"]["x"] - coords["LB"]["x"]

                integral_filtered[y][x][c] = (
                    integral[coords["RB"]["y"]][coords["RB"]["x"]][c]
                    - integral[coords["RT"]["y"]][coords["RT"]["x"]][c]
                    - integral[coords["LB"]["y"]][coords["LB"]["x"]][c]
                    + integral[coords["LT"]["y"]][coords["LT"]["x"]][c]
                ) / (window_h * window_w)

    return integral_filtered


def main():
    img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_COLOR)
    if img is None:
        print("Erro abrindo a imagem.\n")
        sys.exit()

    img = img.reshape((img.shape[0], img.shape[1], img.shape[2]))
    img = img.astype(np.float32) / 255

    # Ingenuo
    img_ingenuo = ingenuo(img, KERNEL_SIZE, KERNEL_SIZE)
    # cv2.imshow(f"01 - out-{INPUT_IMAGE}", img_ingenuo)
    cv2.imwrite(f"01 - out-{INPUT_IMAGE}", img_ingenuo * 255)

    # Separavel
    img_separavel = separavel(img, KERNEL_SIZE, 1)
    img_separavel = separavel(img_separavel, 1, KERNEL_SIZE)
    # cv2.imshow(f"02 - out-{INPUT_IMAGE}", img_separavel)
    cv2.imwrite(f"02 - out-{INPUT_IMAGE}", img_separavel * 255)

    # Integral
    img_integral = integral(img, KERNEL_SIZE, KERNEL_SIZE)
    # cv2.imshow(f"03 - out-{INPUT_IMAGE}", img_integral)
    cv2.imwrite(f"03 - out-{INPUT_IMAGE}", img_integral * 255)


    # median_cv2 = cv2.blur(img, (KERNEL_SIZE, KERNEL_SIZE))
    # cv2.imwrite(f"cv2-ingenuo-{INPUT_IMAGE}", (median_cv2 - img_ingenuo) * 255)
    # cv2.imwrite(f"cv2-separavel-{INPUT_IMAGE}", (median_cv2 - img_separavel) * 255)
    # cv2.imwrite(f"cv2-integral-{INPUT_IMAGE}", (median_cv2 - img_integral) * 255)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# ===============================================================================
