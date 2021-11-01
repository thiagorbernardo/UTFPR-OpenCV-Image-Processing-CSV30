# ===============================================================================
# Exemplo: segmentação de uma imagem em escala de cinza.
# -------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu
# Universidade Tecnológica Federal do Paraná
# ===============================================================================
# Trabalho 1
# Autores: Thiago Ramos e João Lucas
# ===============================================================================

import sys
import timeit
import numpy as np
import cv2

sys.setrecursionlimit(3000)
# ===============================================================================

# Arroz
INPUT_IMAGE = "arroz.bmp"

NEGATIVO = False
THRESHOLD = 0.8
ALTURA_MIN = 10
LARGURA_MIN = 10
N_PIXELS_MIN = 15

IMG_HEIGHT=1080
IMG_WIDTH=1504

# Documentos - Letras
# INPUT_IMAGE = "documento-3mp.bmp"

# NEGATIVO = True
# THRESHOLD = 0.4
# ALTURA_MIN = 5
# LARGURA_MIN = 5
# N_PIXELS_MIN = 5

# IMG_HEIGHT=2048
# IMG_WIDTH=1536
# ===============================================================================


def binariza(img, threshold):
    """Binarização simples por limiarização.

    Parâmetros: img: imagem de entrada. Se tiver mais que 1 canal, binariza cada
                  canal independentemente.
                threshold: limiar.

    Valor de retorno: versão binarizada da img_in."""
    return np.where(img >= threshold, 1000.0, 0.0)


# -------------------------------------------------------------------------------

def inunda(label, img, x0, y0, component):
    if(img[y0][x0] != 1000.0): # usar outro valor
        return component

    img[y0][x0] = label

    component["L"] = x0 if x0 <= component["L"] else component["L"]
    component["B"] = y0 if y0 >= component["B"] else component["B"]
    component["R"] = x0 if x0 >= component["R"] else component["R"]

    # T, L, B, R
    neighbourhood = [[y0 - 1, x0], [y0, x0 - 1], [y0 + 1, x0], [y0, x0 + 1]]

    for neigh in neighbourhood:
        [y, x] = neigh

        if (y >= 0 and y < IMG_HEIGHT) and (x >= 0 and x < IMG_WIDTH):
            component = inunda(label, img, x, y, component)

    return component


def rotula(img, largura_min, altura_min, n_pixels_min):
    """Rotulagem usando flood fill. Marca os objetos da imagem com os valores
    [0.1,0.2,etc].

    Parâmetros: img: imagem de entrada E saída.
                largura_min: descarta componentes com largura menor que esta.
                altura_min: descarta componentes com altura menor que esta.
                n_pixels_min: descarta componentes com menos pixels que isso.

    Valor de retorno: uma lista, onde cada item é um vetor associativo (dictionary)
    com os seguintes campos:

    'label': rótulo do componente.
    'n_pixels': número de pixels do componente.
    'T', 'L', 'B', 'R': coordenadas do retângulo envolvente de um componente conexo,
    respectivamente: topo, esquerda, baixo e direita."""

    label = 0.1
    components = []

    for y in range(0, len(img)):
        for x in range(0, len(img[y])):
            if img[y, x] == 1000.0:
                component = {"label": label, "L": x, "B": y, "R": x, "T": y}

                component = inunda(label, img, x, y, component)

                height = component['B'] - component['T']
                width = component['R'] - component['L']

                if(height >= altura_min and width >= largura_min and (height * width) >= n_pixels_min):
                    components.append(component)

                label += 0.1

    return components


# ===============================================================================


def main():

    # Abre a imagem em escala de cinza.
    img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Erro abrindo a imagem.\n")
        sys.exit()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.reshape((img.shape[0], img.shape[1], 1))
    img = img.astype(np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Segmenta a imagem.
    if NEGATIVO:
        img = 1 - img
    img = binariza(img, THRESHOLD)
    # cv2.imshow("01 - binarizada", img)
    cv2.imwrite("01 - binarizada.png", img * 255)

    start_time = timeit.default_timer()
    componentes = rotula(img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)

    n_componentes = len(componentes)

    # print ('Tempo: %f' % (timeit.default_timer () - start_time))
    print ('%d componentes detectados.' % n_componentes)

    # Mostra os objetos encontrados.
    for c in componentes:
        cv2.rectangle(img_out, (c["L"], c["T"]), (c["R"], c["B"]), (0, 0, 1))

    cv2.imshow("02 - out", img_out)
    cv2.imwrite("02 - out.png", img_out * 255)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# ===============================================================================
