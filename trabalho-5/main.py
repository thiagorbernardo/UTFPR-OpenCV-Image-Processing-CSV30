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
import statistics as st

sys.setrecursionlimit(3000)


def main():
    for INPUT_IMAGE in range(0,1):
    # for INPUT_IMAGE in range(9,10):
    # for INPUT_IMAGE in range(10,11):
    # for INPUT_IMAGE in range(11,12):
    # for INPUT_IMAGE in range(12,13):
    # for INPUT_IMAGE in range(13,14):
        img = cv2.imread(f"img/{INPUT_IMAGE}.bmp", cv2.IMREAD_COLOR)
        print(img)
        if img is None:
            print("Erro abrindo a imagem.\n")
            sys.exit()

        # img = img.reshape((img.shape[0], img.shape[1], img.shape[2]))

        # Make a copy of the image
        image_copy = img.copy()

        # Display the image copy
        # cv2.imwrite(f"out/{INPUT_IMAGE}-read.bmp", image_copy)
        print(image_copy[0][0])

        # Change color to RGB (from BGR)
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2HLS)
        print(image_copy[0][0][0])
        
        mask = img.copy()
        # mask = mask.reshape((img.shape[0], img.shape[1], 1))
        # mask = np.ones(mask.shape[:2], dtype="uint8")

        test = []
        for y in range(image_copy.shape[0]):
            for x in range(image_copy.shape[1]):
                hue = image_copy[y][x][0] - 60
                # print(hue)
                if(hue < 0):
                    
                    hue = (hue * -2) / 120
                else:
                    hue = hue / 120

                mask[y][x] = hue * 255
                # print(hue)
                
                test.append(hue)

        print(np.average(test))
        # print(np.average(test) * 255)
        print(mask)
        # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        # Vizualize the mask
        cv2.imwrite(f"out/{INPUT_IMAGE}-mask.bmp", mask)

        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# ===============================================================================