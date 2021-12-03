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
    # for INPUT_IMAGE in range(0,1):
    # for INPUT_IMAGE in range(9,10):
    for INPUT_IMAGE in range(10,11):
    # for INPUT_IMAGE in range(11,12):
    # for INPUT_IMAGE in range(12,13):
    # for INPUT_IMAGE in range(13,14):
    # for INPUT_IMAGE in range(14,15):
        img = cv2.imread(f"img/{INPUT_IMAGE}.bmp", cv2.IMREAD_COLOR)
        if img is None:
            print("Erro abrindo a imagem.\n")
            sys.exit()

        # img = img.reshape((img.shape[0], img.shape[1], img.shape[2]))

        # Make a copy of the image
        image_copy = img.copy()

        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2HLS)

        
        # maskHue = img.copy()
        # maskSat = img.copy()
        # maskLig = img.copy()    
        # mask = mask.reshape((img.shape[0], img.shape[1], 1))
        maskHue = np.ones(img.shape[:2], dtype="uint8") * 255
        maskLig = np.ones(img.shape[:2], dtype="uint8") * 255
        maskSat = np.ones(img.shape[:2], dtype="uint8") * 255

        for y in range(image_copy.shape[0]):
            for x in range(image_copy.shape[1]):
                hue = image_copy[y][x][0] - 60
                Lig = image_copy[y][x][1]
                Sat = image_copy[y][x][2]

                # HUE E SATURATION CONFIÁVEL
                if(Lig > 76 and Lig < 178 and Sat > 76):
                    if(hue < 0):
                        maskHue[y][x] = ((hue * -2) / 120) * 255
                    else:
                        maskHue[y][x] = (hue / 120) * 255

                # else:
                #     maskLig[y][x] = Lig
                
                maskSat[y][x] = Sat

                # if(Lig > 76 and Lig < 178):
                #     if(Sat >76):
                #         if(hue < 0):
                #             hue = (hue * -2) / 120
                #         else:
                #             hue = hue / 120

                #         maskHue[y][x] = hue * 255
                #         # MASK HUE -> BRANCA ONDE N É VERDE
                        
                #         maskSat[y][x] = 255
                #     else:
                #         # maskHue[y][x] = 0
                #         maskSat[y][x] =  Sat


                # else:
                #     # if(Lig >  100):
                        
                #     #     maskLig[y][x] = 255
                #     # elif(Lig < 30):
                #     #     maskLig[y][x] = 0

                #     # else:
                #     maskLig[y][x] = Lig


        # Vizualize the mask
        # maskHue = 255 - maskHue
        maskSat = 255 - maskSat
        # maskLig = 255 - maskLig

        maskFinal = ((maskLig/255)*(maskSat/255)*(maskHue/255))
        maskFinal = cv2.normalize(maskFinal, maskFinal, 0, 1, cv2.NORM_MINMAX)
        # maskFinal = np.clip(maskFinal, a_min = 0, a_max = 255)

        masked_image = np.copy(img)
        # 3 try
        # for y in range(masked_image.shape[0]):
        #     for x in range(masked_image.shape[1]):
        #         for c in range(masked_image.shape[2]):
        #             masked_image[y][x][c] = maskFinal[y][x] * masked_image[y][x][c]

        # print(np.amax(maskFinal))
        masked_image[maskFinal < 0.2] = [0, 0, 0]
        # masked_image[maskFinal < 10 ] = [0, 0, 0]

        cv2.imwrite(f"out/{INPUT_IMAGE}-maskHue.bmp", maskHue)
        cv2.imwrite(f"out/{INPUT_IMAGE}-maskSat.bmp", maskSat)
        cv2.imwrite(f"out/{INPUT_IMAGE}-maskLig.bmp", maskLig)
        cv2.imwrite(f"out/{INPUT_IMAGE}-maskFinal.bmp", maskFinal * 255)
        cv2.imwrite(f"out/{INPUT_IMAGE}-massked.bmp", masked_image)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# ===============================================================================
