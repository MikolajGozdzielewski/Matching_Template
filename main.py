#MIKOŁAJ GOŹDZIELEWSKI 193263
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.signal import convolve2d
import funkcje as funkcje
import klasy as klasy

if __name__ == "__main__":
    template_path = "template.png"
    area_details = funkcje.process_template(template_path)

    pomoc = []

    def katy_area(i):
        for j in range(0, 181, 15):
            nowy = funkcje.obrot_regionu(i.pixels_y_x, j)

            y, x = zip(*nowy)
            if ((max(x) - min(x)) * (max(y) - min(y))) <= (i.x_distance * i.y_distance) and (max(x) - min(x)) > (
                    max(y) - min(y)):
                i.x_distance = max(x) - min(x)
                i.y_distance = max(y) - min(y)
                i.angle = j

        return klasy.Pomoc(i.avg_color, i.center_of_mass, i.angle, i.x_distance, i.size)


    pomoc.extend(map(katy_area, area_details))


    if pomoc[0].gray < pomoc[1].gray:
        buffor = pomoc[0]
        pomoc[0] = pomoc[1]
        pomoc[1] = buffor

    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    template = funkcje.gaussian_blur(template, 3, 2.5)


    input_image_path = ['photos/bodo_2_g13.jpg','photos/budapest_2_g13.jpg','photos/buenos_g13.jpg','photos/ny_2_g13.jpg','photos/rio_2_g13.jpg','photos/ushuaia_g13.jpg']
    images_final = []

    for image_name in input_image_path:
        print("Analiza pliku: ",image_name)
        keviny = []
        image = cv2.imread(image_name)
        img_height, img_width = image.shape[:2]
        if (img_height * img_width > 5000*3000):
            image = cv2.resize(image, (round(img_width / 2), round(img_height / 2)))
        blurred_image = funkcje.gaussian_blur(image, 3, 2.5)
        mask = funkcje.process_image_based_on_gray(blurred_image,pomoc[0].gray,pomoc[1].gray)
        mask = funkcje.color_to_gray_based_on_similarity(blurred_image,(pomoc[0].r,pomoc[0].g,pomoc[0].b),(pomoc[1].r,pomoc[1].g,pomoc[1].b),mask)

        regions1 = funkcje.find_regions_and_display(mask, 255, min_area_size=round(pomoc[0].area*0.05),max_area_size=round(pomoc[0].area*2))
        regions2 = funkcje.find_regions_and_display(mask, 0, min_area_size=round(pomoc[1].area*0.05),max_area_size=round(pomoc[1].area*2))

        regions1 = list(map(funkcje.szukanie_katu_obszru, regions1))
        regions2 = list(map(funkcje.szukanie_katu_obszru, regions2))

        for i in regions1:
            keviny = funkcje.szukanie_polozenia_template(i,pomoc[0],pomoc[0].angle - i.angle,template,blurred_image,keviny)
            keviny = funkcje.szukanie_polozenia_template(i, pomoc[0], pomoc[0].angle - i.angle + 180, template, blurred_image,keviny)

        for i in regions2:
            keviny = funkcje.szukanie_polozenia_template(i,pomoc[1],pomoc[1].angle - i.angle,template,blurred_image,keviny)
            keviny = funkcje.szukanie_polozenia_template(i, pomoc[1], pomoc[1].angle - i.angle + 180, template, blurred_image,keviny)


        keviny.sort(key=lambda kevin: kevin.wynik)
        cv2.rectangle(image,keviny[0].lewy_gorny,keviny[0].prawy_dolny,(0,255,0),2)
        kevin = funkcje.kolejny_kevin(keviny)
        cv2.rectangle(image, kevin.lewy_gorny, kevin.prawy_dolny,(0,255,0),2)
        images_final.append(image)
        print("Koniec analizy pliku: ",image_name)

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    x_matplot = 0
    y_matplot = 0
    k = 1

    def display_image(i, final_image):
        x_matplot, y_matplot = divmod(i, 3)
        axs[x_matplot, y_matplot].imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
        axs[x_matplot, y_matplot].axis('off')
        cv2.imwrite(f"finalny_obraz_{i}.jpg", final_image)


    list(map(display_image, range(len(images_final)), images_final))


    plt.tight_layout()
    plt.show()
