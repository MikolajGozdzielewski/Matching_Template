import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.signal import convolve2d
import klasy as klasy

#FUNKCJA WYGENEROWANA PRZEZ CHATGPT
def gaussian_kernel(kernel_size, sigma):
    """
    Tworzy jądro Gaussowskie o podanym rozmiarze i sigma.
    """
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
            - ((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2) / (2 * sigma ** 2)
        ),
        (kernel_size, kernel_size)
    )
    return kernel / np.sum(kernel)

#FUNKCJA WYGENEROWANA PRZEZ CHATGPT
def apply_convolution(image, kernel):
    """
    Zastosowanie konwolucji do obrazu z podanym jądrem.
    """
    if len(image.shape) == 3:  # Jeśli obraz jest kolorowy (3 kanały)
        def przetworz_kanal(c):
            return convolve2d(image[:, :, c], kernel, mode='same', boundary='symm')
        output = np.zeros_like(image)
        output[:, :, :] = np.array(list(map(przetworz_kanal, range(image.shape[2])))).transpose(1, 2, 0)
    else:  # Jeśli obraz jest w odcieniach szarości
        output = convolve2d(image, kernel, mode='same', boundary='symm')
    return output

#FUNKCJA WYGENEROWANA PRZEZ CHATGPT
def gaussian_blur(image, kernel_size=5, sigma=1):
    """
    Zastosowanie rozmycia Gaussowskiego na obrazie.
    """
    kernel = gaussian_kernel(kernel_size, sigma)
    return apply_convolution(image, kernel)

#FUNKCJA STWORZONA Z POMOCA CHATGPT, FUNKCJE WEWNETRZNE WYGENEROWANE PRZEZ CHATGPT
def process_template(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image.shape[2] == 4:
        b, g, r, alpha = cv2.split(image)
    else:
        print("Obraz nie ma kanału alfa!")
        return []

    mask = alpha > 0
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    image_rgb = cv2.bitwise_and(image_rgb, image_rgb, mask=mask.astype(np.uint8) * 255)

    blurred_image = gaussian_blur(image_rgb, 3, 2.5)

    labels = np.zeros(image_rgb.shape[:2], dtype=int)
    area_count = 0

    def color_distance(c1, c2):
        return np.sqrt(np.sum((c1 - c2) ** 2))

    visited = np.zeros(image_rgb.shape[:2], dtype=bool)
    areas = defaultdict(list)

    def flood_fill(x, y, area_id):
        stack = [(x, y)]
        areas[area_id].append((x, y))
        visited[y, x] = True
        labels[y, x] = area_id

        while stack:
            cx, cy = stack.pop()

            def process_pixel(nx, ny, cx, cy, image_rgb, blurred_image, mask, visited, labels, stack, areas, area_id):
                if 0 <= nx < image_rgb.shape[1] and 0 <= ny < image_rgb.shape[0]:
                    if not visited[ny, nx] and mask[ny, nx]:
                        current_color = blurred_image[ny, nx]
                        area_pixels = np.array(areas[area_id])
                        pixel_values = blurred_image[area_pixels[:, 1], area_pixels[:, 0]]
                        avg_color = np.mean(pixel_values, axis=0)
                        if color_distance(current_color, avg_color) < 30:
                            stack.append((nx, ny))
                            areas[area_id].append((nx, ny))
                            visited[ny, nx] = True
                            labels[ny, nx] = area_id

            neighbors = [(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)]

            list(map(lambda pos: process_pixel(pos[0], pos[1], cx, cy, image_rgb, blurred_image, mask, visited, labels,
                                               stack, areas, area_id), neighbors))

    points = np.column_stack(np.where(mask == True))
    for y,x in points:
        if not visited[y, x]:
            area_count += 1
            flood_fill(x, y, area_count)

    area_objects = {}

    def create_area(area_id_pixels_tuple):
        area_id, pixels = area_id_pixels_tuple
        return (area_id, klasy.Area(area_id, pixels, blurred_image))

    area_objects = dict(map(create_area, areas.items()))

    sorted_area_objects = sorted(area_objects.values(), key=lambda x: x.size, reverse=True)

    top_two_areas = sorted_area_objects[:2]

    return top_two_areas


def process_image_based_on_gray(blurred_image, odcien1, odcien2):

    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

    new_image_array = np.zeros_like(gray_image, dtype=np.uint8)

    new_image_array[255 >= gray_image] = 150
    new_image_array[odcien1 + 20 >= gray_image] = 255
    new_image_array[odcien1 - 20 >= gray_image] = 150
    new_image_array[odcien2 + 20 >= gray_image] = 0
    new_image_array[odcien2 - 20 >= gray_image] = 150

    return new_image_array


def color_to_gray_based_on_similarity(blurred_img,c1,c2,maska):

    color1 = np.array([c1[2], c1[1], c1[0]])
    color2 = np.array([c2[2], c2[1], c2[0]])

    gray_img = np.ones((blurred_img.shape[0], blurred_img.shape[1]), dtype=np.uint8) * 150

    points1 = np.column_stack(np.where(maska == 255))
    for i,j in points1:
        diff1 = np.linalg.norm(blurred_img[i, j] - color1)
        if diff1 < 40:
            gray_img[i, j] = 255

    points2 = np.column_stack(np.where(maska == 0))
    for i, j in points2:
        diff2 = np.linalg.norm(blurred_img[i, j] - color2)
        if diff2 < 40:
            gray_img[i, j] = 0
    return gray_img

#FUNKCJA STWORZONA Z POMOCA CHATGPT, FUNKCJE WEWNETRZNE W PELNI WYGENEROWANE PRZEZ CHATGPT
def find_regions_and_display(image, shade, min_area_size, max_area_size):
    rows, cols = image.shape

    visited = np.zeros_like(image, dtype=bool)

    def in_bounds(r, c):
        return 0 <= r < rows and 0 <= c < cols

    def bfs(start_r, start_c, region_id):
        queue = [(start_r, start_c)]
        region = []
        visited[start_r, start_c] = True

        while queue:
            r, c = queue.pop(0)
            region.append((r, c))

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if in_bounds(nr, nc) and not visited[nr, nc]:
                    if image[nr, nc] == shade:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

        return region

    regions = []
    colored_image = np.copy(image)

    region_id = 1
    points = np.column_stack(np.where(image == shade))
    for r,c in points:
        if not visited[r, c]:

            region = bfs(r, c, region_id)

            if len(region) < min_area_size or len(region) > max_area_size:
                continue

            regions.append(klasy.Region(region))

            region_id += 1

    return regions

def obrot_obrazu(obraz,kat,y_obrotu,x_obrotu):
    rad = np.radians(kat)
    wysokosc,szerokosc = obraz.shape[:2]
    pusty_obraz = np.zeros((wysokosc*4,szerokosc*4,4), dtype=np.uint8)
    alfa = obraz[:,:,3]
    points = np.column_stack(np.where(alfa == 255))
    for y,x in points:
        pusty_obraz[round((np.sin(rad) * (x - x_obrotu) + np.cos(rad) * (y - y_obrotu)) + y_obrotu * 2)][round((np.cos(rad) * (x - x_obrotu) - np.sin(rad) * (y - y_obrotu)) + x_obrotu * 2)] = obraz[y][x]
    kolumny = np.any(pusty_obraz[:,:,3] != 0, axis=0)
    pusty_obraz = pusty_obraz[:,kolumny]
    wiersze = np.any(pusty_obraz[:, :, 3] != 0, axis=1)
    pusty_obraz = pusty_obraz[wiersze]
    nowy_x = x_obrotu * 2 - np.argmax(kolumny)
    nowy_y = y_obrotu * 2 - np.argmax(wiersze)
    return pusty_obraz, (nowy_y,nowy_x)


def obrot_regionu(obraz, kat):
    rad = np.radians(kat)
    wysokosc, szerokosc = max(obraz[0]),max(obraz[1])
    y_obrotu,x_obrotu = wysokosc/2,szerokosc/2
    pusty_obraz = np.zeros((len(obraz),2))
    obraz = np.array(obraz)
    dy = obraz[:, 0] - y_obrotu
    dx = obraz[:, 1] - x_obrotu
    new_y = np.round(np.sin(rad) * dx + np.cos(rad) * dy + y_obrotu * 2).astype(int)
    new_x = np.round(np.cos(rad) * dx - np.sin(rad) * dy + x_obrotu * 2).astype(int)
    pusty_obraz[:, 0] = new_y
    pusty_obraz[:, 1] = new_x
    return pusty_obraz


#FUNKCJA STWORZONA Z POMOCA CHATGPT
def custom_template_matching(image, template, start_point_image, start_point_template):
    img_height, img_width = image.shape[:2]
    tmpl_height, tmpl_width = template.shape[:2]

    new_start_x = start_point_image[1] - start_point_template[1]
    new_start_y = start_point_image[0] - start_point_template[0]
    if new_start_x < 0 or new_start_y < 0:
        return 100000


    if new_start_x + tmpl_width > img_width or new_start_y + tmpl_height > img_height:
        return 200000

    region = image[new_start_y:new_start_y + tmpl_height, new_start_x:new_start_x + tmpl_width]

    alpha_mask = template[:, :, 3] / 255.0

    masked_region = region * alpha_mask[:, :, np.newaxis]
    masked_template = template[:, :, :3] * alpha_mask[:, :, np.newaxis]

    non_alpha_pixels = np.sum(alpha_mask > 0)

    diff = masked_region - masked_template
    squared_diff = diff ** 2

    similarity = np.sum(squared_diff)

    return similarity / non_alpha_pixels

def szukanie_katu_obszru(region):
    for j in range(15, 181, 15):
        nowy = obrot_regionu(region.pixels, j)
        y, x = zip(*nowy)
        if ((max(x) - min(x)) * (max(y) - min(y))) <= (region.x_range * region.y_range) and (max(x) - min(x)) > (max(y) - min(y)):
            region.x_range = max(x) - min(x)
            region.y_range = max(y) - min(y)
            region.angle = j
    return region

def szukanie_polozenia_template(region,pomoc,kat,template,blurred_image,keviny):
    skala = region.x_range / pomoc.x_distance
    wysokosc, szerokosc = template.shape[:2]
    template_rozmiar = cv2.resize(template, (round(szerokosc * skala), round(wysokosc * skala)))
    y1_nowe = round(pomoc.centroid[0] * skala)
    x1_nowe = round(pomoc.centroid[1] * skala)
    koncowy_x_template = pomoc.centroid[1] * skala
    koncowy_y_template = pomoc.centroid[0] * skala
    wynik = 100000
    y_obraz = round(region.centroid[0])
    x_obraz = round(region.centroid[1])
    for j in range(kat-15,kat+16,15):
        template_kat, nowy_punkt_kat = obrot_obrazu(template_rozmiar, j, y1_nowe, x1_nowe)
        wynik_kat = custom_template_matching(blurred_image, template_kat, (y_obraz, x_obraz), nowy_punkt_kat)
        if wynik_kat < wynik:
            wynik = wynik_kat
            kat = j
            koncowy_x_template = nowy_punkt_kat[1]
            koncowy_y_template = nowy_punkt_kat[0]
    if wynik > 15000:
        return keviny
    wynik_buf = wynik
    template_buf = 0
    nowy_punkt_buf = 0
    if kat != 0 and kat != 360:
        template_buf, nowy_punkt_buf = obrot_obrazu(template, kat, pomoc.centroid[0], pomoc.centroid[1])
    else:
        template_buf = template
        nowy_punkt_buf = (pomoc.centroid[0], pomoc.centroid[1])


    for j in range(int(round(region.x_range)), int(round(region.x_range) + pomoc.x_distance + 1), 1):
        skalabuf = j / pomoc.x_distance
        nowy_x = round(nowy_punkt_buf[1] * skalabuf)
        nowy_y = round(nowy_punkt_buf[0] * skalabuf)
        template_nowy = cv2.resize(template_buf, (round(szerokosc * skalabuf), round(wysokosc * skalabuf)))
        nowy_wynik = custom_template_matching(blurred_image, template_nowy, (y_obraz,x_obraz), (nowy_y, nowy_x))
        if nowy_wynik > wynik:
            break
        else:
            skala = skalabuf
            wynik = nowy_wynik
            koncowy_x_template = nowy_x
            koncowy_y_template = nowy_y

    template_nowy = cv2.resize(template_buf, (round(szerokosc * skalabuf), round(wysokosc * skalabuf)))
    nowy_x = round(nowy_punkt_buf[1] * skala)
    nowy_y = round(nowy_punkt_buf[0] * skala)
    nowy_x_obraz = x_obraz
    nowy_y_obraz = y_obraz

    for j in range(x_obraz - 10, x_obraz + 11, 1):
        nowy_wynik = custom_template_matching(blurred_image, template_nowy, (y_obraz, j), (nowy_y, nowy_x))
        if nowy_wynik > wynik:
            continue
        else:
            nowy_x_obraz = j
            wynik = nowy_wynik

    for j in range(y_obraz - 10, y_obraz + 11, 1):
        nowy_wynik = custom_template_matching(blurred_image, template_nowy, (j, nowy_x_obraz), (nowy_y, nowy_x))
        if nowy_wynik > wynik:
            continue
        else:
            nowy_y_obraz = j
            wynik = nowy_wynik

    if wynik < 5000:
        keviny.append(klasy.Kevin(wynik,(nowy_x_obraz-koncowy_x_template,nowy_y_obraz-koncowy_y_template),(nowy_x_obraz-koncowy_x_template+round(szerokosc*skala),nowy_y_obraz-koncowy_y_template+round(wysokosc*skala))))

    return keviny

def kolejny_kevin(keviny):
    x1 = keviny[0].lewy_gorny[1]
    y1 = keviny[0].lewy_gorny[0]
    x2 = keviny[0].prawy_dolny[1]
    y2 = keviny[0].prawy_dolny[0]

    keviny.pop(0)

    for i in keviny:
        kolejne_x1 = i.lewy_gorny[1]
        kolejne_y1 = i.lewy_gorny[0]
        kolejne_x2 = i.prawy_dolny[1]
        kolejne_y2 = i.prawy_dolny[0]

        if abs(kolejne_x1-x1) < 10 or abs(kolejne_y1-y1) < 10 or abs(kolejne_x2-x2) < 10 or abs(kolejne_y2-y2) < 10:
            continue
        else:
            return i