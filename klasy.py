#MIKOŁAJ GOŹDZIELEWSKI 193263
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.signal import convolve2d

#FUNKCJE KLASY STWORZONE Z POMOCA CHATGPT
class Area:
    def __init__(self, area_id, pixels, blurred_image):
        self.area_id = area_id
        self.pixels = pixels
        self.pixels_y_x = self.reverse()
        self.size = len(pixels)
        self.avg_color = self.calculate_average_color(blurred_image)
        self.x_distance = self.calculate_x_distance()
        self.y_distance = self.calculate_y_distance()
        self.center_of_mass = self.calculate_center_of_mass()
        self.angle = 0

    def calculate_average_color(self, blurred_image):
        pixels_array = np.array(self.pixels)
        pixel_values = blurred_image[pixels_array[:, 1], pixels_array[:, 0]]
        return np.mean(pixel_values, axis=0)

    def calculate_x_distance(self):
        min_x = min(self.pixels, key=lambda p: p[0])[0]
        max_x = max(self.pixels, key=lambda p: p[0])[0]
        return max_x - min_x

    def calculate_y_distance(self):
        min_y = min(self.pixels, key=lambda p: p[1])[1]
        max_y = max(self.pixels, key=lambda p: p[1])[1]
        return max_y - min_y

    def calculate_center_of_mass(self):
        pixels_array = np.array(self.pixels)
        center_of_mass_x = np.mean(pixels_array[:, 0])
        center_of_mass_y = np.mean(pixels_array[:, 1])
        return (center_of_mass_y, center_of_mass_x)

    def reverse(self):
        return list(map(lambda p: (p[1], p[0]), self.pixels))

    #def __repr__(self):
    #    return (f"Area {self.area_id}: Size={self.size}, Avg Color={self.avg_color}, "
    #            f"X Distance={self.x_distance}, Y Distance={self.y_distance}, "
    #            f"Center of Mass={self.center_of_mass}, Angle={self.angle}")

#FUNKCJE KLASY STWORZONE Z POMOCA CHATGPT
class Region:
    def __init__(self, pixels,angle=0):

        self.pixels = pixels
        self.num_pixels = len(pixels)

        self.centroid = self.calculate_centroid()

        self.x_range = self.calculate_x_range()

        self.y_range = self.calculate_y_range()

        self.angle = angle

    def calculate_centroid(self):
        if self.num_pixels == 0:
            return (0, 0)
        pixels_array = np.array(self.pixels)
        sum_x = np.sum(pixels_array[:, 1])
        sum_y = np.sum(pixels_array[:, 0])
        return (sum_y / self.num_pixels, sum_x / self.num_pixels)

    def calculate_x_range(self):
        pixels_array = np.array(self.pixels)
        x_values = pixels_array[:, 1]
        return max(x_values) - min(x_values)

    def calculate_y_range(self):
        pixels_array = np.array(self.pixels)
        y_values = pixels_array[:, 0]
        return max(y_values) - min(y_values)

    #def __repr__(self):
    #    return f"Region(num_pixels={self.num_pixels}, centroid={self.centroid}, x_range={self.x_range})"

class Pomoc:
    def __init__(self, colors_bgr, centroid,angle,x_distance,area):
        self.r = int(round(colors_bgr[2]))
        self.g = int(round(colors_bgr[1]))
        self.b = int(round(colors_bgr[0]))
        self.gray = int(round(self.calculate_gray()))
        self.centroid = (int(round(centroid[0])),int(round(centroid[1])))
        self.x_distance = x_distance
        self.angle = angle
        self.area = area

    def calculate_gray(self):
        return self.r * 0.2989 + self.g * 0.587 + self.b * 0.114

class Kevin:
    def __init__(self, wynik, lewy_gorny,prawy_dolny):
        self.wynik = wynik
        self.lewy_gorny = lewy_gorny
        self.prawy_dolny = prawy_dolny