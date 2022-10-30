import cv2
import pyautogui
import numpy as np


class OpencvWrapper:
    @staticmethod
    def screenshot(coordinate=None):
        """
        Args:
            coordinate: (x1, y1, x2, y2) with (x1, y1) in 2nd Quadrant and (x2, y2) in 4th Quadrant.
        Returns:
            img (Image):
        """
        if coordinate:
            x, y = coordinate[:2]
            width = coordinate[2]-x
            height = coordinate[3]-y
            print(x, y, width, height)
            img = pyautogui.screenshot(region=(x, y, width, height))
        else:
            img = pyautogui.screenshot()
        return img

    @staticmethod
    def image_gray(rgb_img):
        return cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def search_image(background, template_path, precision=0.95) -> tuple:
        """
        Search a given (template) from background by using cv2.matchTemplate

        Args:
            background
            template_path
            precision

        Returns:
            coordinate
            res
        """
        rgb_img = np.array(background)
        gray_img = OpencvWrapper.image_gray(rgb_img)
        template_rgb = cv2.imread(template_path)
        if template_rgb is None:
            raise TypeError(f'Image is not found in path: {template_path}')
        template_gray = OpencvWrapper.image_gray(template_rgb)

        size = np.asarray(template_gray.shape[::-1])
        res = cv2.matchTemplate(gray_img, template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        top_left = np.add(np.asarray(max_loc), -0.5*size)
        bottom_right = np.add(np.asarray(max_loc), 0.5*size)
        coordinate = [top_left, bottom_right]
        return coordinate, res

    @staticmethod
    def click_image():
        return
