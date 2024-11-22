import tkinter as tk
from tkinter import filedialog
from src.crystal_index.crystal import Crystal
import cv2


class ImageMarker:
    def __init__(self):
        self.points = []
        self.image_path = ""
        self.image = None

    def select_image(self):
        root = tk.Tk()
        root.withdraw()
        self.image_path = filedialog.askopenfilename()
        if not self.image_path:
            raise ValueError("No image selected")

    def mark_points(self):
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError("Image not found or unable to read")

        cv2.imshow("Image", self.image)
        cv2.setMouseCallback("Image", self.click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Image", self.image)

    def run(self):
        self.select_image()
        self.mark_points()
        return self.image_path, self.points


if __name__ == "__main__":
    marker = ImageMarker()
    image_path, points = marker.run()
    crystal = Crystal("FCC", 8)
    crystal.calculate_from_image(image_path, points)
