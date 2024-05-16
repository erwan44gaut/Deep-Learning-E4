import os
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageDraw

from multicouche import *
from convolution import *


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Paint Application")

        # Combobox at the top center
        self.combo = ttk.Combobox(root, values=["multicouche", "multicouche convolutionnelle"])
        self.combo.pack(pady=10)

        # Frames to organize the layout
        self.frame_left = tk.Frame(root, width=300, height=300, bg="white")
        self.frame_right = tk.Frame(root, width=300, height=300, bg="white")
        self.frame_button = tk.Frame(root)

        self.frame_left.pack(side=tk.LEFT, padx=10, pady=10)
        self.frame_right.pack(side=tk.RIGHT, padx=10, pady=10)
        self.frame_button.pack(side=tk.LEFT, padx=10, pady=10)

        # Canvas to draw on (left)
        self.canvas_draw = tk.Canvas(self.frame_left, bg="white", width=300, height=300)
        self.canvas_draw.pack()
        self.canvas_draw.bind("<B1-Motion>", self.paint)
        self.canvas_draw.bind("<ButtonPress-1>", self.start_draw)

        # Canvas to display text (right)
        self.canvas_display = tk.Canvas(self.frame_right, bg="white", width=300, height=300)
        self.canvas_display.pack()

        # Button to save the drawing
        self.save_button = ttk.Button(self.frame_button, text="IA Drawing", command=self.save_drawing)
        self.save_button.pack(pady=10)

        # Button to clear the drawing
        self.clear_button = ttk.Button(self.frame_button, text="Clear Drawing", command=self.clear_drawing)
        self.clear_button.pack(pady=10)

        # PIL image to draw on
        self.image1 = Image.new("RGB", (300, 300), "white")
        self.draw = ImageDraw.Draw(self.image1)

        # Variables to track the last position
        self.last_x, self.last_y = None, None

    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y

    def paint(self, event):
        if self.last_x and self.last_y:
            x1, y1 = self.last_x, self.last_y
            x2, y2 = event.x, event.y
            self.canvas_draw.create_line(x1, y1, x2, y2, fill="black", width=5)
            self.draw.line([x1, y1, x2, y2], fill="black", width=5)
        self.last_x, self.last_y = event.x, event.y

    def save_drawing(self):
        res = "Select a model."
        file_path = "saved_drawing.png"
        self.image1.save(file_path)
        if self.combo.get() == "multicouche":
            res = multicouche()
        elif self.combo.get() == "multicouche convolutionnelle":
            res = convolution()
        self.display_text(res)

    def clear_drawing(self):
        self.canvas_draw.delete("all")
        self.image1 = Image.new("RGB", (300, 300), "white")
        self.draw = ImageDraw.Draw(self.image1)
        self.clear_text()

    def display_text(self, text):
        self.canvas_display.delete("all")
        self.canvas_display.create_text(150, 150, text=text, fill="black", font=("Arial", 14))

    def clear_text(self):
        self.canvas_display.delete("all")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
