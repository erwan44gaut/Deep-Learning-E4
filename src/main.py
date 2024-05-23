import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageOps
import sys
import io

# Constants
CANVAS_SIZE = 150
PEN_THICKNESS = 10
BACKGROUND_COLOR = '#DDDDDD'
CANVAS_BG_COLOR = 'white'
PEN_COLOR = 'black'
FONT = ('Arial', 100)
TITLE = "Handwritten digits classifier"
# Assume these are available from your ML modules
from multicouche import *
from convolution import *

class App:
    def __init__(self, root):
        self.root = root
        self.configure_window()
        self.create_widgets()
        self.setup_bindings()
        self.create_canvas_image()
        self.configure_stdout()

    def configure_window(self):
        self.root.title(TITLE)
        self.root.configure(background=BACKGROUND_COLOR)

    def create_widgets(self):
        self.combo = ttk.Combobox(self.root, values=["multicouche", "multicouche convolutionnelle"], width=30)
        self.combo.pack(pady=10)
        self.combo.set("multicouche")

        self.frame_left = tk.Frame(self.root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg=BACKGROUND_COLOR)
        self.frame_right = tk.Frame(self.root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg=BACKGROUND_COLOR)
        self.frame_button = tk.Frame(self.root, bg=BACKGROUND_COLOR)

        self.frame_left.pack(side=tk.LEFT, padx=10, pady=10)
        self.frame_right.pack(side=tk.RIGHT, padx=10, pady=10)
        self.frame_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.canvas_draw = tk.Canvas(self.frame_left, bg=CANVAS_BG_COLOR, width=CANVAS_SIZE, height=CANVAS_SIZE)
        self.canvas_draw.pack()

        self.canvas_display = tk.Canvas(self.frame_right, bg=CANVAS_BG_COLOR, width=CANVAS_SIZE, height=CANVAS_SIZE)
        self.canvas_display.pack()

        self.save_button = ttk.Button(self.frame_button, text="Predict", command=self.evaluate_drawing)
        self.save_button.pack(pady=10)

        self.clear_button = ttk.Button(self.frame_button, text="Clear Drawing", command=self.clear_drawing)
        self.clear_button.pack(pady=10)

    def setup_bindings(self):
        self.canvas_draw.bind("<B1-Motion>", self.paint)

    def create_canvas_image(self):
        self.image1 = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), CANVAS_BG_COLOR)
        self.draw = ImageDraw.Draw(self.image1)

    def configure_stdout(self):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    def paint(self, event):
        r = PEN_THICKNESS // 2
        bounding_box = [event.x - r, event.y - r, event.x + r, event.y + r]
        self.canvas_draw.create_oval(bounding_box, fill=PEN_COLOR, outline=PEN_COLOR)
        self.draw.ellipse(bounding_box, fill=PEN_COLOR, outline=PEN_COLOR)

    def evaluate_drawing(self):
        res = "Select a model."
        
        # Invert image (suits MNIST better)
        self.image1 = ImageOps.invert(self.image1.convert('RGB'))
        if self.combo.get() == "multicouche":
            res = multicouche(self.image1)
        elif self.combo.get() == "multicouche convolutionnelle":
            res = convolution(self.image1)
        else:
            res = multicouche(self.image1)

        self.display_text(res)

    def clear_drawing(self):
        self.canvas_draw.delete("all")
        self.create_canvas_image()
        self.clear_text()

    def display_text(self, text):
        self.canvas_display.delete("all")
        self.canvas_display.create_text(CANVAS_SIZE / 2, CANVAS_SIZE / 2, text=text, fill=PEN_COLOR, font=FONT)

    def clear_text(self):
        self.canvas_display.delete("all")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
