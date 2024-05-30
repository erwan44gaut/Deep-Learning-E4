import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageOps, ImageStat
import sys
import io
import threading

# Constants
CANVAS_SIZE = 150
PEN_THICKNESS = 10
BACKGROUND_COLOR = '#DDDDDD'
CANVAS_BG_COLOR = 'white'
PEN_COLOR = 'black'
FONT = ('Arial', 100)
TITLE = "Handwritten digits classifier"

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
        self.schedule_evaluation()
        self.previous_predictions = []
        self.previous_prediction_text = ""

    def configure_window(self):
        self.root.title(TITLE)
        self.root.configure(background=BACKGROUND_COLOR)

    def create_widgets(self):
        self.top_frame = tk.Frame(self.root, bg=BACKGROUND_COLOR)
        self.top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.combo = ttk.Combobox(self.top_frame, values=["Multilayer", "Convolutional multilayer"], width=30)
        self.combo.pack(pady=10)
        self.combo.set("Multilayer")

        self.frame_left = tk.Frame(self.top_frame, width=CANVAS_SIZE, height=CANVAS_SIZE, bg=BACKGROUND_COLOR)
        self.frame_right = tk.Frame(self.top_frame, width=CANVAS_SIZE, height=CANVAS_SIZE, bg=BACKGROUND_COLOR)
        self.frame_button = tk.Frame(self.top_frame, bg=BACKGROUND_COLOR)

        self.frame_left.pack(side=tk.LEFT, padx=10, pady=10)
        self.frame_right.pack(side=tk.RIGHT, padx=10, pady=10)
        self.frame_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.canvas_draw = tk.Canvas(self.frame_left, bg=CANVAS_BG_COLOR, width=CANVAS_SIZE, height=CANVAS_SIZE)
        self.canvas_draw.pack()

        self.canvas_display = tk.Canvas(self.frame_right, bg=CANVAS_BG_COLOR, width=CANVAS_SIZE, height=CANVAS_SIZE)
        self.canvas_display.pack()

        self.clear_button = ttk.Button(self.frame_button, text="Clear Drawing", command=self.clear_drawing)
        self.clear_button.pack(pady=10)

        # Add a new frame for chart at the bottom
        self.frame_chart = tk.Frame(self.root, width=CANVAS_SIZE, height=200, bg="white")
        self.frame_chart.pack(side=tk.BOTTOM, fill=tk.BOTH, padx=10, pady=10)
        self.frame_chart.pack_propagate(False)  # Prevent frame_chart from resizing the window

        # Add scrollbars and text widget
        self.scrollbar_y = tk.Scrollbar(self.frame_chart, orient=tk.VERTICAL)
        self.scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

        self.scrollbar_x = tk.Scrollbar(self.frame_chart, orient=tk.HORIZONTAL)
        self.scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

        self.text_widget = tk.Text(self.frame_chart, wrap=tk.NONE, yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set, height=16)
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar_y.config(command=self.text_widget.yview)
        self.scrollbar_x.config(command=self.text_widget.xview)

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

    def is_image_all_white(self, image):
        stat = ImageStat.Stat(image)
        return all(x >= 255 for x in stat.mean)

    def evaluate_drawing(self):
        if self.is_image_all_white(self.image1):
            self.display_text("")
            self.text_widget.delete('1.0', tk.END)
            self.text_widget.insert(tk.END, "Draw a digit to get started.")
            return

        res = "Select a model."
        predictions = []
        # Invert image (suits MNIST better)
        if self.combo.get() == "Multilayer":
            res, predictions = multicouche(ImageOps.invert(self.image1.convert('RGB')))
        elif self.combo.get() == "Convolutional multilayer":
            res, predictions = convolution(ImageOps.invert(self.image1.convert('RGB')))
        else:
            res, predictions = multicouche(ImageOps.invert(self.image1.convert('RGB')))

        self.display_text(res)
        self.plot_predictions(predictions)

    def plot_predictions(self, predictions):
        if self.previous_predictions is None or not np.array_equal(predictions, self.previous_predictions):
            self.previous_predictions = predictions
            self.text_widget.delete('1.0', tk.END)
            for i, pred in enumerate(predictions):
                percentage = int(pred * 100)
                self.text_widget.insert(tk.END, f"{i}: {'#' * (percentage // 3)}[{percentage}%]\n")

    def schedule_evaluation(self):
        thread = threading.Thread(target=self.evaluate_drawing)
        thread.start()
        self.root.after(200, self.schedule_evaluation) 

    def clear_drawing(self):
        self.canvas_draw.delete("all")
        self.create_canvas_image()
        self.clear_text()

    def display_text(self, text):
        if text != self.previous_prediction_text:
            self.canvas_display.delete("all")
            self.canvas_display.create_text(CANVAS_SIZE / 2, CANVAS_SIZE / 2, text=text, fill=PEN_COLOR, font=FONT)
            self.previous_prediction_text = text

    def clear_text(self):
        self.canvas_display.delete("all")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
