import sys
import os
from app.webcam import WebcamApp
import tkinter as tk

def main():
    root = tk.Tk()
    app = WebcamApp(root, "Webcam Monitoring")
    root.mainloop()

if __name__ == "__main__":
    main()
