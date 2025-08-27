import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

# Global variables to store images
original_image = None
preprocessed_image = None
segmented_image = None


def preprocess_image(image):
    """Simple preprocessing: Convert to grayscale and apply GaussianBlur."""
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5)  # CLAHE Filtering
    final_img = clahe.apply(image_bw) + 30
    bilateral = cv2.bilateralFilter(final_img, 15, 75, 75)
    return bilateral


def segment_image(image):
    """Simple segmentation: Apply color filtering and thresholding."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 50, 50])
    upper_bound = np.array([179, 255, 255])
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    colorful_part = cv2.bitwise_and(image, image, mask=mask)

    gray_image = cv2.cvtColor(colorful_part, cv2.COLOR_BGR2GRAY)
    binarized = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    segmented = cv2.erode(binarized, np.ones((1, 1), np.uint8), iterations=1)
    return segmented


def classify_image(segmented_image):
    """Classify image as 'Normal' or 'Abnormal' based on the segmented area size."""
    contours, _ = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = sum(cv2.contourArea(contour) for contour in contours)
    threshold_area = 500

    if total_area > threshold_area:
        return "Abnormal"
    else:
        return "Normal"


def load_image():
    """Load an image from the user's file system."""
    global original_image
    # Get current working directory and set path to "Image_Dataset"
    current_directory = os.getcwd()
    directory = os.path.join(current_directory, 'Image_Dataset')

    # Open the file dialog starting from the current directory
    file_path = filedialog.askopenfilename(
        initialdir=directory,
        title="Select an Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        return

    # Read the image
    original_image = cv2.imread(file_path)
    if original_image is None:
        messagebox.showerror("Error", "Could not load the selected image.")
        return

    # Convert to RGB and display
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    display_image(image_rgb, row=1, col=1, title="Original Image")


def preprocess():
    """Apply preprocessing to the loaded image."""
    global preprocessed_image
    if original_image is None:
        messagebox.showerror("Error", "Please load an image first.")
        return

    preprocessed_image = preprocess_image(original_image)
    # Convert to RGB for display
    image_rgb = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)
    display_image(image_rgb, row=2, col=1, title="Preprocessed Image")


def segment():
    """Apply segmentation to the preprocessed image."""
    global segmented_image
    if preprocessed_image is None:
        messagebox.showerror("Error", "Please preprocess the image first.")
        return

    segmented_image = segment_image(original_image)
    # Convert to RGB for display
    image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
    display_image(image_rgb, row=3, col=1, title="Segmented Image")


def display_image(image, row, col, title=""):
    """Display image on the tkinter window in a specified row and column."""
    # Convert the image to a format that Tkinter can use
    image_pil = Image.fromarray(image)
    image_tk = ImageTk.PhotoImage(image_pil)

    # # Create a new label to display the image in the specified grid position
    image_label = tk.Label(root, image=image_tk)
    image_label.image = image_tk  # Keep a reference to avoid garbage collection
    image_label.grid(row=row, column=col, padx=10, pady=10)


def display_label():
    """Display the label (Normal/Abnormal) for the segmented image."""
    if segmented_image is None:
        messagebox.showerror("Error", "Please segment the image first.")
        return

    # Classify the segmented image
    classification = classify_image(segmented_image)

    # Display the classification message in the second column of the 4th row
    label = tk.Label(root, text=classification, font=("Arial", 16), fg="red")
    label.grid(row=4, column=1, padx=10, pady=10)


# Main GUI window
def GUI():
    global root
    root = tk.Tk()
    root.title("GUI")

    # Window size and center alignment
    window_width, window_height = 600, 500  # Adjust window height for better image display
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_coordinate = int((screen_width / 2) - (window_width / 2))
    y_coordinate = int((screen_height / 2) - (window_height / 2))
    root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

    # Configure rows and columns for grid
    root.grid_rowconfigure(0, weight=0)
    root.grid_rowconfigure(1, weight=1, minsize=200)  # Image rows
    root.grid_rowconfigure(2, weight=1, minsize=200)
    root.grid_rowconfigure(3, weight=1, minsize=200)
    root.grid_rowconfigure(4, weight=1, minsize=50)  # Label row

    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=3)

    # Instructions
    label = tk.Label(root, text="Click the buttons to process the image", font=("Arial", 14))
    label.grid(row=0, column=0, padx=10, pady=10, columnspan=2)

    # Buttons for each step
    load_button = ttk.Button(root, text="Load Image", command=load_image)
    load_button.grid(row=1, column=0, padx=10, pady=10)

    preprocess_button = ttk.Button(root, text="Preprocess Image", command=preprocess)
    preprocess_button.grid(row=2, column=0, padx=10, pady=10)

    segment_button = ttk.Button(root, text="Segment Image", command=segment)
    segment_button.grid(row=3, column=0, padx=10, pady=10)

    display_button = ttk.Button(root, text="Show Label", command=display_label)
    display_button.grid(row=4, column=0, padx=10, pady=10)

    # Start the GUI event loop
    root.mainloop()


if __name__ == "__main__":
    GUI()
