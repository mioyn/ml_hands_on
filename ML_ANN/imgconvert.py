import base64
import os
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw
from rembg import remove


def remove_bg(jpg_file_path, png_file_path):
    """
    Identifies and removes the person's background from an image
    and saves it as a transparent PNG.

    Args:
        input_path (str): Path to the input image (e.g., 'person_image.jpg').
        output_path (str): Path to save the output transparent PNG (e.g., 'output.png').
    """
    try:
        # Open the image file
        with Image.open(jpg_file_path) as im:
            # Save the image with a new extension; the format is inferred from the filename

            output_image = remove(im)
            output_image.save(png_file_path, format="PNG")
        print(f"Successfully converted {jpg_file_path} to {png_file_path}")
    except IOError as e:
        print(f"Error converting image: {e}")


def create_svg_from_png(png_file_path, svg_file_path):
    """
    Converts a PNG image to an SVG format by embedding the PNG as a base64-encoded image.

    Args:
        png_file_path (str): Path to the input PNG image.
        svg_file_path (str): Path to save the output SVG file.
    """
    try:
        # Load image
        with Image.open(png_file_path) as img:
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

            svg_content = f"""<svg style="height:1em" viewBox="0 0 370 391">
  <image href="data:image/png;base64,{encoded}" />
</svg>"""

            with open(svg_file_path, "w") as f:
                f.write(svg_content)
        print(f"Successfully created SVG at {svg_file_path}")
    except IOError as e:
        print(f"Error creating SVG: {e}")


def image_to_text(input_image_path, output_text_path):
    # Load image
    img = Image.open(input_image_path).convert("L")  # convert to grayscale (0–255)

    # Convert to numpy array
    pixel_array = np.array(img)

    # Save as text file
    with open(output_text_path, "w") as f:
        for row in pixel_array:
            row_str = " ".join(str(int(pixel)) for pixel in row)
            f.write(row_str + "\n")

    print("Saved 2D grayscale values to:", output_text_path)


def image_to_npy(input_image_path, output_npy_path):
    # Load the image
    img = Image.open(input_image_path)

    # Convert to NumPy array
    arr = np.array(img)

    # Save as .npy binary file
    np.save(output_npy_path, arr)

    print("Saved .npy file to:", output_npy_path)


def make_circle_image(input_path, output_path, size=256):
    # Open the image
    img = Image.open(input_path).convert("RGBA")

    # Resize to square (keeping aspect ratio, then center-crop)
    img.thumbnail((size, size), Image.LANCZOS)
    w, h = img.size
    square = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    square.paste(img, ((size - w) // 2, (size - h) // 2))

    # Create circular mask
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size, size), fill=255)

    # Apply mask
    square.putalpha(mask)

    # Save result
    square.save(output_path, format="PNG")


if __name__ == "__main__":
    # --- Example Usage ---
    # Make sure 'download.jpg' exists in the same directory as your script
    input_file = "tempdata/postge.png"
    output_file = "tempdata/output_transparent.png"
    # Check if the input file exists for the example
    if os.path.exists(input_file):

        remove_bg(input_file, output_file)
    else:
        print(
            f"Error: {input_file} not found. Please create or provide the correct path to a JPG file."
        )

    # create_svg_from_png("tempdata/soccer-player.png", "tempdata/output_image.svg")

    # image_to_text("tempdata/download.jpg", "tempdata/img_output.txt")

    # image_to_npy("tempdata/download.jpg", "tempdata/img_output.npy")
