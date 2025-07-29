from PIL import Image, ImageDraw
from pdf2image import convert_from_path

# Padding between images
padding = 30
border_color = 'black'
border_width = 3  # Thickness of the border

# Convert PDFs to images
img1 = convert_from_path('best_model_mae_distribution.pdf')[0]
img2 = convert_from_path('best_model_mse_distribution.pdf')[0]
img3 = convert_from_path('best_model_rmse_distribution.pdf')[0]
img4 = convert_from_path('best_model_residuals.pdf')[0]

# Resize all to the same size
width, height = img1.size
img2 = img2.resize((width, height))
img3 = img3.resize((width, height))
img4 = img4.resize((width, height))

# New canvas size
merged_width = 2 * width + padding
merged_height = 2 * height + padding
merged_img = Image.new('RGB', (merged_width, merged_height), color='white')

# Coordinates for pasting
positions = [
    (0, 0),  # top-left
    (width + padding, 0),  # top-right
    (0, height + padding),  # bottom-left
    (width + padding, height + padding)  # bottom-right
]

# Images list
images = [img1, img2, img3, img4]

# Draw images and borders
draw = ImageDraw.Draw(merged_img)
for (x, y), img in zip(positions, images):
    merged_img.paste(img, (x, y))
    # Draw rectangle border around the image
    draw.rectangle(
        [x, y, x + width, y + height],
        outline=border_color,
        width=border_width
    )

# Save output
merged_img.save('combined_figure_with_borders.png')
