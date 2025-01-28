import cv2
import numpy as np
import pytesseract

# Function to show image with a title and wait for key press to continue
def show_image(title, img):
    cv2.imshow(title, img)  # Show the image
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()  # Close the window after key press

# Function to perform OCR and write results to a text file
def perform_ocr(image, step_name, file):
    text = pytesseract.image_to_string(image)
    ocr_text = f"OCR after {step_name}:\n{text}\n{'-'*40}\n"
    print(ocr_text)  # Optionally print to the terminal
    file.write(ocr_text)  # Write the OCR result to the text file
    return text

# Function to remove noise and preprocess the image
def preprocess_image(image_path, output_file):
    # Read the image
    img = cv2.imread(image_path)

    # Show the original image and perform OCR
    show_image("Original Image", img)
    original_text = perform_ocr(img, "Original Image", output_file)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_image("Grayscale Image", gray)
    gray_text = perform_ocr(gray, "Grayscale Image", output_file)

    # Remove noise using Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    show_image("Gaussian Blur", blur)
    blur_text = perform_ocr(blur, "Gaussian Blur", output_file)

    # Apply binary thresholding (Otsu's method)
    _, thresholded = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    show_image("Binary Thresholded Image", thresholded)
    thresholded_text = perform_ocr(thresholded, "Binary Thresholded Image", output_file)

    # Alternative: Use adaptive thresholding (for varying lighting conditions)
    adaptive_thresholded = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    show_image("Adaptive Thresholded Image", adaptive_thresholded)
    adaptive_text = perform_ocr(adaptive_thresholded, "Adaptive Thresholded Image", output_file)

    # Dilation and erosion to close gaps in characters
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresholded, kernel, iterations=1)
    show_image("Dilated Image", dilated)
    dilated_text = perform_ocr(dilated, "Dilated Image", output_file)

    eroded = cv2.erode(dilated, kernel, iterations=1)
    show_image("Eroded Image", eroded)
    eroded_text = perform_ocr(eroded, "Eroded Image", output_file)

    # Deskewing the image (rotate image to correct skew)
    coords = np.column_stack(np.where(eroded > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # Correct the skew by rotating the image
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = eroded.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(eroded, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    show_image("Deskewed Image", rotated)
    rotated_text = perform_ocr(rotated, "Deskewed Image", output_file)

    # Resize image if needed (increase image size for better OCR accuracy)
    resized = cv2.resize(rotated, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    show_image("Resized Image", resized)
    resized_text = perform_ocr(resized, "Resized Image", output_file)

    return resized

# Main function
def main(image_path, output_file_path):
    # Open the file in write mode
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        # Preprocess the image and perform OCR at each step
        final_image = preprocess_image(image_path, output_file)

        # Perform final OCR after all steps
        final_text = pytesseract.image_to_string(final_image)
        final_output = f"Final OCR result after all preprocessing steps:\n{final_text}\n"
        print(final_output)  # Optionally print to the terminal
        output_file.write(final_output)  # Write the final OCR result to the text file

if __name__ == "__main__":
    image_path = "path_to_your_image.jpg"  # Replace with your image path
    output_file_path = "ocr_output.txt"  # The file where OCR results will be saved
    main(image_path, output_file_path)
