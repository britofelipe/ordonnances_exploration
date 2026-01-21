import cv2

def crop_image(image_path, output_path):
    img = cv2.imread(image_path)
    roi = cv2.selectROI("Crop medication", img, showCrosshair=True)
    x, y, w, h = roi
    cropped = img[y:y+h, x:x+w]
    cv2.imwrite(output_path, cropped)
    cv2.destroyAllWindows()
    print(f"Cropped image saved as '{output_path}'")

if __name__ == "__main__":
    import os
    if not os.path.exists("test_crop_folder"):
        os.makedirs("test_crop_folder")
        
    input_image_path = r"ordo_imgs\ordo_0000.png"
    output_image_path = r"test_crop_folder\cropped_ordo_test.png"

    crop_image(input_image_path, output_image_path)