import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

img = cv2.imread('./dataset/train/images/1.jpg')

input_img = img[np.newaxis, :, :, :]

blurred_img = iaa.GaussianBlur((0, 3.0))(images=input_img)
blurred_img = blurred_img.squeeze()
cv2.imshow("Blur Result", blurred_img)

bright_changed_img = iaa.Add((-50, 50), per_channel=0.5)(images=input_img)
bright_changed_img = bright_changed_img.squeeze()
cv2.imshow("Bright Change Result", bright_changed_img)

hls_changed_img = iaa.AddToHueAndSaturation((-20, 20))(images=input_img)
hls_changed_img = hls_changed_img.squeeze()
cv2.imshow("HLS Changed Result", hls_changed_img)

blended_alpha_noise_img = iaa.BlendAlphaFrequencyNoise(
                                exponent=(-4, 0),
                                foreground=iaa.Multiply((0.5, 1.5), per_channel=True),
                                background=iaa.contrast.LinearContrast((0.5, 2.0))
                            )(images=input_img)
blended_alpha_noise_img = blended_alpha_noise_img.squeeze()
cv2.imshow("Alpha Frequency Noise Result", blended_alpha_noise_img)

contrast_img = iaa.contrast.LinearContrast((0.5, 2.0), per_channel=0.5)(images=input_img)
contrast_img = contrast_img.squeeze()
cv2.imshow("Contrast Result", contrast_img)

grayscale_img = iaa.Grayscale(alpha=(0.0, 1.0))(images=input_img)
grayscale_img = grayscale_img.squeeze()
cv2.imshow("Grayscale Result", grayscale_img)

sharpened_img = iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))(images=input_img)
sharpened_img = sharpened_img.squeeze()
cv2.imshow("Sharpen Result", sharpened_img)

cv2.waitKey()