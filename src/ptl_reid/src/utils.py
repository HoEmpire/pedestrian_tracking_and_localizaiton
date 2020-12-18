import numpy as np


def image_block_preprocess(img_block):
    # too long
    if (img_block.shape[0] > img_block.shape[1] * 2):
        zero_pad_width = int((img_block.shape[0] / 2 - img_block.shape[1]) / 2)
        zero_pad_height = img_block.shape[0]
        zero_pad_half = np.zeros(
            (zero_pad_height, zero_pad_width, 3)).astype('uint8')
        img_block_new = np.concatenate((zero_pad_half, img_block), 1)
        img_block_new = np.concatenate((img_block_new, zero_pad_half), 1)
    else:
        zero_pad_width = img_block.shape[1]
        zero_pad_height = int(img_block.shape[1] - img_block.shape[0] / 2)
        zero_pad_half = np.zeros(
            (zero_pad_height, zero_pad_width, 3)).astype('uint8')
        img_block_new = np.concatenate((zero_pad_half, img_block), 0)
        img_block_new = np.concatenate((img_block_new, zero_pad_half), 0)
    return img_block_new


if __name__ == "__main__":
    img_block = np.zeros((256, 100, 3))
    img_block = image_block_preprocess(img_block)
    print(img_block.shape)
