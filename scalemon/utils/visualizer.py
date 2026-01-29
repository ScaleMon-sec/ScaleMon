import numpy as np
import matplotlib.pyplot as plt

def visualize_two_channels(
    arr,
    blue_color=(0, 0, 255),
    red_color=(255, 0, 0),
    mask=False,
    save_path=None,
    dpi=300
):
    arr = arr[:, ::-1, :] 

    if arr.shape[0] != 2:
        raise ValueError("Input must be shaped (2, H, W)")

    ch0 = arr[0]
    ch1 = arr[1]

    if mask:
        ch0_norm = (ch0 != 0).astype(np.float32)
        ch1_norm = (ch1 != 0).astype(np.float32)
    else:
        def normalize(x):
            x = x.astype(np.float32)
            if x.max() == x.min():
                return np.zeros_like(x)
            return (x - x.min()) / (x.max() - x.min())
        ch0_norm = normalize(ch0)
        ch1_norm = normalize(ch1)

    H, W = ch0.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    rgb[:, :, 0] = ch1_norm * (red_color[0]/255) + ch0_norm * (blue_color[0]/255)
    rgb[:, :, 1] = ch1_norm * (red_color[1]/255) + ch0_norm * (blue_color[1]/255)
    rgb[:, :, 2] = ch1_norm * (red_color[2]/255) + ch0_norm * (blue_color[2]/255)

    background_mask = (ch0_norm == 0) & (ch1_norm == 0)
    rgb[background_mask] = 1.0

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb)
    
    ax.set_xticks(np.linspace(0, W, 20))
    ax.set_yticks(np.linspace(0, H, 20))

    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

    ax.set_axis_off()

    plt.show()


    return rgb
