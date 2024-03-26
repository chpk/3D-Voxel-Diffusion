import torch
from video_diffusion_pytorch_image_cond import Unet3D, Trainer, GaussianDiffusion, gif_to_tensor, video_tensor_to_gif
from torchvision import transforms as T
from PIL import Image  
import PIL 
import numpy as np

# load model 
model = Unet3D(
    dim = 64,
    cond_dim = 768,
    dim_mults = (1, 2, 4, 8),
)

diffusion = GaussianDiffusion(
    model,
    channels = 1,
    image_size = 64,
    num_frames = 56,
    timesteps = 400,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    folder = "new_surogate_model_dataset/",
    cond_vector_folder = "condition_vectors/",
    train_batch_size = 2,
    train_lr = 1e-4,
    save_and_sample_every = 400,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True                        # turn on mixed precision
) 

# load pretrained weights
trainer.load("88")
# prepare condition gif

cond = torch.from_numpy(np.load("C:/Users/cpremithkumar/Desktop/DDPM_latest_code/condition_vectors/design_BCC600_stl.npy")).squeeze(0).cuda()
print(cond.shape)
# sample
samples = diffusion.p_sample_loop(shape = (1, 1, 56, 64, 64), cond=cond)

# save samples
#samples = (samples + 1) / 2 # unnormalize
#samples = samples.clamp(0, 1)
#samples = samples.permute(0, 2, 3, 4, 1).cuda() # BCTHW -> BTHWC
print(len(samples))
print(samples.shape)
print(samples[0].shape)
print(samples[0][0].shape)

# # def video_tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
# #     images = map(T.ToPILImage(), tensor.unbind())
# #     first_img, *rest_imgs = images
# #     first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
# #     return images

# for i in range (0,63):
#     im1 = T.ToPILImage()(samples[0][i][1])
#     im1 = im1.save("out/"+str(i)+".jpg") 

video_tensor_to_gif(samples[0], "Temp_image_cond_gif_save.gif")