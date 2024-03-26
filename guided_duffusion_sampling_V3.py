import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer, gif_to_tensor, video_tensor_to_gif
from PIL import Image, ImageDraw, ImageSequence
import os
# Assuming the surrogate model and its loader function are defined in a separate file
from surrogate_model_inference_V9 import load_model, predict_quality, predict_quality_V2
from torchvision import transforms as T
import gc
# Load the trained video diffusion model
model = Unet3D(dim=64, dim_mults=(1, 2, 4, 8))
diffusion = GaussianDiffusion(model, channels=1, image_size=64, num_frames=56, timesteps=400).cuda()

# Load pretrained weights for the video diffusion model
trainer = Trainer(diffusion, 'test_sampling/')
trainer.load("40")  # Load the latest checkpoint

# Load the surrogate model
surrogate_model = load_model('best_model_fold_3.pth')

def unnormalize_img(t):
    return (t + 1) * 0.5


def process_gif(gif_path):
    # Open the gif file
    with Image.open(gif_path) as gif:
        frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
    
    # If frames are less than 64, add black frames
    while len(frames) < 64:
        frames.append(Image.new('RGB', (64, 64), color='black'))

    # If frames are more than 64, keep only the first 64
    frames = frames[:64]
    
    # Create a new image with an 8x8 grid
    grid_img = Image.new('RGB', (512, 512))
    
    # Paste the frames into the grid
    for i, frame in enumerate(frames):
        row, col = divmod(i, 8)
        grid_img.paste(frame, (col * 64, row * 64))
    
    # Save the result
    #"temporary_folder_save_gif/temp_output_"+str(t)+".gif"
    
    temporay_image_path = "temporary_folder_save_image/" + gif_path.split("temporary_folder_save_gif/")[1].split(".gif")[0] + ".jpg"
    grid_img.save(temporay_image_path, 'JPEG')
    
    
    return temporay_image_path

def process_gif_V1(tensor):
    """
    Stack the frames in the tensor to an 8x8 grid representation.
    :param tensor: Tensor of shape [channels, num_frames, height, width].
    :return: Tensor in grid format.
    """
    channels, num_frames, height, width = tensor.shape

    # Add a black frame if there are 63 frames to make it 64
    if num_frames == 63:
        black_frame = torch.zeros((channels, 1, height, width), device=tensor.device)
        tensor = torch.cat((tensor, black_frame), dim=1)
    elif num_frames != 64:
        raise ValueError("The tensor does not have the expected number of frames for an 8x8 grid.")

    # Reshape and rearrange the tensor to create an 8x8 grid
    grid_tensor = tensor.view(channels, 8, 8, height, width)
    grid_tensor = grid_tensor.permute(1, 3, 2, 4, 0).reshape(512, 512, channels)
    print(grid_tensor.shape)
    # Return the reshaped tensor
    return grid_tensor

def quality_assessment_fn(x):
    """
    Function to assess the quality of the image using the surrogate model.
    :param x: The current image tensor.
    :return: Gradient based on the quality score.
    """
    x_proc = x[0]
    x_in = x_proc.detach().requires_grad_(True)
    processed_tensor = process_gif_V1(x_in)  # Modify process_gif_V1 to work directly on the tensor

    # Ensure processed_tensor has requires_grad=True
    processed_tensor.requires_grad_(True)

    # Get the quality score as a tensor from the surrogate model
    quality_score_tensor = predict_quality(surrogate_model, processed_tensor)

    # Ensure the surrogate model's output is a scalar tensor
    if quality_score_tensor.ndim != 0:
        raise ValueError("Quality score must be a scalar tensor.")

    # Compute the gradient
    quality_grad = torch.autograd.grad(quality_score_tensor, x_in, allow_unused=True)[0]
    #print(quality_grad)
    if quality_grad is None:
        raise RuntimeError("Failed to compute gradient. Ensure the computation graph is maintained.")

    return quality_grad

def condition_mean(p_mean_var, x):
    """
    Adjust the mean based on quality feedback from the surrogate model.
    """
    quality_gradient = quality_assessment_fn(x)

    # Adjust the mean based on the quality gradient
    new_mean = p_mean_var["mean"] + p_mean_var["variance"] * quality_gradient
    return new_mean

def p_sample(x, t, cond=None, cond_scale=1.0, clip_denoised=True):
    """
    Sample from the model using quality-guided adjustments.
    """
    #b, *_, device = *x.shape, x.device
    #model_mean, _, model_log_variance = diffusion.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, cond=cond, cond_scale=cond_scale)
    b, *_, device = *x.shape, x.device
    #t_tensor = torch.tensor([t], device=device)  # Convert t to a tensor and ensure it's on the correct device

    model_mean, _, model_log_variance = diffusion.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, cond=cond, cond_scale=cond_scale)
    # Apply quality-guided adjustment to the model mean
    adjusted_mean = condition_mean({"mean": model_mean, "variance": model_log_variance.exp()}, x)

    noise = torch.randn_like(x)
    #noise = x
    nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
    
    temporary_unnorm_image = adjusted_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise # check this !!!
    
    return temporary_unnorm_image

def generate_quality_guided_gif(input_gif):
    input_tensor = gif_to_tensor(input_gif)
    input_tensor = input_tensor.unsqueeze(0).cuda()
    #input_tensor = input_tensor*255
    t = diffusion.num_timesteps - 1
    print("starting to add noise")
    noisy_input = diffusion.q_sample(input_tensor, torch.full((1,), t, device=input_tensor.device, dtype=torch.long))
    print("added noise")
    #noisy_input = unnormalize_img(noisy_input)
    video_tensor_to_gif(noisy_input[0], "temp_noise_save.gif")
    # Initialize the diffusion process
    quality_threshold = 0.55
    current_quality = 0
    #t = 0  # Initialize the diffusion timestep
    input_tensor = noisy_input.cuda()
    device = input_tensor.device
    while current_quality < quality_threshold and t >= 0:
        # Perform the guided diffusion sampling step
        print("present t is -- " + str(t))
        b = input_tensor.shape[0]
        print("cond_cale:" + str((quality_threshold*10)-(current_quality*10)))
        #output = p_sample(input_tensor, torch.full((b,), t, device=device, dtype=torch.long), cond=input_tensor, cond_scale=float((quality_threshold*10)-(current_quality*10)), clip_denoised=True)
        output = p_sample(input_tensor, torch.full((b,), t, device=device, dtype=torch.long), cond=input_tensor, cond_scale=1.0, clip_denoised=True)
        input_tensor = output.detach()
        # Assess the quality of the current output
        temp_gif_path = "temporary_folder_save_gif/temp_output_"+str(t)+".gif"
        temporary_ouput_gif = unnormalize_img(output)
        video_tensor_to_gif(temporary_ouput_gif[0], temp_gif_path)
        temp_image_path = process_gif(temp_gif_path)
        current_quality = predict_quality_V2(surrogate_model, temp_image_path)
        print(current_quality)
        # Update the input for the next timestep
        
        t = t-1  # Increment timestep
        
        gc.collect()
        torch.cuda.empty_cache()

    # Finalize and return the output GIF
    final_output = video_tensor_to_gif(input_tensor[0].cpu(), "output_quality_guided.gif")
    return final_output

# Example usage
input_gif_path = 'test_sampling/design_BCC420_stl.gif'
output_gif = generate_quality_guided_gif(input_gif_path)
