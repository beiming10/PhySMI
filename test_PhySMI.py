
import torch
import argparse
import os
import numpy as np
from tifffile import imread, imwrite
import json

from FSR_PhySMI import PhySMI,FSRWithFAN
import time

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

import matplotlib.pyplot as plt

def load_model(model_path, config_data):
    """
    Load trained PhySMI model
    """
    signal_init, background_init = None, None
    if config_data.get('endmember_path') and os.path.exists(config_data['endmember_path']):
        print(f"Loading endmember dictionary: {config_data['endmember_path']}")
        try:
            endmembers = np.loadtxt(config_data['endmember_path'], delimiter='\t')
        except:
            endmembers = np.loadtxt(config_data['endmember_path'], delimiter=',')
        endmembers = torch.from_numpy(endmembers).float()
        if endmembers.shape[0] == config_data['num_endmembers']:
            signal_init, background_init = endmembers[:-1, :], endmembers[-1, :]
    
    num_input_channels_per_branch = len(config_data['input_indices_a'])
    print(f"Generator branch input channels: {num_input_channels_per_branch}")
    
    if config_data.get('model_type') == 'FSRWithFAN':
        ModelClass = FSRWithFAN
    else:
        ModelClass = PhySMI
        
    model = ModelClass(
        in_channels=num_input_channels_per_branch, 
        out_channels=config_data['hyper_frames'], 
        num_endmembers=config_data['num_endmembers'],
        signal_init=signal_init,
        background_init=background_init,
        freeze_signal=True,
        freeze_background=True
    )
    
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Successfully loaded model weights: {model_path}")
    return model


def preprocess_input(image_path, config_data):
    """
    Preprocess input image
    """
    print(f"Reading input image: {image_path}")
    image = imread(image_path).astype(np.float32)
    
    if 'scalefactor' in config_data:
        image = image / config_data['scalefactor']
    
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    elif len(image.shape) != 4:
        raise ValueError(f"Input image dimension error, expected (frames, height, width) or (batch, frames, height, width), got {image.shape}")
    
    input_indices_a = config_data['input_indices_a']
    input_indices_b = config_data['input_indices_b']
    
    subset_a = image[:, input_indices_a, :, :]
    subset_b = image[:, input_indices_b, :, :]
    
    subset_a = torch.from_numpy(subset_a)
    subset_b = torch.from_numpy(subset_b)
    
    return subset_a, subset_b, image


def run_inference(model, subset_a, subset_b, device, output_mode):
    """
    Run model inference and select or fuse outputs based on output_mode
    """
    model.eval()
    with torch.no_grad():
        subset_a = subset_a.to(device)
        subset_b = subset_b.to(device)
        
        start_time = time.time()
        
        recon_A, abundances_A = model(subset_a)
        recon_B, abundances_B = model(subset_b)
        
        if output_mode == 'dual':
            #NEED DOUBLE INPUT , Not recommended
            recon_final = (recon_A + recon_B) / 2.0
            abundances_final = (abundances_A + abundances_B) / 2.0
            
        elif output_mode == 'single_a':
            #Only use subset_a
            recon_final = recon_A
            abundances_final = abundances_A
            
        elif output_mode == 'single_b':
            recon_final = recon_B
            abundances_final = abundances_B
            
        else:
            raise ValueError(f"Unsupported output mode: {output_mode}")
        
        inference_time = time.time() - start_time

        recon_final = recon_final.cpu().numpy()
        abundances_final = abundances_final.cpu().numpy()
        
    return recon_final, abundances_final, inference_time


def postprocess_output(recon_final, abundances_final, config_data, original_image=None):
    """
    Postprocess output results
    """

    if 'scalefactor' in config_data:
        recon_final = recon_final * config_data['scalefactor']
        abundances_final = abundances_final * config_data['scalefactor']
    

    recon_final = recon_final.astype(np.uint16)
    abundances_final = abundances_final.astype(np.uint16)
    
    return recon_final, abundances_final


def save_results(recon_final, abundances_final, output_dir, input_filename, output_mode):
    """
    Save inference results with output mode in filename
    """
    create_dir(output_dir)
    
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    output_prefix = f"{base_name}_{output_mode}" 
    
    recon_path = os.path.join(output_dir, f"{output_prefix}_recon.tif")
    imwrite(recon_path, recon_final.squeeze(0))
    print(f"Saved reconstruction to: {recon_path}")
    
    abundances_path = os.path.join(output_dir, f"{output_prefix}_abundances.tif")
    imwrite(abundances_path, abundances_final.squeeze(0))
    print(f"Saved abundances to: {abundances_path}")
    
    return recon_path, abundances_path


def visualize_results(original, reconstructed, abundances, config_data, output_dir, input_filename, output_mode):
    """
    Visualize results with output mode in filename
    """
    try:
        import matplotlib.pyplot as plt
        
        viz_dir = os.path.join(output_dir, 'visualization')
        create_dir(viz_dir)
        
        base_name = os.path.splitext(os.path.basename(input_filename))[0]
        output_prefix = f"{base_name}_{output_mode}" 
        
        if original is not None:
            mid_frame = original.shape[1] // 2
            
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(original[0, mid_frame], cmap='gray')
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(132)
            plt.imshow(reconstructed[0, mid_frame], cmap='gray')
            plt.title(f'Reconstruction ({output_mode})')
            plt.axis('off')
            
            plt.subplot(133)
            plt.imshow(abundances[0, 0], cmap='jet')
            title_suffix = f" ({config_data.get('stc_channels', ['Endmember1'])[0]})"
            plt.title(f'Abundance{title_suffix}')
            plt.axis('off')
            
            plt.tight_layout()
            viz_path = os.path.join(viz_dir, f"{output_prefix}_comparison.png")
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved comparison visualization to: {viz_path}")
        
        if 'stc_channels' in config_data:
            num_endmembers = len(config_data['stc_channels'])
            fig, axes = plt.subplots(1, num_endmembers, figsize=(5 * num_endmembers, 5))
            
            if num_endmembers == 1:
                axes = [axes]
                
            for i in range(num_endmembers):
                axes[i].imshow(abundances[0, i], cmap='jet')
                axes[i].set_title(f"Abundance ({config_data['stc_channels'][i]})")
                axes[i].axis('off')
            
            plt.tight_layout()
            abund_viz_path = os.path.join(viz_dir, f"{output_prefix}_all_abundances.png")
            plt.savefig(abund_viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved all abundances to: {abund_viz_path}")
            
    except ImportError:
        print("Warning: matplotlib not installed, visualization disabled.")
    except Exception as e:
        print(f"Error generating visualization: {e}")


def process_single_image(model, image_path, config_data, output_dir, device, visualize=True, output_mode='dual'):
    """
    Process single image file
    """
    try:
        subset_a, subset_b, original_image = preprocess_input(image_path, config_data)
        
        recon_final, abundances_final, inference_time = run_inference(model, subset_a, subset_b, device, output_mode)
        
        recon_final, abundances_final = postprocess_output(recon_final, abundances_final, config_data)
        
        recon_path, abundances_path = save_results(recon_final, abundances_final, output_dir, image_path, output_mode)
        
        if visualize:
            if 'scalefactor' in config_data:
                original_display = original_image * config_data['scalefactor']
            else:
                original_display = original_image.copy()
            visualize_results(original_display, recon_final, abundances_final, config_data, output_dir, image_path, output_mode)
        
        print(f"Processed {os.path.basename(image_path)} (mode: {output_mode}), inference time: {inference_time:.4f}s")
        
        return recon_path, abundances_path
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="PIANet-D Model Inference Script")
    parser.add_argument('--config', type=str, required=True, help='Path to JSON config file')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input TIFF image path or directory containing TIFF images')
    parser.add_argument('--output', type=str, default='./results_pianet', help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--no-visualize', action='store_true', help='Disable visualization')
    
    parser.add_argument('--output-mode', type=str, default='dual', 
                        choices=['dual', 'single_a', 'single_b'],
                        help='Output mode: dual (fuse A and B), single_a (Branch A only), single_b (Branch B only)')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config_data = json.load(f)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = load_model(args.model, config_data)
    model = model.to(device)
    
    create_dir(args.output)
    
    config_save_path = os.path.join(args.output, 'config.json')
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)
    print(f"Saved config to: {config_save_path}")
    
    if os.path.isfile(args.input):
        process_single_image(model, args.input, config_data, args.output, device, not args.no_visualize, args.output_mode)
    elif os.path.isdir(args.input):
        tiff_files = [f for f in os.listdir(args.input) if f.lower().endswith(('.tif', '.tiff'))]
        total_files = len(tiff_files)
        print(f"Found {total_files} TIFF files to process, output mode: {args.output_mode}")
        
        for i, tiff_file in enumerate(tiff_files):
            tiff_path = os.path.join(args.input, tiff_file)
            print(f"Processing file {i+1}/{total_files}: {tiff_file}")
            process_single_image(model, tiff_path, config_data, args.output, device, not args.no_visualize, args.output_mode)
    else:
        print(f"Error: Input path {args.input} does not exist or is not a file/directory.")
        return
    
    print("All images processed!")


if __name__ == '__main__':
    main()
