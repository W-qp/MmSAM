# TODO:Show network structure, GFLOPS, Params, memory, speed, FPS
import time
import torch
from prettytable import PrettyTable
from thop import clever_format, profile
from torch.backends import cudnn
from model.Build_models import MLoRA_SAM2, FT_SAM2, Adapter_SAM2, BitFit_SAM2


def Params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6}M")

    requires_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Requires_grad parameters: {requires_grad_params / 1e6}M")

    ratio = requires_grad_params / total_params
    print(f'ratio={ratio * 100}%')

    return total_params, requires_grad_params, ratio


def compute_speed(model, input_size, device, n, operations='* & +', iteration=300):
    torch.cuda.set_device(device)
    cudnn.benchmark = True

    model.eval()
    model = model.cuda()
    input = []
    for i in range(1, n + 1):
        input_img = torch.randn(input_size, device=device)
        input.append(input_img)

    flops, _ = profile(model.to(device), (input,), verbose=False)
    flops = flops * 2 if operations == '* & +' else flops
    torch.cuda.empty_cache()

    params, trained_params, r = Params(model)
    print(f'GFLOPs = {flops / 1000 ** 3}')
    flops, params, trained_params = clever_format([flops, params, trained_params], "%.3f")

    for _ in range(50):
        model(input)
        # if _ == 40:
        #     print(torch.cuda.memory_summary())

    print('Total Params: %s' % params)
    print('Total FLOPS: %s' % flops)
    print('========= Calculate FPS=========')
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iteration):
        model(input)
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    speed_time = elapsed_time / iteration * 1000
    fps = iteration / elapsed_time

    print('Elapsed Time: [%.6f s / %d iter]' % (elapsed_time, iteration))
    print('Speed Time: %.6f ms / iter   FPS: %.6f' % (speed_time, fps))

    return flops, params, trained_params, r, speed_time, fps


if __name__ == "__main__":
    # args
    input_shape = [1, 3, 1024, 1024]
    n_imgs = 2
    operation = '*'  # write matrix multiplication and addition as only one operation
    # operation = '* & +'  # write matrix multiplication and addition as two operations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_chans = input_shape[1] if n_imgs == 1 else tuple([input_shape[1] for _ in range(n_imgs)])
    n_classes, img_size = 1, input_shape[-1]
    net_type = 'Adapter_SAM2'

    if net_type == 'FT_SAM2':
        net = FT_SAM2(n_classes=n_classes, in_chans=in_chans, topk=1, model_type="s", img_size=img_size)
    elif net_type == 'MLoRA_SAM2':
        net = MLoRA_SAM2(n_classes=n_classes, in_chans=in_chans, topk=1, model_type="s", img_size=img_size)
    elif net_type == 'Adapter_SAM2':
        net = Adapter_SAM2(n_classes=n_classes, in_chans=in_chans, topk=1, model_type="s", img_size=img_size)
    elif net_type == 'BitFit_SAM2':
        net = BitFit_SAM2(n_classes=n_classes, in_chans=in_chans, topk=1, model_type="s", img_size=img_size)
    else:
        raise NotImplementedError(f"Model type:'{net_type}' does not exist, please check the 'net_type' arg!")

    model = net.to(device)
    # for i in model.children():
    #     print(i)
    #     print('==============================')

    with torch.no_grad():
        flops, params, trained_params, r, speed_time, fps = compute_speed(model, input_shape, device=0, n=n_imgs, operations=operation)
    table = PrettyTable()
    table.field_names = ['', 'Value']
    table.add_row(['Params', f'{trained_params} / {params} / {r*100}%'])
    table.add_row(['FLOPS', f'{flops}'])
    table.add_row(['FPS', round(fps, 3)])
    table.add_row(['Speed', f'{round(speed_time, 3)}ms/iter'])
    print('\nShow:')
    print(table)
