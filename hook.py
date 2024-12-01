def register_hook(model):
    layer_activations = {}
    hooks = []  # 用于存储多个钩子

    def hook_fn(module, input, output):
        # 将每个层的激活保存为独立的键
        layer_activations[module] = {'input': input, 'output': output}

    # 在需要的层上注册钩子，例如 conv1 层
    #选取对应权重路径，可使用权重文件获取
    hooks.append(model.backbone.layers[0].register_forward_hook(hook_fn)) #.adapterpromptp2
    hooks.append(model.backbone.layers[1].register_forward_hook(hook_fn))
    hooks.append(model.backbone.layers[2].register_forward_hook(hook_fn))
    hooks.append(model.backbone.layers[3].register_forward_hook(hook_fn))

    return hooks, layer_activations

if  __name__='__main__':
    model=model()
    hook, layer_activations = register_hook(model)
    input_activation = [layer_activations[model.backbone.layers[i]]['input'] for i in range(len(layer_activations))]
    output_activation = [layer_activations[model.backbone.layers[i]]['output'] for i in range(len(layer_activations))]

    