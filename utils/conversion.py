def trunc_denorm(image, trunc_max=3072.0, trunc_min=-1024.0, norm_range_max=3072.0, norm_range_min=-1024.0):
    image = denormalize(image, norm_range_max, norm_range_min)
    image = trunc(image, trunc_max, trunc_min)
    return image


def denormalize(image, norm_range_max=3072.0, norm_range_min=-1024.0):
    image = image * (norm_range_max - norm_range_min) + norm_range_min
    return image


def trunc(image, trunc_max, trunc_min):
    image[image <= trunc_min] = trunc_min
    image[image >= trunc_max] = trunc_max
    return image
