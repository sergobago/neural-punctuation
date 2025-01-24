from numpy.random import rand

alpha_sub = 0.40
alpha_del = 0.40

def augment_none(model_options, index, x, y_mask, y, x_aug, y_mask_aug, y_aug):
    x_aug.append(x[index])
    y_aug.append(y[index])
    y_mask_aug.append(y_mask[index])

def augment_substitute(model_options, index, x, y_mask, y, x_aug, y_mask_aug, y_aug):
    x_aug.append(model_options['UNK'])
    y_aug.append(y[index])
    y_mask_aug.append(y_mask[index])

def augment_insert(model_options, index, x, y_mask, y, x_aug, y_mask_aug, y_aug):
    if len(x) - index + len(x_aug) < model_options['INPUT_LENGTH']:
        x_aug.append(model_options['UNK'])
        y_aug.append(0)
        y_mask_aug.append(1)

    x_aug.append(x[index])
    y_aug.append(y[index])
    y_mask_aug.append(y_mask[index])

def augment_delete(model_options, index, x, y_mask, y, x_aug, y_mask_aug, y_aug):
    return

def augment_all(*args):
    chance_augmentation = rand()

    if chance_augmentation < alpha_sub:
        augment_substitute(*args)
    elif chance_augmentation < alpha_sub + alpha_del:
        augment_delete(*args)
    else:
        augment_insert(*args)

def get_augment(model_options, tokenizer, augment_rate, x, y_mask, y):
    x_aug = []
    y_aug = []
    y_mask_aug = []
    excluded_augment_values = set(tokenizer.all_special_ids)

    for index in range(len(x)):
        if y_mask[index] == 0 and x[index] not in excluded_augment_values and rand() < augment_rate:
            augment_all(model_options, index, x, y_mask, y, x_aug, y_mask_aug, y_aug)
        else:
            augment_none(model_options, index, x, y_mask, y, x_aug, y_mask_aug, y_aug)

    return x_aug, y_mask_aug, y_aug

AUGMENTATIONS = {
    'NONE': augment_none,
    'SUBSTITUTE': augment_substitute,
    'DELETE': augment_delete,
    'ALL': augment_all,
}
