import os
from collections import OrderedDict
import copy

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

import arg_parser
import evaluation
import unlearn
import utils
from trainer import validate
from unlearn.mia import membership_inference_attack


def replace_loader_dataset(dataset, batch_size, seed=1, shuffle=True):
    utils.setup_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=shuffle,
    )


def main():
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed

    model, train_loader_full, val_loader, test_loader, marked_loader = utils.setup_model_dataset(args)
    model.cuda()

    forget_dataset = copy.deepcopy(marked_loader.dataset)
    if args.dataset == "svhn":
        try:
            marked = forget_dataset.targets < 0
        except:
            marked = forget_dataset.labels < 0
        forget_dataset.data = forget_dataset.data[marked]
        try:
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
        except:
            forget_dataset.labels = -forget_dataset.labels[marked] - 1
        forget_loader = replace_loader_dataset(forget_dataset, args.batch_size, seed=seed, shuffle=True)
        retain_dataset = copy.deepcopy(marked_loader.dataset)
        try:
            marked = retain_dataset.targets >= 0
        except:
            marked = retain_dataset.labels >= 0
        retain_dataset.data = retain_dataset.data[marked]
        try:
            retain_dataset.targets = retain_dataset.targets[marked]
        except:
            retain_dataset.labels = retain_dataset.labels[marked]
        retain_loader = replace_loader_dataset(retain_dataset, args.batch_size, seed=seed, shuffle=True)
        assert len(forget_dataset) + len(retain_dataset) == len(train_loader_full.dataset)
    else:
        try:
            marked = forget_dataset.targets < 0
            forget_dataset.data = train_loader_full.dataset.data[marked]
            forget_dataset.targets = train_loader_full.dataset.targets[marked]
            forget_loader = replace_loader_dataset(forget_dataset, args.batch_size, seed=seed, shuffle=True)
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.data = train_loader_full.dataset.data[marked]
            retain_dataset.targets = train_loader_full.dataset.targets[marked]
            retain_loader = replace_loader_dataset(retain_dataset, args.batch_size, seed=seed, shuffle=True)
            assert len(forget_dataset) + len(retain_dataset) == len(train_loader_full.dataset)
        except:
            marked = forget_dataset.targets < 0
            forget_dataset.data = train_loader_full.dataset.data[marked]
            forget_dataset.targets = train_loader_full.dataset.targets[marked]
            forget_loader = replace_loader_dataset(forget_dataset, args.batch_size, seed=seed, shuffle=True)
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.data = train_loader_full.dataset.data[marked]
            retain_dataset.targets = train_loader_full.dataset.targets[marked]
            retain_loader = replace_loader_dataset(retain_dataset, args.batch_size, seed=seed, shuffle=True)
            assert len(forget_dataset) + len(retain_dataset) == len(train_loader_full.dataset)

    print(f"number of retain dataset {len(retain_dataset)}")
    print(f"number of forget dataset {len(forget_dataset)}")
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, test=test_loader)

    criterion = nn.CrossEntropyLoss()
    evaluation_result = None

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        checkpoint = torch.load(args.model_path, map_location=device)
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]

        if args.unlearn != "retrain":
            model.load_state_dict(checkpoint, strict=False)

        unlearn_method = unlearn.get_unlearn_method(args.unlearn)
        unlearn_method(unlearn_data_loaders, model, criterion, args)
        unlearn.save_unlearn_checkpoint(model, None, args)

    if evaluation_result is None:
        evaluation_result = {}

    if "new_accuracy" not in evaluation_result:
        accuracy = {}
        for name, loader in unlearn_data_loaders.items():
            utils.dataset_convert_to_test(loader.dataset, args)
            val_acc = validate(loader, model, criterion, args)
            accuracy[name] = val_acc
            print(f"{name} acc: {val_acc}")
        evaluation_result["accuracy"] = accuracy
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    for deprecated in ["MIA", "SVC_MIA", "SVC_MIA_forget"]:
        if deprecated in evaluation_result:
            evaluation_result.pop(deprecated)

    if "SVC_MIA_forget_efficacy" not in evaluation_result:
        test_len = len(test_loader.dataset)
        forget_len = len(forget_dataset)
        retain_len = len(retain_dataset)

        utils.dataset_convert_to_test(retain_dataset, args)
        utils.dataset_convert_to_test(forget_loader, args)
        utils.dataset_convert_to_test(test_loader, args)

        shadow_train = torch.utils.data.Subset(retain_dataset, list(range(test_len)))
        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=args.batch_size, shuffle=False
        )

        evaluation_result["SVC_MIA_forget_efficacy"] = evaluation.SVC_MIA(
            shadow_train=shadow_train_loader,
            shadow_test=test_loader,
            target_train=None,
            target_test=forget_loader,
            model=model,
        )
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    if "UA" not in evaluation_result:
        evaluation_result["UA"] = 100.0 - evaluation_result["accuracy"]["forget"]
    
    if 'MIA' not in evaluation_result:
        evaluation_result['MIA'] = membership_inference_attack(model, unlearn_data_loaders, seed)
    
    unlearn.save_unlearn_checkpoint(model, evaluation_result, args)
    print(args.unlearn)
    print(args.num_indexes_to_replace)
    print(args.dataset)
    print(evaluation_result)


if __name__ == "__main__":
    main()