import os


def load_dataset(path, fold=None, phase=None, valid=False):
    subjects_dict = {}
    image_pair_dict = {'full': [], 'quarter': []}
    subjects = ['L067', 'L096', 'L109', 'L143', 'L192', 'L286', 'L291', 'L310', 'L333', 'L506']

    if type(fold) == str:
        fold = int(fold)

    if valid:
        if phase == "train":
            del subjects[fold-1]
        elif phase == "valid":
            subjects = [subjects[fold-2]]
        elif phase == "test":
            subjects = [subjects[fold-1]]
        else:
            pass
    elif not valid:
        if phase == "train":
            del subjects[fold-1]
        elif phase == "valid" or phase == "test":
            subjects = [subjects[fold-1]]
        else:
            pass

    for subject in subjects:
        subjects_dict[subject] = os.path.join(path, subject)

    for key in subjects_dict.keys():
        inputs, targets = [], []
        full = os.path.join(subjects_dict[key], 'full_3mm')
        quarter = os.path.join(subjects_dict[key], 'quarter_3mm')

        for image in sorted(os.listdir(quarter)):
            inputs.append(os.path.join(quarter, image))

        for image in sorted(os.listdir(full)):
            targets.append(os.path.join(full, image))

        assert len(targets) == len(inputs)

        for i in range(len(inputs)):
            image_pair_dict['quarter'].append(inputs[i])
            image_pair_dict['full'].append(targets[i])

    return image_pair_dict


def main():
    for i in range(1, 11):
        trainset = load_dataset(path='../../Datasets/AAPM', fold=i, phase="train")
        validset = load_dataset(path='../../Datasets/AAPM', fold=i, phase="valid")
        testset = load_dataset(path='../../Datasets/AAPM', fold=i, phase="test")

        print("Fold", i, len(trainset["full"]), len(validset["full"]), len(testset["full"]),
              len(trainset["full"]) + len(validset["full"]) + len(testset["full"]))


if __name__ == "__main__":
    main()
