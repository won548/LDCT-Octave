import os
import glob


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
        inputs = sorted(glob.glob(os.path.join(subjects_dict[key], 'quarter_3mm', '*.IMA')))
        targets = sorted(glob.glob(os.path.join(subjects_dict[key], 'full_3mm', '*.IMA')))
        assert len(inputs) == len(targets)

        for i in range(len(inputs)):
            image_pair_dict['quarter'].append(inputs[i])
            image_pair_dict['full'].append(targets[i])

    return image_pair_dict


def main():
    for i in range(1, 11):
        trainset = load_dataset(path='/home/dongkyu/Datasets/AAPM', fold=i, phase="train", valid=True)
        validset = load_dataset(path='/home/dongkyu/Datasets/AAPM', fold=i, phase="valid", valid=True)
        testset = load_dataset(path='/home/dongkyu/Datasets/AAPM', fold=i, phase="test", valid=True)

        print("Fold", i, len(trainset["full"]), len(validset["full"]), len(testset["full"]),
              len(trainset["full"]) + len(validset["full"]) + len(testset["full"]))


if __name__ == "__main__":
    main()
