import os
import shutil
import logging
from speechbrain.dataio.dataio import read_audio
from speechbrain.utils.data_utils import download_file
import glob
import json
import random
from tqdm import tqdm

logger = logging.getLogger(__name__)


def prepare_aishell(data_folder, save_folder, skip_prep=False):
    """
    This function prepares the AISHELL-3 dataset.
    If the folder does not exist, the zip file will be extracted. If the zip file does not exist, it will be downloaded.

    data_folder : path to AISHELL-3 dataset.
    save_folder: path where to store the manifest csv files.
    skip_prep: If True, skip data preparation.

    """
    if skip_prep:
        return

    # If the data folders do not exist, we need to extract the data
    if not os.path.isdir(os.path.join(data_folder, "wavs")):
        # Check for zip file and download if it doesn't exist
        zip_location = os.path.join(data_folder, "data_aishell3.tgz")
        if not os.path.exists(zip_location):
            url = "https://www.openslr.org/resources/93/data_aishell3.tgz"
            download_file(url, zip_location, unpack=True)
        logger.info("Extracting data_aishell.tgz...")
        shutil.unpack_archive(zip_location, data_folder)
        # wav_dir = os.path.join(data_folder, "data_aishell3/wav")
        # tgz_list = glob.glob(wav_dir + "/*.tar.gz")
        # for tgz in tgz_list:
        #     shutil.unpack_archive(tgz, wav_dir)
        #     os.remove(tgz)

    # Create filename maps for creating json label file.
    valid_wavs = {}
    wav_split = {
        "train": set(),
        "test": set(),
    }

    # Store transcripts from the two content.txt
    for split in ["train", "test"]:
        with open(
            os.path.join(
                data_folder, split, "content.txt"
            ),
            "r",
        ) as f:
            lines = f.readlines()
            for line in lines:
                key = line.split()[0].split(".")[0]
                characters = line.split()[1::2]
                if has_erhua(characters):
                    continue
                value = " ".join(characters)
                valid_wavs[key] = value
                wav_split[split].add(key)



    # Split the dataset into train, valid, and test
    logger.info("Splitting the dataset...")
    all_wavs = glob.glob(
        os.path.join(data_folder)
        + "/wavs/*/*.wav"
    )
    data_split = split_train_test(all_wavs, wav_split)
    data_split.update(split_train_valid(data_split["train"], [80, 20]))

    # Read other labels from corresponding json files
    for split in data_split.keys():
        new_filename = os.path.join(save_folder, split + ".json")
        if not os.path.exists(new_filename):
            logger.info("Preparing %s..." % new_filename)
            json_dict = {}
            invalid_list = []
            unmatched_list = []

            current_wavs = data_split[split]
            for i in tqdm(range(len(current_wavs))):
                filename = current_wavs[i].split("/")[-1].split(".wav")[0]
                if filename not in valid_wavs:
                    continue
                signal = read_audio(current_wavs[i])
                duration = signal.shape[0] / 16000
                # duration = signal.shape[0] / 44100
                transcript = valid_wavs[filename]

                # Read character-level onset/offset from corresponding json files
                json_file = os.path.join(
                    data_folder, "wavs", filename[:7], filename + ".json"
                )
                with open(json_file, "r") as f:
                    json_data = json.load(f)
                    # Validate the json file
                    if json_data is None:
                        invalid_list.append(json_file + "\n")
                        continue
                # Get the onset and offset of each character
                onsets = list(map(lambda x: x["start_time"], json_data))
                offsets = list(map(lambda x: x["end_time"], json_data))
                # Check if the onset and offset are valid
                if len(onsets) != len(transcript.split()) or len(offsets) != len(transcript.split()) or len(onsets) != len(offsets):
                    unmatched_list.append(json_file + "\n")
                    continue
                # Add into the entry
                json_dict[filename] = {
                        "wav": current_wavs[i],
                        "duration": duration,
                        "transcript": transcript,
                        "onsets": onsets,
                        "offsets": offsets,
                }

            # Write the json file
            with open(new_filename, "w", encoding="utf8") as f:
                json.dump(json_dict, f, indent=4, ensure_ascii=False)

            msg = "\t%s successfully created!" % (new_filename)
            logger.info(msg)

            # Write the invalid and unmatched list
            with open(os.path.join(save_folder, split + "_invalid.txt"), "w") as f:
                f.writelines(invalid_list)
            with open(os.path.join(save_folder, split + "_unmatched.txt"), "w") as f:
                f.writelines(unmatched_list)


def has_erhua(characters):
    """
    Check if the character tokens contain erhua (two characters together).
    """
    for token in characters:
        if len(token) > 1:
            return True
    return False

def split_train_test(wav_list, valid_wavs):
    """
    Split the dataset into train and valid according to the train/valid dict.
    """
    splits = ["train", "test"]
    data_split = {
        "train": [],
        "test": [],
    }
    for wav in wav_list:
        filename = wav.split("/")[-1].split(".wav")[0]
        for split in splits:
            if filename in valid_wavs[split]:
                data_split[split].append(wav)
    return data_split

# Dataset split from libritts_prepare
def split_train_valid(train_list, split_ratio):
    """Randomly splits the training wav list into training and validation.

    Arguments
    ---------
    wav_list : list
        list of all the signals in the dataset
    split_ratio: list
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. For instance split_ratio=[80, 20] will
        assign 80% of the sentences to training, 20% for validation, and 10%.

    Returns
    ------
    dictionary containing train, valid, and test splits.
    """
    # Random shuffles the list
    random.shuffle(train_list)
    total_split = sum(split_ratio)
    total_wavs = len(train_list)
    data_split = {}
    splits = ["train", "valid"]

    for i, split in enumerate(splits):
        n_wavs = int(total_wavs * split_ratio[i] / total_split)
        data_split[split] = train_list[0:n_wavs]
        del train_list[0:n_wavs]

    return data_split