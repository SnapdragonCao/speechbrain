import os
import shutil
import logging
from speechbrain.dataio.dataio import read_audio
from speechbrain.utils.data_utils import download_file
import glob
import json
import random

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


    # Split the dataset into train, valid, and test
    logger.info("Splitting the dataset...")
    all_wavs = glob.glob(
        os.path.join(data_folder)
        + "/wavs/*/*.wav"
    )
    data_split = split_sets(all_wavs, [80, 10, 10])

    # Read other labels from corresponding json files
    for split in data_split.keys():
        new_filename = os.path.join(save_folder, split + ".json")
        if not os.path.exists(new_filename):
            logger.info("Preparing %s..." % new_filename)
            entry = []

            current_wavs = data_split[split]
            for i in range(len(current_wavs)):
                filename = current_wavs[i].split("/")[-1].split(".wav")[0]
                if filename not in valid_wavs:
                    continue
                signal = read_audio(current_wavs[i])
                duration = signal.shape[0] / 16000
                # duration = signal.shape[0] / 44100
                transcript = valid_wavs[filename]

                # Read character-level onset/offset from corresponding json files
                json_file = os.path.join(
                    data_folder, "wavs", filename + ".json"
                )
                with open(json_file, "r") as f:
                    json_data = json.load(f)
                # Get the onset and offset of each character
                onsets = list(map(lambda x: x["start_time"], json_data))
                offsets = list(map(lambda x: x["end_time"], json_data))
                # Add into the entry
                entry.append(
                    {
                        "id": filename,
                        "path": current_wavs[i],
                        "duration": duration,
                        "transcript": transcript,
                        "onsets": onsets,
                        "offsets": offsets,
                    }
                )

        # Write the json file
        with open(new_filename, "w", encoding="utf8") as f:
            json.dump(entry, f, indent=4, ensure_ascii=False)

        msg = "\t%s successfully created!" % (new_filename)
        logger.info(msg)


def has_erhua(characters):
    """
    Check if the character tokens contain erhua (two characters together).
    """
    for token in characters:
        if len(token) > 1:
            return True
    return False

# Dataset split from libritts_prepare
def split_sets(wav_list, split_ratio):
    """Randomly splits the wav list into training, validation, and test lists.

    Arguments
    ---------
    wav_list : list
        list of all the signals in the dataset
    split_ratio: list
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. For instance split_ratio=[80, 10, 10] will
        assign 80% of the sentences to training, 10% for validation, and 10%
        for test.

    Returns
    ------
    dictionary containing train, valid, and test splits.
    """
    # Random shuffles the list
    random.shuffle(wav_list)
    total_split = sum(split_ratio)
    total_wavs = len(wav_list)
    data_split = {}
    splits = ["train", "valid"]

    for i, split in enumerate(splits):
        n_wavs = int(total_wavs * split_ratio[i] / total_split)
        data_split[split] = wav_list[0:n_wavs]
        del wav_list[0:n_wavs]
    data_split["test"] = wav_list

    return data_split