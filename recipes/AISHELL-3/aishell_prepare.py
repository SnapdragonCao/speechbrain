import os
import shutil
import logging
from speechbrain.dataio.dataio import read_audio
from speechbrain.utils.data_utils import download_file
import glob
import csv

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
    if not os.path.isdir(os.path.join(data_folder, "train/wav")) or not os.path.isdir(os.path.join(data_folder, "test/wav")):
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

    splits = [
        "train",
        "test",
    ]

    # Create filename-to-transcript and file-to-pinyin dictionary
    filename2transcript = {}
    filename2pinyin = {}

    for split in splits:
        with open(
            os.path.join(
                data_folder, split, "content.txt"
            ),
            "r",
        ) as f:
            lines = f.readlines()
            for line in lines:
                key = line.split()[0].split(".")[0]
                value = " ".join(line.split()[1::2])
                filename2transcript[key] = value

    ID_start = 0  # needed to have a unique ID for each audio
    for split in splits:
        new_filename = os.path.join(save_folder, split) + ".csv"
        if os.path.exists(new_filename):
            continue
        logger.info("Preparing %s..." % new_filename)

        csv_output = [["ID", "duration", "wav", "transcript"]]
        entry = []

        all_wavs = glob.glob(
            os.path.join(data_folder, split)
            + "/wav/*/*.wav"
        )
        for i in range(len(all_wavs)):
            filename = all_wavs[i].split("/")[-1].split(".wav")[0]
            if filename not in filename2transcript:
                continue
            signal = read_audio(all_wavs[i])
            # duration = signal.shape[0] / 16000
            duration = signal.shape[0] / 44100
            transcript_ = filename2transcript[filename]
            csv_line = [
                ID_start + i,
                str(duration),
                all_wavs[i],
                transcript_,
            ]
            entry.append(csv_line)

        csv_output = csv_output + entry

        with open(new_filename, mode="w") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            for line in csv_output:
                csv_writer.writerow(line)

        msg = "\t%s successfully created!" % (new_filename)
        logger.info(msg)

        ID_start += len(all_wavs)

    # Create csv file for validation set from training set
    new_filename = os.path.join(save_folder, "valid.csv")
    if os.path.exists(new_filename):
        return
    logger.info("Preparing %s..." % new_filename)
    # Read the training csv file and split it into train and valid
    train_csv = os.path.join(save_folder, "train.csv")
    with open(train_csv, "r") as f:
        reader = csv.reader(f)
        lines = list(reader)
    # Split the training data into train and valid
    train_lines = lines[: int(0.7 * len(lines))]
    valid_lines = lines[int(0.7 * len(lines)) :]
    # Add the header to the valid csv file
    csv_header = [["ID", "duration", "wav", "transcript"]]
    valid_lines = csv_header + valid_lines
    # Write the valid csv file
    with open(new_filename, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in valid_lines:
            csv_writer.writerow(line)
    msg = "\t%s successfully created!" % (new_filename)
    logger.info(msg)
    # Write the train csv file
    with open(train_csv, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in train_lines:
            csv_writer.writerow(line)
    msg = "\t%s successfully splitted !" % (train_csv)
    logger.info(msg)