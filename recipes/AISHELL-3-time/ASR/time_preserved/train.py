#!/usr/bin/env/python3
"""AISHELL-1 CTC recipe.
The system employs a wav2vec2 encoder and a CTC decoder.
Decoding is performed with greedy decoding.

To run this recipe, do the following:
> python train.py hparams/train.yaml

With the default hyperparameters, the system employs a pretrained wav2vec2 encoder.
The wav2vec2 model is pretrained on 10k hours Chinese data
The network is trained with CTC on characters extracted from a pretrained tokenizer.

Authors
 * Yingzhi WANG 2022
"""

import sys
import torch
from torch.utils.data import DataLoader
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig # wav_lens: normalized length of the waveform in the batch, 1.000 for full length
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "SpeedPerturb"):
                wavs = self.hparams.SpeedPerturb(wavs, wav_lens)

            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])

        # Forward pass
        feats = self.modules.wav2vec2(wavs, wav_lens)


        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "SpecAugment"):
                feats = self.hparams.SpecAugment(feats)

        x = self.modules.enc(feats)
        # print(x.size())
        logits = self.modules.ctc_lin(x)
        # print(logits.size())
        # CELoss expects unnormalized logits
        # p_ctc = self.hparams.log_softmax(logits)
        # print(p_ctc.size())

        return logits, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the frame-level cross entropy loss given predictions and targets."""
        logits, wav_lens = predictions
        ids = batch.id

        # Construct frame-level token labels
        token_labels = self.get_frame_level_token(predictions, batch)

        # Compute frame-level CE loss
        token_labels = token_labels.to(self.device)
        loss = self.hparams.ce_loss(logits.transpose(1, 2), token_labels)

        if stage != sb.Stage.TRAIN:
            predicted_words_list = self.decode(predictions)
            target_words_list = self.get_frame_level_character(predictions, batch)

            self.cer_metric.append(
                ids=ids, predict=predicted_words_list, target=target_words_list,
            )

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()

        if self.check_gradients(loss):
            if not self.hparams.wav2vec2.freeze:
                self.wav2vec_optimizer.step()
            self.model_optimizer.step()

        if not self.hparams.wav2vec2.freeze:
            self.wav2vec_optimizer.zero_grad()
        self.model_optimizer.zero_grad()

        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        self.batch_idx = 0
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(
                stage_stats["loss"]
            )
            old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.model_optimizer, new_lr_model
            )
            if not self.hparams.wav2vec2.freeze:
                sb.nnet.schedulers.update_learning_rate(
                    self.wav2vec_optimizer, new_lr_wav2vec
                )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": old_lr_model,
                    "lr_wav2vec": old_lr_wav2vec,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"CER": stage_stats["CER"]}, min_keys=["CER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.cer_file, "w") as w:
                self.cer_metric.write_stats(w)

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"

        # If the wav2vec encoder is unfrozen, we create the optimizer
        if not self.hparams.wav2vec2.freeze:
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
                self.modules.wav2vec2.parameters()
            )
            if self.checkpointer is not None:
                self.checkpointer.add_recoverable(
                    "wav2vec_opt", self.wav2vec_optimizer
                )

        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)

    def zero_grad(self, set_to_none=False):
        if not self.hparams.wav2vec2.freeze:
            self.wav2vec_optimizer.zero_grad(set_to_none)
        self.model_optimizer.zero_grad(set_to_none)

    def get_frame_level_token(self, predictions, batch):
        """Computes the frame-level token labels given predictions and targets."""
        logits, wav_lens = predictions
        tokens, tokens_lens = batch.tokens
        onsets = batch.onsets
        offsets = batch.offsets
        durations = batch.durations
        batch_size = batch.batchsize

        # Construct frame-level token labels
        full_frames = logits.size(1)
        token_labels = torch.zeros(batch_size, full_frames, dtype=torch.long)
        full_tokens = tokens.size(1)
        for i in range(batch_size):
            n_frames = int(full_frames * wav_lens[i])
            frame_level_labels = torch.zeros(full_frames, dtype=torch.long)
            frame_length = durations[i] / n_frames

            n_tokens = int(full_tokens * tokens_lens[i]) # number of tokens in the sentence
            for j in range(n_tokens):
                if j == 0 or j == n_tokens - 1: # [CLS] and [SEP]
                    continue
                else:
                    start = int(onsets[i][j - 1] / frame_length)
                    end = int(offsets[i][j - 1] / frame_length)
                    frame_level_labels[start:end] = tokens[i][j]

            token_labels[i] = frame_level_labels
        return token_labels
    
    def decode(self, predictions):
        """Decode token terms to words."""
        logits, wav_lens = predictions
        full_frames = logits.size(1)

        # Choose the token with the highest value for each frame
        sequences = []
        for seq, seq_len in zip(logits, wav_lens):
            seq = seq[:int(full_frames * seq_len)]
            _, max_index = torch.max(seq, dim=1)
            sequences.append(max_index.tolist())
        predicted_words_list = []

        for sequence in sequences:
            # Decode token terms to words
            predicted_words = self.tokenizer.convert_ids_to_tokens(
                sequence
            )

            predicted_words_list.append(predicted_words)
        return predicted_words_list
    
    def get_frame_level_character(self, predictions, batch):
        """Computes the frame-level character labels given predictions and targets."""
        logits, wav_lens = predictions
        wrd = batch.wrd
        onsets = batch.onsets
        offsets = batch.offsets
        durations = batch.durations
        batch_size = batch.batchsize

        full_frames = logits.size(1)
        # Construct frame-level character labels
        target_words_list = []
        wrd = batch.wrd
        
        for i in range(batch_size):
            target_words = ["<eps>"] * full_frames
            n_frames = int(full_frames * wav_lens[i])
            frame_length = durations[i] / n_frames
            for j, word in enumerate(wrd[i]):
                start = int(onsets[i][j] / frame_length)
                end = int(offsets[i][j] / frame_length)
                target_words[start:end] = [word] * (end - start)
                
                if j == 0:
                    target_words[:start] = ["[PAD]"] * start
                if j == len(wrd[i]) - 1:
                    target_words[end:] = ["[PAD]"] * (full_frames - end)
            target_words_list.append(target_words)
        return target_words_list
    
    def transcribe(self, inference_set, max_key=None, min_key=None, loader_kwargs={}):
        """Transcribe the whole dataset."""
        if not (
            isinstance(inference_set, DataLoader)
            or isinstance(inference_set, sb.dataio.dataloader.LoopedLoader)
        ):
            loader_kwargs["ckpt_prefix"] = None
            inference_set = self.make_dataloader(
                inference_set, stage=sb.Stage.TEST, **loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        self.modules.eval()

        with torch.no_grad():
            transcripts = []
            for batch in tqdm(inference_set, dynamic_ncols=True):
                predictions = self.compute_forward(batch, sb.Stage.TEST)
                predicted_words_list = self.decode(predictions)
                transcripts.append(predicted_words_list)
        
        with open(self.hparams.transcripts_file, "w") as w:
            w.write(" ".join(transcripts[0][0]) + "\n")
        return transcripts
                
    
    # def transcribe_file(self, path, min_key, loader_kwargs):
    #     sig = sb.dataio.dataio.read_audio(path)
    #     sig = sig.to(self.device)

    #     # Fake a batch
    #     batch = sig.unsqueeze(0)
    #     rel_length = torch.tensor([1.0])


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["train_data"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["valid_data"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_data"], replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    inference_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["inference_data"], replacements={"data_root": data_folder},
    )
    inference_data = inference_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data, inference_data]

    # Defining tokenizer and loading it
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("transcript")
    @sb.utils.data_pipeline.provides("wrd", "tokens_list", "tokens")
    def text_pipeline(wrd):
        wrd = "".join(wrd.split(" "))
        yield wrd
        tokens_list = tokenizer(wrd)["input_ids"]
        yield tokens_list
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # Define onset/offset pipeline:
    @sb.utils.data_pipeline.takes("onsets", "offsets")
    @sb.utils.data_pipeline.provides("onsets", "offsets")
    def onsets_offsets_pipeline(onsets, offsets):
        onsets = torch.Tensor(onsets)
        yield onsets
        offsets = torch.Tensor(offsets)
        yield offsets
    
    sb.dataio.dataset.add_dynamic_item(datasets, onsets_offsets_pipeline)

    # Use duration to infer frame length
    @sb.utils.data_pipeline.takes("duration")
    @sb.utils.data_pipeline.provides("durations")
    def duration_pipeline(duration):
        yield duration
    
    sb.dataio.dataset.add_dynamic_item(datasets, duration_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "tokens", "tokens_list", "onsets", "offsets", "durations"],
    )

    # 5. If Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams = hparams["dynamic_batch_sampler"]
        num_buckets = dynamic_hparams["num_buckets"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            dynamic_hparams["max_batch_len"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        )

        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            dynamic_hparams["max_batch_len"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        )

    return (
        train_data,
        valid_data,
        test_data,
        tokenizer,
        train_batch_sampler,
        valid_batch_sampler,
        inference_data
    )

if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # 1.  # Dataset prep (parsing Librispeech)
    from aishell_prepare import prepare_aishell  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_aishell,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["output_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    (
        train_data,
        valid_data,
        test_data,
        tokenizer,
        train_bsampler,
        valid_bsampler,
        inference_data
    ) = dataio_prepare(hparams)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    asr_brain.tokenizer = tokenizer

    if not hparams["is_inference"]:
        # Changing the samplers if dynamic batching is activated
        train_dataloader_opts = hparams["train_dataloader_opts"]
        valid_dataloader_opts = hparams["valid_dataloader_opts"]

        if train_bsampler is not None:
            train_dataloader_opts = {
                "batch_sampler": train_bsampler,
                "num_workers": hparams["num_workers"],
            }
        if valid_bsampler is not None:
            valid_dataloader_opts = {"batch_sampler": valid_bsampler}

        # Training
        asr_brain.fit(
            asr_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=train_dataloader_opts,
            valid_loader_kwargs=valid_dataloader_opts,
        )

        # Testing
        asr_brain.evaluate(
            test_data,
            min_key="CER",
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )
    
    else:

        # Inference
        transcipt = asr_brain.transcribe(
            inference_data,
            min_key="CER",
            loader_kwargs=hparams["inference_dataloader_opts"],
        )
