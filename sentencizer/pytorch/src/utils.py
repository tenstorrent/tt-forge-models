# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Consolidated imports and dependencies
import os
import re
import zipfile
import requests
import hashlib
import urllib.parse
from pathlib import Path
from collections import namedtuple
from copy import deepcopy

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# Define regex patterns locally to avoid circular imports
NEWLINE_WHITESPACE_RE = re.compile(r"\n\s*\n")
SPACE_RE = re.compile(r"\s")
PUNCTUATION = re.compile(
    r"""["''\(\)\[\]\{\}<>:\,‒–—―…!\.«»\-‐\?''"";/⁄␠·&@\*\\•\^¤¢\$€£¥₩₪†‡°¡¿¬\#№%‰‱¶′§~¨_\|¦⁂☞∴‽※"]"""
)


def word_lens_to_idxs_fast(token_lens):
    max_token_num = max([len(x) for x in token_lens])
    max_token_len = max([max(x) for x in token_lens])
    idxs, masks = [], []
    for seq_token_lens in token_lens:
        seq_idxs, seq_masks = [], []
        offset = 0
        for token_len in seq_token_lens:
            seq_idxs.extend(
                [i + offset for i in range(token_len)]
                + [-1] * (max_token_len - token_len)
            )
            seq_masks.extend(
                [1.0 / token_len] * token_len + [0.0] * (max_token_len - token_len)
            )
            offset += token_len
        seq_idxs.extend([-1] * max_token_len * (max_token_num - len(seq_token_lens)))
        seq_masks.extend([0.0] * max_token_len * (max_token_num - len(seq_token_lens)))
        idxs.append(seq_idxs)
        masks.append(seq_masks)
    return idxs, masks, max_token_num, max_token_len


# Dataset classes
instance_fields = [
    "paragraph_index",
    "wordpieces",
    "wordpiece_labels",
    "wordpiece_ends",
    "piece_idxs",
    "attention_masks",
    "token_type_idxs",
    "wordpiece_num",
]

batch_fields = [
    "paragraph_index",
    "wordpieces",
    "wordpiece_labels",
    "wordpiece_ends",
    "piece_idxs",
    "attention_masks",
    "token_type_idxs",
    "wordpiece_num",
]

Instance = namedtuple("Instance", field_names=instance_fields)
Batch = namedtuple("Batch", field_names=batch_fields)


class TokenizeDatasetLive(Dataset):
    def __init__(self, config, raw_text, max_input_length=512):
        self.config = config
        self.max_input_length = max_input_length
        self.treebank_name = config.treebank_name
        self.raw_text = raw_text
        self.data = []
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def load_data(self):
        self.data = charlevel_format_to_wordpiece_format(
            wordpiece_splitter=self.config.wordpiece_splitter,
            max_input_length=self.max_input_length,
            plaintext=self.raw_text,
            treebank_name=self.config.treebank_name,
        )

    def numberize(self, wordpiece_splitter):
        data = []
        for inst in self.data:
            wordpieces = inst["wordpieces"]
            wordpiece_labels = inst["wordpiece_labels"]
            wordpiece_ends = inst["wordpiece_ends"]
            paragraph_index = inst["paragraph_index"]
            piece_idxs = wordpiece_splitter.encode(
                wordpieces,
                add_special_tokens=True,
                max_length=self.max_input_length,
                truncation=True,
            )
            assert len(piece_idxs) <= self.max_input_length

            pad_num = self.max_input_length - len(piece_idxs)
            attn_masks = [1] * len(piece_idxs) + [0] * pad_num
            piece_idxs = piece_idxs + [0] * pad_num

            token_type_idxs = [
                -100 if piece_id >= len(wordpieces) else wordpiece_labels[piece_id]
                for piece_id in range(len(piece_idxs) - 2)
            ]

            instance = Instance(
                paragraph_index=paragraph_index,
                wordpieces=wordpieces,
                wordpiece_labels=wordpiece_labels,
                wordpiece_ends=wordpiece_ends,
                piece_idxs=piece_idxs,
                attention_masks=attn_masks,
                token_type_idxs=token_type_idxs,
                wordpiece_num=len(wordpieces),
            )
            data.append(instance)
        self.data = data

    def collate_fn(self, batch):
        batch_paragraph_index = []
        batch_wordpieces = []
        batch_wordpiece_labels = []
        batch_wordpiece_ends = []
        batch_piece_idxs = []
        batch_attention_masks = []
        batch_token_type_idxs = []
        batch_wordpiece_num = []

        for inst in batch:
            batch_paragraph_index.append(inst.paragraph_index)
            batch_wordpieces.append(inst.wordpieces)
            batch_wordpiece_labels.append(inst.wordpiece_labels)
            batch_wordpiece_ends.append(inst.wordpiece_ends)
            batch_piece_idxs.append(inst.piece_idxs)
            batch_attention_masks.append(inst.attention_masks)
            batch_token_type_idxs.append(inst.token_type_idxs)
            batch_wordpiece_num.append(inst.wordpiece_num)

        batch_piece_idxs = torch.tensor(
            batch_piece_idxs, dtype=torch.long, device=self.config.device
        )
        batch_attention_masks = torch.tensor(
            batch_attention_masks, dtype=torch.long, device=self.config.device
        )
        batch_token_type_idxs = torch.tensor(
            batch_token_type_idxs, dtype=torch.long, device=self.config.device
        )
        batch_wordpiece_num = torch.tensor(
            batch_wordpiece_num, dtype=torch.long, device=self.config.device
        )

        return Batch(
            paragraph_index=batch_paragraph_index,
            wordpieces=batch_wordpieces,
            wordpiece_labels=batch_wordpiece_labels,
            wordpiece_ends=batch_wordpiece_ends,
            piece_idxs=batch_piece_idxs,
            attention_masks=batch_attention_masks,
            token_type_idxs=batch_token_type_idxs,
            wordpiece_num=batch_wordpiece_num,
        )


# Utility functions
def is_string(input):
    if type(input) == str and len(input.strip()) > 0:
        return True
    return False


def normalize_input(input):
    tmp = input.lstrip()
    lstrip_offset = len(input) - len(input.lstrip())
    return tmp, lstrip_offset


def get_start_char_idx(substring, text):
    start_char_idx = text.index(substring)
    text = text[start_char_idx + len(substring) :]
    return text, start_char_idx


def split_to_substrings(sent_text):
    tokens_by_space = sent_text.split()
    substrings = []
    for token in tokens_by_space:
        if len(PUNCTUATION.findall(token)) > 0:
            tmp = ""
            for char in token:
                if PUNCTUATION.match(char):
                    if tmp != "":
                        substrings.append(tmp)
                        tmp = ""
                    substrings.append(char)
                else:
                    tmp += char
            if tmp != "":
                substrings.append(tmp)
        else:
            substrings.append(token)
    return substrings


def get_startchar(word, text):
    start_char_idx = 0
    for k in range(len(text)):
        if len(text[k].strip()) > 0:
            start_char_idx = k
            break
    text = text[start_char_idx + len(word) :]
    return text, start_char_idx


def get_character_locations(string_units, text):
    tmp_text = deepcopy(text)
    offset = 0
    end_positions = []
    for str_unit in string_units:
        tmp_text, start_position = get_startchar(str_unit, tmp_text)
        start_position += offset
        end_position = start_position + len(str_unit) - 1
        end_positions.append(end_position)
        offset = start_position + len(str_unit)
    return end_positions


def get_mapping_wp_character_to_or_character(
    wordpiece_splitter, wp_single_string, or_single_string
):
    wp_char_to_or_char = {}
    converted_text = ""
    for char_id, char in enumerate(or_single_string):
        converted_chars = "".join(
            [
                c if not c.startswith("▁") else c[1:]
                for c in wordpiece_splitter.tokenize(char)
                if c != "▁"
            ]
        )
        for converted_c in converted_chars:
            c_id = len(converted_text)
            wp_char_to_or_char[c_id] = char_id
            converted_text += converted_c
    return wp_char_to_or_char


def wordpiece_tokenize_from_raw_text(
    wordpiece_splitter,
    sent_text,
    sent_labels,
    sent_position_in_paragraph,
    treebank_name,
):
    if "Chinese" in treebank_name or "Japanese" in treebank_name:
        pseudo_tokens = [c for c in sent_text]  # characters as pseudo tokens
    else:
        if treebank_name == "UD_Urdu-UDTB":
            sent_text = sent_text.replace("۔", ".")
        elif treebank_name == "UD_Uyghur-UDT":
            sent_text = sent_text.replace("-", "،")
        pseudo_tokens = split_to_substrings(sent_text)
    end_pids = set()
    group_pieces = [wordpiece_splitter.tokenize(t) for t in pseudo_tokens]
    flat_wordpieces = []
    for group in group_pieces:
        if len(group) > 0:
            for p in group:
                if p != "▁":
                    pid = len(flat_wordpieces)
                    flat_wordpieces.append((p, pid))
            end_pids.add(len(flat_wordpieces) - 1)

    single_original_string = "".join([c.strip() for c in sent_text])
    original_characters = [c for c in single_original_string]
    character_locations = get_character_locations(original_characters, sent_text)
    single_wordpiece_string = "".join(
        [p if not p.startswith("▁") else p.lstrip("▁") for p, pid in flat_wordpieces]
    )
    wp_character_2_or_character = get_mapping_wp_character_to_or_character(
        wordpiece_splitter, single_wordpiece_string, single_original_string
    )

    flat_wordpiece_labels = []
    flat_wordpiece_ends = []
    offset = 0
    for wordpiece, _ in flat_wordpieces:
        if wordpiece.startswith("▁"):
            str_form = wordpiece[1:]
        else:
            str_form = wordpiece
        end_char = offset + len(str_form) - 1
        ori_char = wp_character_2_or_character[end_char]
        location_in_sentence = character_locations[ori_char]
        wp_label = int(sent_labels[location_in_sentence])
        wp_end = sent_position_in_paragraph + location_in_sentence
        flat_wordpiece_labels.append(wp_label)
        flat_wordpiece_ends.append(wp_end)
        offset = end_char + 1

    return flat_wordpieces, flat_wordpiece_labels, flat_wordpiece_ends, end_pids


def split_to_sentences(paragraph_text, charlabels):
    sent_text = ""
    sent_labels = ""
    sentences = []
    start = 0

    for k in range(len(charlabels)):
        sent_text += paragraph_text[k]
        sent_labels += charlabels[k]

        if charlabels[k] == "2" or charlabels[k] == "4":
            end = k
            sentences.append((deepcopy(sent_text), deepcopy(sent_labels), start, end))
            start = end + 1
            sent_text = ""
            sent_labels = ""

    if len(sentences) > 0:
        if not (len(sent_text) == 0 and len(sent_labels) == 0):
            sentences.append(
                (
                    deepcopy(sent_text),
                    deepcopy(sent_labels),
                    start,
                    len(paragraph_text) - 1,
                )
            )
    else:
        sentences = [(paragraph_text, charlabels, 0, len(paragraph_text) - 1)]
    return sentences


def split_to_subsequences(
    wordpieces, wordpiece_labels, wordpiece_ends, end_piece_ids, max_input_length
):
    subsequences = []
    subseq = [[], [], []]

    for wp_wpid, wl, we in zip(wordpieces, wordpiece_labels, wordpiece_ends):
        wp, wpid = wp_wpid
        subseq[0].append((wp, wpid))
        subseq[1].append(wl)
        subseq[2].append(we)
        if wpid in end_piece_ids and len(subseq[0]) >= max_input_length - 10:
            subsequences.append((subseq[0], subseq[1], subseq[2], end_piece_ids))
            subseq = [[], [], []]

    if len(subseq[0]) > 0:
        subsequences.append((subseq[0], subseq[1], subseq[2], end_piece_ids))
    return subsequences


def charlevel_format_to_wordpiece_format(
    wordpiece_splitter,
    max_input_length,
    plaintext,
    treebank_name,
    char_labels_output_fpath=None,
):
    if char_labels_output_fpath is not None:
        with open(char_labels_output_fpath) as f:
            corpus_labels = "".join(f.readlines()).rstrip()
    else:
        corpus_labels = "\n\n".join(
            ["0" * len(pt.rstrip()) for pt in NEWLINE_WHITESPACE_RE.split(plaintext)]
        )

    data = [
        {"text": pt.rstrip(), "charlabels": pc}
        for pt, pc in zip(
            NEWLINE_WHITESPACE_RE.split(plaintext),
            NEWLINE_WHITESPACE_RE.split(corpus_labels),
        )
        if len(pt.rstrip()) > 0
    ]

    wordpiece_examples = []
    kept_tokens = 0
    total_tokens = 0
    for paragraph_index, paragraph in enumerate(data):
        paragraph_text = paragraph["text"]
        paragraph_labels = paragraph["charlabels"]
        sentences = split_to_sentences(paragraph_text, paragraph_labels)
        tmp_examples = []
        for sent in sentences:
            sent_text, sent_labels, sent_start, sent_end = sent
            (
                wordpieces,
                wordpiece_labels,
                wordpiece_ends,
                end_piece_ids,
            ) = wordpiece_tokenize_from_raw_text(
                wordpiece_splitter, sent_text, sent_labels, sent_start, treebank_name
            )
            kept_tokens += len([x for x in wordpiece_labels if x != 0])
            total_tokens += len([x for x in sent_labels if x != "0"])
            if len(wordpieces) <= max_input_length - 2:
                tmp_examples.append(
                    (wordpieces, wordpiece_labels, wordpiece_ends, end_piece_ids)
                )
            else:
                subsequences = split_to_subsequences(
                    wordpieces,
                    wordpiece_labels,
                    wordpiece_ends,
                    end_piece_ids,
                    max_input_length,
                )
                for subseq in subsequences:
                    tmp_examples.append(subseq)
        new_example = [[], [], []]
        for example in tmp_examples:
            if len(new_example[0]) + len(example[0]) > max_input_length - 2:
                num_extra_wordpieces = min(
                    max_input_length - 2 - len(new_example[0]), len(example[0])
                )
                end_piece_ids = example[-1]
                takeout_position = 0
                for tmp_id in range(num_extra_wordpieces):
                    wp, wpid = example[0][tmp_id]
                    if wpid in end_piece_ids:
                        takeout_position = tmp_id + 1
                num_extra_wordpieces = takeout_position
                new_example[0] += deepcopy(example[0][:num_extra_wordpieces])
                new_example[1] += deepcopy(example[1][:num_extra_wordpieces])
                new_example[2] += deepcopy(example[2][:num_extra_wordpieces])
                wordpiece_examples.append(
                    (
                        [wp for wp, wpid in new_example[0]],
                        new_example[1],
                        new_example[2],
                        paragraph_index,
                    )
                )
                new_example = [[], [], []]

            new_example[0] += deepcopy(example[0])
            new_example[1] += deepcopy(example[1])
            new_example[2] += deepcopy(example[2])
        if len(new_example[0]) > 0:
            wordpiece_examples.append(
                (
                    [wp for wp, wpid in new_example[0]],
                    new_example[1],
                    new_example[2],
                    paragraph_index,
                )
            )

    final_examples = []
    for wp_example in wordpiece_examples:
        wordpieces, wordpiece_labels, wordpiece_ends, paragraph_index = wp_example
        final_examples.append(
            {
                "wordpieces": wordpieces,
                "wordpiece_labels": wordpiece_labels,
                "wordpiece_ends": wordpiece_ends,
                "paragraph_index": paragraph_index,
            }
        )

    return final_examples


def normalize_token(treebank_name, token, ud_eval=True):
    token = SPACE_RE.sub(" ", token.lstrip())
    if ud_eval:
        if (
            "chinese" in treebank_name.lower()
            or "korean" in treebank_name.lower()
            or "japanese" in treebank_name.lower()
        ):
            token = token.replace(" ", "")
    return token


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def unzip(dir, filename):
    with zipfile.ZipFile(os.path.join(dir, filename)) as f:
        f.extractall(dir)
    os.remove(os.path.join(dir, filename))


def download(language, saved_model_version, embedding_name):
    """
    Args:
        language: Language code for the model
        saved_model_version: Version of the saved model
        embedding_name: Name of the embedding model

    Returns:
        str: Path to the cache directory where files were downloaded
    """
    from pathlib import Path
    import shutil

    # Construct the URL
    url = f"https://huggingface.co/uonlp/trankit/resolve/main/models/{saved_model_version}/{embedding_name}/{language}.zip"

    # Use get_file to handle download and cache directory logic automatically
    zip_file_path = get_file(url)

    # Extract cache directory from the downloaded file path
    zip_path = Path(zip_file_path)
    cache_base = zip_path.parent.parent  # Go up from url_cache to main cache dir

    # Create trankit directory structure
    trankit_cache = cache_base / "trankit"
    lang_dir = trankit_cache / embedding_name / language
    downloaded_marker = lang_dir / f"{language}.downloaded"

    # Only process if not already done
    if not downloaded_marker.exists():
        print(f"Setting up {language} models...")

        # Ensure directory exists
        lang_dir.mkdir(parents=True, exist_ok=True)

        # Copy zip file to trankit location
        target_zip = lang_dir / f"{language}.zip"
        shutil.copy2(zip_file_path, target_zip)

        # Unzip
        unzip(str(lang_dir), f"{language}.zip")

        # Mark as completed
        with open(downloaded_marker, "w") as f:
            f.write("")

        print(f"Successfully set up models for {language}")

    return str(trankit_cache)


def get_file(path):
    """Get a file from local filesystem, cache, or URL.

    This function handles both local files and URLs, retrieving from cache when available
    or downloading/fetching as needed. For URLs, it creates a unique cached filename using
    a hash of the URL to prevent collisions.

    Args:
        path: Path to a local file or URL to download

    Returns:
        Path to the file in the cache
    """
    # Check if path is a URL - handle URLs and files differently
    path_is_url = path.startswith(("http://", "https://"))

    if path_is_url:
        # Create a hash from the URL to ensure uniqueness and prevent collisions
        url_hash = hashlib.md5(path.encode()).hexdigest()[:10]

        # Get filename from URL, or create one if not available
        file_name = os.path.basename(urllib.parse.urlparse(path).path)
        if not file_name:
            file_name = f"downloaded_file_{url_hash}"
        else:
            file_name = f"{url_hash}_{file_name}"

        rel_path = Path("url_cache")
        cache_dir_fallback = Path.home() / ".cache/url_cache"
    else:
        rel_dir, file_name = os.path.split(path)
        rel_path = Path("models/tt-ci-models-private") / rel_dir
        cache_dir_fallback = Path.home() / ".cache/lfcache" / rel_dir

    # Determine the base cache directory based on environment variables
    if (
        "DOCKER_CACHE_ROOT" in os.environ
        and Path(os.environ["DOCKER_CACHE_ROOT"]).exists()
    ):
        cache_dir = Path(os.environ["DOCKER_CACHE_ROOT"]) / rel_path
    elif "LOCAL_LF_CACHE" in os.environ:
        cache_dir = Path(os.environ["LOCAL_LF_CACHE"]) / rel_path
    else:
        cache_dir = cache_dir_fallback

    file_path = cache_dir / file_name

    # Support case where shared cache is read only and file not found. Can read files from it, but
    # fall back to home dir cache for storing downloaded files. Common w/ CI cache shared w/ users.
    cache_dir_rdonly = not os.access(cache_dir, os.W_OK)
    if not file_path.exists() and cache_dir_rdonly and cache_dir != cache_dir_fallback:
        print(
            f"Warning: {cache_dir} is read-only, using {cache_dir_fallback} for {path}"
        )
        cache_dir = cache_dir_fallback
        file_path = cache_dir / file_name

    cache_dir.mkdir(parents=True, exist_ok=True)

    # If file is not found in cache, download URL from web, or get file from IRD_LF_CACHE web server.
    if not file_path.exists():
        if path_is_url:
            try:
                print(f"Downloading file from URL {path} to {file_path}")
                response = requests.get(path, stream=True, timeout=(15, 60))
                response.raise_for_status()  # Raise exception for HTTP errors

                with open(file_path, "wb") as f:
                    f.write(response.content)

            except Exception as e:
                raise RuntimeError(f"Failed to download {path}: {str(e)}")
        elif "DOCKER_CACHE_ROOT" in os.environ:
            raise FileNotFoundError(
                f"File {file_path} is not available, check file path. If path is correct, DOCKER_CACHE_ROOT syncs automatically with S3 bucket every hour so please wait for the next sync."
            )
        else:
            if "IRD_LF_CACHE" not in os.environ:
                raise ValueError(
                    "IRD_LF_CACHE environment variable is not set. Please set it to the address of the IRD LF cache."
                )
            print(f"Downloading file from path {path} to {cache_dir}/{file_name}")
            exit_code = os.system(
                f"wget -nH -np -R \"indexg.html*\" -P {cache_dir} {os.environ['IRD_LF_CACHE']}/{path} --connect-timeout=15 --read-timeout=60 --tries=3"
            )
            # Check for wget failure
            if exit_code != 0:
                raise RuntimeError(
                    f"wget failed with exit code {exit_code} when downloading {os.environ['IRD_LF_CACHE']}/{path}"
                )

            # Ensure file_path exists after wget command
            if not file_path.exists():
                raise RuntimeError(
                    f"Download appears to have failed: File {file_name} not found in {cache_dir} after wget command"
                )

    return file_path
