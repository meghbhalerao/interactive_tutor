import random

import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

from src.domains.cldc.sterile_field import rule_to_idx
from src.utils.utils import oh_encode_categories

from transformers import AutoProcessor, AutoModelForImageTextToText
import torch


class PassiveClassifier(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_classes, bias=False)

    def forward(self, features: torch.Tensor):
        return self.linear(features)


class PassiveClassifierVLM(nn.Module):
    def __init__(
        self,
        model_id="allenai/Molmo2-8B",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
        video_path=None,
        prompt_path=None,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        max_new_tokens=2048,
    ):
        super().__init__()

        # assert video_path is not None, "Video path must not be None!"
        # assert os.path.exists(video_path), f"Video path must exist, path {video_path} is not found!"

        self.model_id = model_id
        self.video_path = video_path
        self.prompt_path = prompt_path
        self.prompt = None


        if prompt_path is not None:
            assert os.path.exists(
                prompt_path
            ), f"Prompt path must exist, path {prompt_path} is not found!"
            with open(prompt_path, "r") as f:
                self.prompt = f.read().strip()
        self.max_new_tokens = max_new_tokens

        self.template_kwargs = dict(
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            return_tensors=return_tensors,
            return_dict=return_dict,
        )

        # load the processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            device_map=device_map,
        )

        # load the model
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            device_map=device_map,
        )


    def forward(self, prompt: str = None, video_path: str = None, image_list: list = None):
        # priority: explicit prompt > prompt loaded from prompt_path > default

        if prompt is None:
            prompt = (
                self.prompt
                if self.prompt is not None
                else ""
            )

        video_path = video_path if video_path is not None else self.video_path

        # build the content: text, then optional images, then optional video.
        # each image can be a file path, URL, or PIL.Image
        content = [dict(type="text", text=prompt)]
        for img in image_list or []:
            content.append(dict(type="image", image=img))
        if video_path is not None:
            content.append(dict(type="video", video=video_path))

        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(messages, **self.template_kwargs)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # generate output
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )

        # only get generated tokens; decode them to text
        generated_tokens = generated_ids[0, inputs["input_ids"].size(1) :]
        return self.processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )


error_types = ["mistake", "slip", "lapse"]
idx_to_rule = {1: "hbt", 2: "ra", 3: "tb", 4: "oib"}


class Human:
    def __init__(self, rule_error_profile):
        self.rule_error_profile = rule_error_profile


def passive_categorization(passive_clf, feat_vec, context_frames = None):
    device = next(passive_clf.parameters()).device
    error_feat = torch.as_tensor(feat_vec, dtype=torch.float32, device=device)
    return passive_clf(error_feat)


def error_accuracy(
    passive_clf, human_profile, error_report, timestamps, total_duration, batch_size=64
):
    """Fraction of (violation, phase, intervention) entries at `timestamps` whose
    predicted error type (argmax) matches the human's true error type (argmax)."""
    dataset = QueriedErrorDataset(
        error_report, human_profile, timestamps, total_duration
    )
    if len(dataset) == 0:
        return float("nan")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = next(passive_clf.parameters()).device

    correct = 0
    with torch.no_grad():
        for feats, labels in loader:
            pred = passive_clf(feats.to(device))
            correct += (
                (pred.argmax(dim=1) == labels.to(device).argmax(dim=1)).sum().item()
            )
    return correct / len(dataset)


class QueriedErrorDataset(Dataset):
    """One sample per (violation, phase, intervention) entry at the given timestamps.
    Features: one-hot rule + [timestamp, phase, intervention]; label: human's true error profile.
    """

    def __init__(self, error_report, human_profile, timestamps, total_duration):
        self.feats, self.labels = [], []
        for tstamp in timestamps:
            for violation, phase, intervention in error_report[tstamp].keys():
                rule_idx = (
                    rule_to_idx[violation] - 1
                )  # rule indices are 1-based (0 = no human)
                feat_vec = np.concatenate(
                    [
                        oh_encode_categories(rule_idx, len(rule_to_idx)),
                        np.atleast_1d(tstamp) / total_duration,
                        np.atleast_1d(phase),
                        np.atleast_1d(intervention),
                    ]
                ).astype(np.float32)
                self.feats.append(torch.as_tensor(feat_vec, dtype=torch.float32))
                self.labels.append(
                    torch.as_tensor(human_profile[violation], dtype=torch.float32)
                )

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        return self.feats[i], self.labels[i]


def active_categorization(
    passive_clf,
    human_profile,
    error_report,
    total_duration,
    query_mode="random",
    num_queries=5,
    batch_size=8,
    num_epochs=1,
):

    if query_mode == "random":
        violating_timestamps = [t for t, v in error_report.items() if v]
        tqueries = random.sample(
            violating_timestamps, min(num_queries, len(violating_timestamps))
        )
    else:
        raise ValueError(f"Query mode {query_mode} not yet implemented!")

    queried = set(tqueries)
    other_timestamps = [t for t in error_report if t not in queried]
    optimizer = torch.optim.Adam(passive_clf.parameters(), lr=1e-3)

    # compute error prediction accuracy before training (on held-out timestamps)
    acc_before = error_accuracy(
        passive_clf, human_profile, error_report, other_timestamps, total_duration
    )

    print(f"accuracy before training: {acc_before:.3f}")

    # dataloader over the queried timestamps
    dataset = QueriedErrorDataset(error_report, human_profile, tqueries, total_duration)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    device = next(passive_clf.parameters()).device
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for feats, labels in loader:
            pred_error_label = passive_clf(feats.to(device))
            loss = F.cross_entropy(pred_error_label, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * feats.size(0)
        print(f"epoch {epoch + 1}/{num_epochs} loss: {epoch_loss / len(dataset)}")

    # compute error prediction accuracy after training (on held-out timestamps)
    acc_after = error_accuracy(
        passive_clf, human_profile, error_report, other_timestamps, total_duration
    )
    print(f"accuracy after training: {acc_after:.3f}")
    return acc_before, acc_after


def main():
    x = PassiveClassifierVLM()
    out = x.forward(
        prompt="Describe the video in one sentence only.",
        video_path="/home/mb230/projects/interactive_tutor/results/cldc/example_human1/realtime_raw.mp4",
    )
    print(out)


if __name__ == "__main__":
    main()
