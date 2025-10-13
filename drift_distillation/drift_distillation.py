device = "cuda"
image_set = "utzap"
text_set = "nemotron"
embedding_mode = "clip"

# image_set = os.getenv("IMAGE_SET", "farfetch")
# text_set = os.getenv("TEXT_SET", "opencharacter")

import os
from google.cloud import aiplatform

aiplatform.init(project="gen-lang-client-0184547990", location="us-central1")


def train_step(
    num_images,
    step_n,
    split_test=0.1,
    lr=1e-5,
    freeze=False,
    batch_size_=50,
    weighting=True,
    epochs=1,
    accumulation_steps=10,
):
    import torch
    import torch.nn as nn
    from transformers import CLIPModel
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    from torch.utils.data import Dataset, DataLoader
    from random import randint
    from torch.optim.lr_scheduler import ExponentialLR
    import pickle
    import glob
    import torch.nn.functional as F

    # %%
    print("loading scores")
    with open(
        f"{image_set}+{text_set}/{embedding_mode}/drift_data/drift_clean_scores{step_n}.pkl",
        "rb",
    ) as f:
        scores = pickle.load(f)
    print("loading personas")
    with open(
        f"{image_set}+{text_set}/{embedding_mode}/drift_data/drift_clean_personas{step_n}.pkl",
        "rb",
    ) as f:
        personas = pickle.load(f)
    print("loading image indices")
    with open(
        f"{image_set}+{text_set}/{embedding_mode}/drift_data/drift_clean_image_indices{step_n}.pkl",
        "rb",
    ) as f:
        image_indices = torch.load(f)
    print("loading embeddings")
    with open(
        f"{image_set}+{text_set}/{embedding_mode}/drift_data/drift_clean_{embedding_mode}_embeddings{step_n}.pkl",
        "rb",
    ) as f:
        embeddings = torch.load(f)

    # %%
    with open(f"../data_preparation/{image_set}/pixel_values_train.pkl", "rb") as f:
        pixel_values = torch.load(f, weights_only=False)

    # %%
    tally = torch.zeros(len(pixel_values))
    for i in range(len(image_indices)):
        for j in range(len(image_indices[i])):
            tally[image_indices[i][j]] += 1

    weights = 1 / (tally + 10).to(device)
    weights = weights / weights[image_indices].mean()
    # %%
    scores = torch.tensor(scores)
    scores = torch.clamp(scores, min=0, max=num_images)

    # %%
    image_indices = torch.tensor(image_indices)

    # %%
    class PersonaImageDataset(Dataset):
        def __init__(self, personas, images, image_indices, scores, transform=None):
            self.personas = personas
            self.images = images
            self.scores = scores
            self.image_indices = image_indices
            self.transform = transform

        def __len__(self):
            return len(self.image_indices)

        def __getitem__(self, idx):
            img = self.images[self.image_indices[idx]]
            persona = self.personas[idx]
            score = self.scores[idx]

            if self.transform:
                img = self.transform(img)

            return (persona, img, self.image_indices[idx]), score

    # %%
    split_val = 0

    n = len(image_indices)

    n_test = int(split_test * n)
    n_val = int(split_val * n)
    n_train = n - n_test - n_val  # ensure all sum to total

    test_image_indices = image_indices[:n_test]
    val_image_indices = image_indices[n_test : n_test + n_val]
    train_image_indices = image_indices[n_test + n_val :]

    test_personas = personas[:n_test]
    val_personas = personas[n_test : n_test + n_val]
    train_personas = personas[n_test + n_val :]

    test_embeddings = embeddings[:n_test]
    val_embeddings = embeddings[n_test : n_test + n_val]
    train_embeddings = embeddings[n_test + n_val :]

    test_scores = scores[:n_test]
    val_scores = scores[n_test : n_test + n_val]
    train_scores = scores[n_test + n_val :]

    # %%
    dataset_test = PersonaImageDataset(
        test_embeddings, pixel_values, test_image_indices, test_scores, transform=None
    )
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size_, shuffle=False)

    # seed = 15
    # g = torch.Generator()
    # g.manual_seed(seed)

    dataset_train = PersonaImageDataset(
        train_embeddings,
        pixel_values,
        train_image_indices,
        train_scores,
        transform=None,
    )
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size_, shuffle=False)

    # %%
    # Example usage
    for input, scores_ in dataloader_test:
        print(input[0].shape)  # string
        print(input[1].shape)
        print(input[2].shape)  # image indices
        print(scores_.shape)
        break

    # %%

    if embedding_mode == "openai":

        class FashionCLIPImageEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_encoder = CLIPModel.from_pretrained(
                    "patrickjohncyh/fashion-clip"
                ).vision_model

                for param in self.vision_encoder.parameters():
                    param.requires_grad = not (freeze)

                self.projection = nn.Linear(768, 1536)

            def forward(self, pixel_values):
                x = self.vision_encoder(pixel_values).pooler_output
                x = self.projection(x)
                return x

    elif embedding_mode == "clip":

        class FashionCLIPImageEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                # Load the full model temporarily to extract projection weights
                full_clip_model = CLIPModel.from_pretrained(
                    "patrickjohncyh/fashion-clip"
                )

                # Extract vision encoder
                self.vision_encoder = full_clip_model.vision_model

                # Extract the visual projection layer weights
                self.projection = nn.Linear(768, 512, bias=False)
                self.projection.weight.data = (
                    full_clip_model.visual_projection.weight.data.clone()
                )

                # Delete the full model to save memory
                del full_clip_model

            def forward(self, pixel_values):
                x = self.vision_encoder(pixel_values).pooler_output
                x = self.projection(x)
                return x

    model = FashionCLIPImageEncoder().to(device)

    # %%
    state_dict = torch.load(
        f"{image_set}+{text_set}/{embedding_mode}/drift_weights/model_weights{step_n-1}.pth",
        map_location=torch.device("cuda"),
    )
    model.load_state_dict(state_dict)

    # %%
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.99)

    # %%
    def test():
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs_, targets_) in enumerate(dataloader_test):
                embeddings_, images_, image_indices_, targets_ = (
                    inputs_[0].to(device),
                    inputs_[1].to(device),
                    inputs_[2].to(device),
                    targets_.to(device),
                )
                gathered_weights = weights[image_indices_]
                weight_matrix = gathered_weights.unsqueeze(
                    2
                ) * gathered_weights.unsqueeze(1)

                images_flat = images_.view(
                    -1, 3, 224, 224
                )  # [batch * num_images, 3, 224, 224]
                outputs_flat = model(images_flat)  # [batch * num_images, D]
                batch_size = images_.shape[0]
                outputs = outputs_flat.view(
                    batch_size, num_images, -1
                )  # [batch, num_images, D]

                sort_indices = targets_ - 1

                for i in range(len(outputs)):
                    outputs[i] = outputs[i][sort_indices[i]]

                # Create upper-triangular mask
                mask = torch.triu(
                    torch.ones(num_images, num_images, device=images_.device),
                    diagonal=1,
                ).bool()  # [num_images, num_images]

                embeddings_norm = F.normalize(embeddings_, p=2, dim=1)
                outputs_norm = F.normalize(outputs, p=2, dim=1)

                model_scores = torch.einsum(
                    "bd,bnd->bn", embeddings_norm.float(), outputs_norm.float()
                )

                batch_diff = []
                for i in range(batch_size):
                    a = model_scores[i].unsqueeze(1)  # [num_images, 1, D]
                    b = model_scores[i].unsqueeze(0)  # [1, num_images, D]
                    diff = a - b  # [num_images, num_images]
                    batch_diff.append(diff)

                diff_tensor = torch.stack(batch_diff)  # [batch, num_images, num_images]
                masked_diff = diff_tensor.masked_fill(~mask, 0)  # mask lower triangle

                # Compute loss
                if weighting:
                    logp = (
                        torch.log(torch.sigmoid(masked_diff) + 1e-8) * weight_matrix
                    ).sum(dim=(1, 2))
                else:
                    logp = (torch.log(torch.sigmoid(masked_diff) + 1e-8)).sum(
                        dim=(1, 2)
                    )

                loss = -logp.mean()
                total_loss += loss.item()

                # Accuracy: count where sigmoid(masked_diff) > 0.5
                with torch.no_grad():
                    probs = torch.sigmoid(
                        masked_diff
                    )  # [batch, num_images, num_images]
                    preds = (probs > 0.5).float()
                    correct += preds.sum().item()
                    total += (
                        mask.sum().item() * batch_size
                    )  # total comparisons per sample

                if batch_idx % 20 == 0:
                    print(f"Test Batch {batch_idx} Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader_test)
        accuracy = correct / total
        print(f"Test Average Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
        return avg_loss, accuracy

    # %%
    train_loss, test_loss = [], []

    for epoch in range(epochs):
        if epoch == 0:
            test_loss.append(test()[0])

        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs_, targets_) in enumerate(dataloader_train):
            embeddings_, images_, image_indices_, targets_ = (
                x.to(device) for x in (*inputs_, targets_)
            )
            gathered_weights = weights[image_indices_]
            weight_matrix = gathered_weights.unsqueeze(2) * gathered_weights.unsqueeze(
                1
            )

            images_flat = images_.view(
                -1, 3, 224, 224
            )  # [batch * num_images, 3, 244, 244]
            outputs_flat = model(images_flat)  # [batch * num_images, D]
            batch_size = images_.shape[0]
            outputs = outputs_flat.view(
                batch_size, num_images, -1
            )  # [batch, num_images, D]

            sort_indices = targets_ - 1
            for i in range(len(outputs)):
                outputs[i] = outputs[i][sort_indices[i]]

            # Create upper-triangular mask
            mask = torch.triu(
                torch.ones(num_images, num_images, device=images_.device), diagonal=1
            ).bool()  # [num_images, num_images]

            embeddings_norm = F.normalize(embeddings_, p=2, dim=1)
            outputs_norm = F.normalize(outputs, p=2, dim=1)

            model_scores = torch.einsum(
                "bd,bnd->bn", embeddings_norm.float(), outputs_norm.float()
            )

            batch_diff = []
            for i in range(batch_size):
                a = model_scores[i].unsqueeze(1)  # [num_images, 1, D]
                b = model_scores[i].unsqueeze(0)  # [1, num_images, D]
                diff = a - b  # [num_images, num_images]
                batch_diff.append(diff)

            diff_tensor = torch.stack(batch_diff)  # [batch, num_images, num_images]
            masked_diff = diff_tensor.masked_fill(~mask, 0)  # mask lower triangle

            # Compute loss
            if weighting:
                logp = (
                    torch.log(torch.sigmoid(masked_diff) + 1e-8) * weight_matrix
                ).sum(dim=(1, 2))
            else:
                logp = (torch.log(torch.sigmoid(masked_diff) + 1e-8)).sum(dim=(1, 2))
            loss = -logp.mean()
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                print(f"Gradient accumulated in batch {batch_idx}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item()
            train_loss.append(loss.item())
            # tb_logger.add_scalar('Loss/train', loss.item(), epoch * len(dataloader_train) + batch_idx)

            # Accuracy: count where sigmoid(masked_diff) > 0.5
            with torch.no_grad():
                probs = torch.sigmoid(masked_diff)  # [batch, num_images, num_images]
                preds = (probs > 0.5).float()
                correct += preds.sum().item()
                total += mask.sum().item() * batch_size  # total comparisons per sample

            if batch_idx % 1 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} Batch {batch_idx} Loss: {loss.item():.4f}, accuracy: {correct / total * 100:.2f}%"
                )

        if epoch % 1 == 0:
            avg_loss, accuracy = test()
            test_loss.append(avg_loss)

        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader_train):.4f}")

    # %%
    plt.plot(np.linspace(0, 1, len(train_loss)), train_loss)
    plt.plot(np.linspace(0, 1, len(test_loss)), test_loss)
    plt.savefig(
        f"{image_set}+{text_set}/{embedding_mode}/loss_figures/loss{step_n}.png"
    )  # You can also use .pdf, .svg, etc.
    plt.close()

    with open(
        f"{image_set}+{text_set}/{embedding_mode}/loss_figures/loss{step_n}.txt",
        "w",
    ) as file:
        file.write(
            f"Test Average Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%"
        )

    # %%
    import torch

    torch.save(
        model.state_dict(),
        f"{image_set}+{text_set}/{embedding_mode}/drift_weights/model_weights{step_n}.pth",
    )


def rank_images(step_n):
    # %%
    import torch
    import torch.nn as nn
    from transformers import CLIPModel
    import matplotlib.pyplot as plt
    import pickle
    from torch.utils.data import Dataset
    from random import randint
    import torch
    import openai
    import numpy as np
    import json

    # %%
    with open(
        f"../data_preparation/{text_set}/{embedding_mode}/train_persona_embeddings.pkl",
        "rb",
    ) as f:
        embeddings = pickle.load(f)
    with open(
        f"../data_preparation/{text_set}/{embedding_mode}/test_persona_embeddings.pkl",
        "rb",
    ) as f:
        test_embeddings = pickle.load(f)
    with open(f"../data_preparation/{image_set}/pixel_values_train.pkl", "rb") as f:
        pixel_values = torch.load(f, weights_only=False)
    with open(
        f"../generate_test_set/{image_set}+{text_set}/test_indices.json",
        "r",
    ) as f:
        test_indices = json.load(f)
    with open(f"../data_preparation/{image_set}/pixel_values_test.pkl", "rb") as f:
        test_pixel_values = torch.load(f, weights_only=False)

    # %%
    if embedding_mode == "openai":

        class FashionCLIPImageEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_encoder = CLIPModel.from_pretrained(
                    "patrickjohncyh/fashion-clip"
                ).vision_model
                self.projection = nn.Linear(768, 1536)

            def forward(self, pixel_values):
                x = self.vision_encoder(pixel_values).pooler_output
                x = self.projection(x)
                return x

    elif embedding_mode == "clip":

        class FashionCLIPImageEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                # Load the full model temporarily to extract projection weights
                full_clip_model = CLIPModel.from_pretrained(
                    "patrickjohncyh/fashion-clip"
                )

                # Extract vision encoder
                self.vision_encoder = full_clip_model.vision_model

                # Extract the visual projection layer weights
                self.projection = nn.Linear(768, 512, bias=False)
                self.projection.weight.data = (
                    full_clip_model.visual_projection.weight.data.clone()
                )

                # Delete the full model to save memory
                del full_clip_model

            def forward(self, pixel_values):
                x = self.vision_encoder(pixel_values).pooler_output
                x = self.projection(x)
                return x

    model = FashionCLIPImageEncoder().to(device)

    # %%
    # state_dict = torch.load("model_weights_fashionclip1.pth", map_location=torch.device("cuda"))
    # model.load_state_dict(state_dict)
    state_dict = torch.load(
        f"{image_set}+{text_set}/{embedding_mode}/drift_weights/model_weights{step_n}.pth",
        map_location=torch.device("cuda"),
    )
    model.load_state_dict(state_dict)

    # %%
    model.eval()

    # %%
    import torch

    def run_model_in_batches(model, pixel_values, batch_size=32, device="cuda"):
        model.eval()  # Set model to evaluation mode
        results = []

        with torch.no_grad():  # No gradients needed for inference
            for i in range(0, len(pixel_values), batch_size):
                batch = pixel_values[i : i + batch_size].to(device)
                output = model(batch)
                results.append(output)

        # Concatenate outputs if they are tensors
        if isinstance(results[0], torch.Tensor):
            results = torch.cat(results, dim=0)

        return results

    # %%
    results = run_model_in_batches(model, pixel_values, batch_size=32, device=device)
    test_results = run_model_in_batches(
        model, test_pixel_values, batch_size=32, device=device
    )

    # %%
    import torch.nn.functional as F

    # Move everything to CPU and convert to float
    results = results.to("cpu").float()
    test_results = test_results.to("cpu").float()

    embeddings = embeddings.to("cpu").float()
    test_embeddings = test_embeddings.to("cpu").float()

    # Normalize the embeddings along the feature dimension (dim=1)
    results_norm = F.normalize(results, p=2, dim=1)
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)

    test_results_norm = F.normalize(test_results, p=2, dim=1)
    test_embeddings_norm = F.normalize(test_embeddings, p=2, dim=1)

    # Compute cosine similarity (dot product of normalized vectors)
    cosine_sim = torch.matmul(results_norm, embeddings_norm.T)
    test_cosine_sim = torch.matmul(test_results_norm, test_embeddings_norm.T)

    sorted_values, sorted_indices = torch.sort(cosine_sim, dim=0)
    test_sorted_values, test_sorted_indices = torch.sort(test_cosine_sim, dim=0)

    test_indices = torch.tensor(test_indices)

    with open(
        f"{image_set}+{text_set}/{embedding_mode}/drift_sorted_indices/sorted_indices{step_n}.pkl",
        "wb",
    ) as f:
        pickle.dump(sorted_indices, f)
    with open(
        f"{image_set}+{text_set}/{embedding_mode}/drift_sorted_indices/results{step_n}.pkl",
        "wb",
    ) as f:
        pickle.dump(results, f)
    with open(
        f"{image_set}+{text_set}/{embedding_mode}/drift_sorted_indices/dot_products{step_n}.pkl",
        "wb",
    ) as f:
        pickle.dump(cosine_sim, f)

    L = []
    for i in range(len(test_indices)):
        test_sorted_indices_column = test_sorted_indices[:, i]
        n = test_sorted_indices_column.tolist().index(test_indices[i])
        L.append(n / len(test_sorted_indices_column))

    filename = f"{image_set}+{text_set}/{embedding_mode}/test_scores.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            scores = json.load(f)
    else:
        scores = []

    # Append new score
    scores.append({"median": np.median(L), "mean": np.mean(L), "step_n": step_n})

    # Write back
    with open(filename, "w") as f:
        json.dump(scores, f, indent=2)


def data_preparation_cycle(step_n, num_images):
    # %%
    from tqdm import tqdm
    from joblib import Parallel, delayed
    from tqdm_joblib import tqdm_joblib
    import pickle
    from random import randint
    import matplotlib.pyplot as plt
    import math
    import torch
    import vertexai
    from vertexai.generative_models import GenerativeModel
    import tempfile
    import os
    from vertexai.generative_models import Image as VertexImage
    import json

    # %%
    # with open('sorted_indices_fashionclip1.pkl', 'rb') as f:
    #     sorted_indices = pickle.load(f)
    with open(
        f"{image_set}+{text_set}/{embedding_mode}/drift_sorted_indices/dot_products{step_n}.pkl",
        "rb",
    ) as f:
        dot_products = pickle.load(f)
    # %%
    with open(f"../data_preparation/{image_set}/images_train.pkl", "rb") as f:
        images = pickle.load(f)
    with open(f"../data_preparation/{text_set}/train_personas.json", "r") as f:
        personas = json.load(f)
    with open(
        f"../data_preparation/{text_set}/{embedding_mode}/train_persona_embeddings.pkl",
        "rb",
    ) as f:
        embeddings = pickle.load(f)

    # %%
    import torch
    import random
    import numpy as np
    import torch.nn.functional as F

    scores = dot_products.T.cpu().numpy()

    def sample_indices(scores, batch_size=5):
        indices = list(range(len(scores)))
        a = min(scores)
        b = max(scores)
        splits = [0, 0.7, 0.90, 0.95]
        bins = [a + (b - a) * split for split in splits]
        bin_ids = np.digitize(scores, bins, right=False)
        grouped = [[] for _ in range(batch_size - 1)]
        for idx, b in zip(indices, bin_ids):
            grouped[b - 1].append(idx)

        sampled_indices = []
        for i in range(batch_size - 1):
            if i < batch_size - 2:
                if len(grouped[i]) > 0:
                    sampled_indices.extend(random.sample(grouped[i], 1))
                else:
                    sampled_indices.extend(random.sample(indices, 1))
            else:
                if len(grouped[i]) > 1:
                    sampled_indices.extend(random.sample(grouped[i], 2))
                else:
                    sampled_indices.extend(random.sample(indices, 2))
        return sampled_indices

    # %%
    def sample_random_batch(embeddings, personas, dot_products_list, batch_size=5):
        persona_idx = random.randint(0, len(embeddings) - 1)
        persona_embedding = embeddings[persona_idx]
        persona_text = personas[persona_idx]

        scores = dot_products_list[persona_idx]

        image_indices = sample_indices(scores, batch_size)

        return persona_embedding, persona_text, image_indices

    # %%
    sampled_persona_embeddings, sampled_personas, sampled_image_indices = [], [], []

    for i in tqdm(range(num_images)):
        sampled_persona_embedding, sampled_persona, sampled_image_index = (
            sample_random_batch(embeddings, personas, scores)
        )
        sampled_persona_embeddings.append(sampled_persona_embedding)
        sampled_personas.append(sampled_persona)
        sampled_image_indices.append(sampled_image_index)

    # %%

    def check_persona_images_fit(persona_text: str, pil_images: list) -> list:
        """
        Accepts a persona text and a list of PIL images.
        Returns a list of response texts (one per image).
        """

        model = GenerativeModel("gemini-2.0-flash")

        from google.api_core.exceptions import InternalServerError
        import time

        def safe_generate(prompt, image_objs, retries=6, delay=3):
            for attempt in range(retries):
                try:
                    return model.generate_content([prompt] + image_objs)
                except InternalServerError as e:
                    print(f"[Retry {attempt+1}/{retries}] Gemini 500 error: {e}")
                    time.sleep(delay * (2**attempt))  # exponential backoff
                except Exception as e:
                    print(f"[Retry {attempt+1}/{retries} Unexpected error: {e}")
                    time.sleep(delay * (2**attempt))  # exponential backoff

            return None

        prompt = f"""
        You are evaluating how well each product in the images matches a given customer persona.

        Persona:
        {persona_text}

        You will be shown multiple product images.

        Please rank the images from best fit to worst fit for this persona.

        Reply ONLY with a comma-separated list of image numbers (starting at 1 for the first image) ordered from best fit to worst fit.

        For example, if image 3 fits best, then image 1, then image 2, reply:
        3,1,2

        Do NOT include any explanation or additional text.
        """
        try:
            # Save all images to temporary files, load into Image objects
            image_objs = []
            temp_files = []
            try:
                for pil_image in pil_images:
                    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    pil_image.save(tmp.name)
                    temp_files.append(tmp.name)
                    image_obj = VertexImage.load_from_file(tmp.name)
                    image_objs.append(image_obj)
                    tmp.close()

                # Generate response with prompt + all images
                response = safe_generate(prompt, image_objs)

                # You might get a single combined response — if you want per image,
                # you could send one call per image in a loop instead.

            finally:
                # Clean up temp files
                for f in temp_files:
                    try:
                        os.remove(f)
                    except Exception:
                        pass

            return response.text
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Error generating response"

    # %%
    def evaluate_pair(n):
        return check_persona_images_fit(
            sampled_personas[n], [images[i] for i in sampled_image_indices[n]]
        )

    # Number of tasks
    num_items = len(sampled_image_indices)
    # Run with progress bar and error handling
    with tqdm_joblib(desc="Scoring persona-image pairs", total=num_items):
        scores = Parallel(n_jobs=8, backend="threading")(
            delayed(evaluate_pair)(n) for n in range(num_items)
        )

    # %%
    import re

    def parse_ranking(text):
        return list(map(int, re.findall(r"\d+", text)))

    # Build filtered lists
    clean_scores = []
    clean_image_indices = []
    clean_personas = []
    clean_embeddings = []

    for score, sampled_persona, sampled_image_index, sampled_embedding in zip(
        scores, sampled_personas, sampled_image_indices, sampled_persona_embeddings
    ):
        ranking = parse_ranking(score)
        if len(ranking) == 5:
            clean_scores.append(ranking)
            clean_image_indices.append(sampled_image_index)
            clean_personas.append(sampled_persona)
            clean_embeddings.append(sampled_embedding)

    print(f"Kept {len(clean_scores)} out of {len(scores)} responses.")

    # %%
    with open(
        f"{image_set}+{text_set}/{embedding_mode}/drift_data/drift_clean_scores{step_n+1}.pkl",
        "wb",
    ) as f:
        pickle.dump(clean_scores, f)
    with open(
        f"{image_set}+{text_set}/{embedding_mode}/drift_data/drift_clean_image_indices{step_n+1}.pkl",
        "wb",
    ) as f:
        torch.save(clean_image_indices, f)
    with open(
        f"{image_set}+{text_set}/{embedding_mode}/drift_data/drift_clean_personas{step_n+1}.pkl",
        "wb",
    ) as f:
        pickle.dump(clean_personas, f)
    with open(
        f"{image_set}+{text_set}/{embedding_mode}/drift_data/drift_clean_{embedding_mode}_embeddings{step_n+1}.pkl",
        "wb",
    ) as f:
        torch.save(clean_embeddings, f)


# %%

import os

# %%

self_sample_first_step = True

if __name__ == "__main__":

    if self_sample_first_step:

        if os.path.exists(
            f"{image_set}+{text_set}/{embedding_mode}/drift_sorted_indices/sorted_indices{-1}.pkl"
        ):
            print(f"Skipping ranking step {-1} as sorted indices already exist.")
        else:
            print(f"Ranking images for step {-1}...")
            rank_images(step_n=-1)

        if os.path.exists(
            f"{image_set}+{text_set}/{embedding_mode}/drift_data/drift_clean_scores{0}.pkl"
        ):
            print(
                f"Skipping data preparation step {-1} as cleaned data already exists."
            )
        else:
            print(f"Preparing data using step {-1} model for step {0}...")
            data_preparation_cycle(step_n=-1, num_images=1000)

    for i in range(30):
        lr_base = 1e-6
        if os.path.exists(
            f"{image_set}+{text_set}/{embedding_mode}/drift_weights/model_weights{i}.pth"
        ):
            print(f"Skipping step {i} as model already exists.")
        else:
            print(f"Training step {i}...")
            train_step(
                num_images=5,
                step_n=i,
                split_test=1 / 21,
                lr=lr_base * (0.95**i),
                freeze=False,
                batch_size_=50,
                weighting=False,
                epochs=1,
                accumulation_steps=20,
            )

        if os.path.exists(
            f"{image_set}+{text_set}/{embedding_mode}/drift_sorted_indices/sorted_indices{i}.pkl"
        ):
            print(f"Skipping ranking step {i} as sorted indices already exist.")
        else:
            print(f"Ranking images for step {i}...")
            rank_images(step_n=i)

        if os.path.exists(
            f"{image_set}+{text_set}/{embedding_mode}/drift_data/drift_clean_scores{i+1}.pkl"
        ):
            print(f"Skipping data preparation step {i} as cleaned data already exists.")
        else:
            print(f"Preparing data using step {i} model for step {i+1}...")
            data_preparation_cycle(step_n=i, num_images=1050)
