import numpy as np
import torch
from google.cloud import bigquery
from recommenders.ikea.results.result_analysis import query_product_url
import pandas as pd
import matplotlib.pyplot as plt
import requests
from PIL import Image


def get_similarities(
    item_string,
    n,
    input_tokenizer,
    embeddings,
    include_self=True,
    only_include_imgs=False,
):
    """
    Compute all similarities to item/image with label
    item_string and return n closest/furthest items as
    well.

    Returns both similarities least to max, so close-sim: 0.5, 0.7, 1
    and furtherst-sim: -0.5, -0.7, -1
    """
    # Encode and embed item
    example_item_encod = input_tokenizer.stoi(item_string)
    example_item_embed = embeddings(torch.tensor(example_item_encod))

    # Define cosine similarity
    cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    all_similarities = []
    closest = np.zeros(n)
    closest_sim = np.zeros(n)
    furthest = np.zeros(n)
    furthest_sim = np.ones(n) * 2

    for item in range(len(input_tokenizer)):
        if not include_self:
            # Skip example_item (sim=1)
            if item == example_item_encod:
                continue
        item_embedding = embeddings(torch.tensor(item))
        similarity = cos_sim(example_item_embed, item_embedding)
        all_similarities.append(similarity)

        # Save only images if specified and jump to next interaction if not img
        if only_include_imgs:
            item_string = str(input_tokenizer.itos(item))
            if not ("-" in item_string):
                continue

        if similarity > closest_sim.min():
            closest[0] = item
            closest_sim[0] = similarity

            # Sort in again in ascending order
            sort_indices = np.argsort(closest_sim)
            closest = closest[sort_indices]
            closest_sim = closest_sim[sort_indices]

        elif similarity < furthest_sim.max():
            furthest[-1] = item
            furthest_sim[-1] = similarity

            # Sort in again in ascending order
            sort_indices = np.argsort(furthest_sim)
            furthest = furthest[sort_indices]
            furthest_sim = furthest_sim[sort_indices]

    # Flip for right ordering
    furthest = np.flip(furthest)
    furthest_sim = np.flip(furthest_sim)

    return all_similarities, (closest, closest_sim), (furthest, furthest_sim)


def get_urls_for_embedding_vis(
    item_list,
    similarities,
    num_urls,
    insp_img_dict,
    only_include_imgs=False,
    proj_name="PROJECT-NAME",
):
    """
    Get urls of each item/image inside the provided list of
    decoded ids. Either as BigQuery query for articles or from
    the dict for inspirational images.

    num_urls: Number of urls needed for the visualization. In case a
              product is not found, the next product is tried. len(item_list)
              should therefore be >> num_urls. Or in get_similarities option
              only_include_imgs is turned on.
    """
    client = bigquery.Client(project=proj_name)
    urls = []
    sims = []
    found = 0
    not_found = 0
    i = len(item_list) - 1

    if only_include_imgs:
        # Only check inspirational images
        while found < num_urls:
            # Check if not enough
            if i < 0:
                raise Exception(f"Not enough values to fill {num_urls} urls.")
            id = item_list[i]
            if id in insp_img_dict:
                urls.append(insp_img_dict[id])
                sims.append(similarities[i])
                found += 1
            elif "-" in id:
                print(f"Not found: {id}")
                not_found += 1
            i -= 1

    else:
        # Check images and products
        while found < num_urls:
            id = item_list[i]

            # Check if not enough
            if i >= (len(item_list) + 1):
                raise Exception(f"Not enough values to fill {num_urls} urls.")
            if id in insp_img_dict:
                urls.append(insp_img_dict[id])
                sims.append(similarities[i])
                found += 1
                i -= 1
            elif "-" in id:
                raise Exception(f"Image not found: {id}")
            else:
                url = query_product_url(product_id=id, client=client, country="se")
                if pd.isna(url):
                    # Product not found, jump to next one
                    not_found += 1
                    i -= 1
                else:
                    urls.append(url)
                    sims.append(similarities[i])
                    found += 1
                    i -= 1
    print(f"Skipped items: {not_found}")

    return urls, sims


def plot_closest_furthest(
    closest_urls,
    furthest_urls,
    closest_sim,
    furthest_sim,
    max_steps=10,
    figsize=(20, 6),
    only_images=True,
):
    """
    Plot max_steps of closest and furhtest items/inspirational images.
    """
    plt.figure(figsize=figsize)

    if max_steps == None:
        # Show all images
        length = len(closest_urls)
    else:
        # Only show max_steps imgs
        length = max_steps

    # Reverse closest
    closest_urls = np.flip(closest_urls)
    closest_sim = np.flip(closest_sim)

    for i, close in enumerate(zip(closest_urls, closest_sim)):
        # if i==length:
        #    break
        url, sim = close[0], close[1]
        response = requests.get(url, stream=True)
        img = Image.open(response.raw)

        plt.subplot(2, length, i + 1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        label = f"Sim: {sim:.3f}"
        plt.title(label)

        if i == 9:
            plt.gca().spines["top"].set_color("limegreen")
            plt.gca().spines["right"].set_color("limegreen")
            plt.gca().spines["left"].set_color("limegreen")
            plt.gca().spines["bottom"].set_color("limegreen")
            plt.gca().spines["top"].set_linewidth(4)
            plt.gca().spines["right"].set_linewidth(4)
            plt.gca().spines["left"].set_linewidth(4)
            plt.gca().spines["bottom"].set_linewidth(4)

    if only_images:
        title_str = "Images"
    else:
        title_str = "Products/Images"

    plt.text(
        0.5,
        0.925,
        f"{length} Closest {title_str}",
        ha="center",
        fontsize=14,
        transform=plt.gcf().transFigure,
        weight="bold",
    )

    # Add subtitle to the second row of plots
    plt.text(
        0.5,
        0.5,
        f"{length} Furthest {title_str}",
        ha="center",
        fontsize=14,
        transform=plt.gcf().transFigure,
        weight="bold",
    )

    # Furthest
    for i, furth in enumerate(zip(furthest_urls, furthest_sim)):
        url, sim = furth[0], furth[1]
        response = requests.get(url, stream=True)
        img = Image.open(response.raw)

        plt.subplot(2, length, i + length + 1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        label = f"Sim: {sim:.3f}"
        plt.title(label)

    plt.subplots_adjust(wspace=0.25)
    plt.tight_layout()
