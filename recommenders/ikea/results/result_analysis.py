import pandas as pd
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import bigquery
import torch
import numpy as np


def get_row_realtion_mask(df):
    """
    Get boolen mask from replay buffer dataframe that
    marks rows that have one or more direct partners
    before or after. So where the next state of the row
    before matches the state of the next row.
    """
    mask = df.state == df.next_state.shift(1)
    return mask


def get_consecutive_mask(mask, min_length=2):
    """
    Enter a boolean mask and get new mask where all
    True blocks are eliminated that are shorter than
    min_length.
    """
    if type(mask) == pd.core.series.Series:
        mask = mask.to_list()

    new_mask = []
    count = 0
    for i in range(len(mask)):
        if mask[i]:
            count += 1
        else:
            if (count + 1) < min_length:
                new_mask += [False] * count
            else:
                new_mask.pop()
                new_mask += [True] * (count + 1)
            new_mask.append(False)
            count = 0

    if (count + 1) < min_length:
        new_mask += [False] * count
    else:
        new_mask.pop()
        new_mask += [True] * (count + 1)

    assert len(new_mask) == len(mask)

    return new_mask


def get_related_rows(df, min_length=2, min_state_len=1):
    """
    Returns the filtered dataframe containing related rows,
    so rows that are in a trajectory of size min_length.
    """
    # Remove rows with states shorter than min_state_len
    df = df[df.true_state_len > min_state_len]

    # Get mask
    mask = get_row_realtion_mask(df)
    new_mask = get_consecutive_mask(mask=mask, min_length=min_length)

    # Filter for related rows
    filtered_df = df[new_mask]

    return filtered_df


def query_product_url(product_id, client, country="se"):
    """
    Query product url from specific product id. Returns the url
    to the country website. Sweden is default.
    """
    query = f"""
    QUERY AGAINST THE IKEA DATABASE
    """
    # Get query result
    query_job = client.query(query)

    # Check if result is empty
    if len(list(query_job.result())) == 0:
        product_url = np.nan
        print(f"Not found: {product_id}")

    # Get url for specified country
    for row in query_job:
        if row["COUNTRY_CODE"] == "se":
            product_url = row["MAIN_IMAGE_URL"]
            break
        else:
            product_url = row["MAIN_IMAGE_URL"]

    return product_url


def get_state_urls(state, insp_img_dict, proj_name="PROJECT-NAME"):
    """
    Get urls of each action inside the state. Either as
    BigQuery query for articles or from the dict for
    inspirational images.
    """
    client = bigquery.Client(project=proj_name)
    urls = []
    for id in state:
        if id in insp_img_dict:
            urls.append(insp_img_dict[id])
        elif "-" in id:
            raise Exception(f"Image not found: {id}")
        else:
            try:
                url = query_product_url(product_id=id, client=client, country="se")
                urls.append(url)
            except:
                print("<pad> added.")
                urls.append("<pad>")
    return urls


def predict_and_get_urls(
    buffer_row, model, insp_img_dict, input_tokenizer, output_tokenizer, topk=12, head=0
):
    state = torch.tensor(buffer_row.state).unsqueeze(0)
    state_len = torch.tensor(buffer_row.true_state_len).unsqueeze(0)

    output = model(s=state, lengths=state_len)

    if type(output) == tuple:
        output = output[head]

    # Compute softmax probabilities
    output = torch.nn.functional.softmax(output, dim=1)

    # Get topk predictions and probabilities
    prob, pred = output.topk(k=topk)
    pred = pred.squeeze().tolist()
    prob = prob.squeeze().tolist()

    # Get urls
    decoded_state = [input_tokenizer.itos(idx) for idx in buffer_row.state]
    real_action_url = insp_img_dict[output_tokenizer.itos(buffer_row.action)]
    state_urls = get_state_urls(state=decoded_state, insp_img_dict=insp_img_dict)

    return state_urls, real_action_url, pred, prob


def plot_stream_and_predictions(
    state_urls,
    real_action_url,
    predictions,
    probabilities,
    insp_img_dict,
    output_tokenizer,
    name,
    max_steps=None,
    figsize=(22, 6),
    fake_target=False,
):
    """
    Plot max_steps of states_urls plus the real action that followed
    in first row and top-(max_steps+1)-predictions in second row.
    """
    plt.figure(figsize=figsize)

    if max_steps == None:
        # Show all steps
        length = len(state_urls) + 1
    else:
        # Only show max_steps imgs (+1 for real)
        length = max_steps + 1

    for i, url in enumerate(state_urls[-(length - 1) :]):
        img_url = url
        response = requests.get(img_url, stream=True)
        img = Image.open(response.raw)

        plt.subplot(2, length, i + 1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        if len(state_urls) >= length:
            t = length - i - 2
        else:
            t = len(state_urls) - i - 1
        label = f"-{t}" if t != 0 else ""
        plt.title(f"$t{label}$")

    if fake_target:
        img = Image.open("./fake_target.png")
    else:
        response = requests.get(real_action_url, stream=True)
        img = Image.open(response.raw)
    plt.subplot(2, length, length)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.title("True Next Image")
    plt.gca().spines["top"].set_color("limegreen")
    plt.gca().spines["right"].set_color("limegreen")
    plt.gca().spines["left"].set_color("limegreen")
    plt.gca().spines["bottom"].set_color("limegreen")
    plt.gca().spines["top"].set_linewidth(4)
    plt.gca().spines["right"].set_linewidth(4)
    plt.gca().spines["left"].set_linewidth(4)
    plt.gca().spines["bottom"].set_linewidth(4)

    plt.text(
        0.5,
        0.92,
        "Real User Sequence",
        ha="center",
        fontsize=14,
        transform=plt.gcf().transFigure,
        weight="bold",
    )

    # Add subtitle to the second row of plots
    plt.text(
        0.5,
        0.5,
        f"{name} - Top {length} Predictions",
        ha="center",
        fontsize=14,
        transform=plt.gcf().transFigure,
        weight="bold",
    )

    # Predicitons
    for i, pred in enumerate(
        zip(predictions[:length][::-1], probabilities[:length][::-1])
    ):
        img, p = pred[0], pred[1]
        pred_url = insp_img_dict[output_tokenizer.itos(img)]
        response = requests.get(pred_url, stream=True)
        img = Image.open(response.raw)
        plt.subplot(2, length, i + length + 1)
        plt.imshow(img)
        plt.title(f"{p*100:.2f}%")
        plt.xticks([])
        plt.yticks([])
    plt.subplots_adjust(wspace=0.25)
    plt.tight_layout()


def predict_and_plot_state(
    buffer_row,
    model,
    insp_img_dict,
    input_tokenizer,
    output_tokenizer,
    name,
    head=0,
    max_steps=None,
    figsize=(20, 6),
    fake_target=False,
):
    state_urls, real_action_url, pred, prob = predict_and_get_urls(
        buffer_row=buffer_row,
        model=model,
        insp_img_dict=insp_img_dict,
        input_tokenizer=input_tokenizer,
        output_tokenizer=output_tokenizer,
        head=head,
        topk=max_steps + 1 if max_steps != None else len(state_urls) + 1,
    )

    plot_stream_and_predictions(
        state_urls=state_urls,
        real_action_url=real_action_url,
        predictions=pred,
        probabilities=prob,
        insp_img_dict=insp_img_dict,
        output_tokenizer=output_tokenizer,
        max_steps=max_steps,
        figsize=figsize,
        name=name,
        fake_target=fake_target,
    )


def find_working_multiple_clicks(
    data, model, input_tokenizer, head=0, topk=12, min_imgs=4
):
    good_idx = []
    for i in range(len(data)):
        buffer_row = data.iloc[i]

        og_id = [1 for item in buffer_row.state if ("-" in input_tokenizer.itos(item))]

        if len(og_id) > min_imgs:
            state = torch.tensor(buffer_row.state).unsqueeze(0)
            state_len = torch.tensor(buffer_row.true_state_len).unsqueeze(0)

            output = model(s=state, lengths=state_len)

            if type(output) == tuple:
                output = output[head]

            # Compute softmax probabilities
            output = torch.nn.functional.softmax(output, dim=1)

            # Get topk predictions and probabilities
            prob, pred = output.topk(k=topk)
            pred = pred.squeeze().tolist()
            prob = prob.squeeze().tolist()

            if buffer_row.action in pred:
                good_idx.append(buffer_row.name)
                # print(f"Worked for index: {buffer_row.name}\n")
    return good_idx


def find_working_example(data, model, head=0, topk=12):
    good_idx = []
    for i in range(len(data)):
        buffer_row = data.iloc[i]
        state = torch.tensor(buffer_row.state).unsqueeze(0)
        state_len = torch.tensor(buffer_row.true_state_len).unsqueeze(0)

        output = model(s=state, lengths=state_len)

        if type(output) == tuple:
            output = output[head]

        # Compute softmax probabilities
        output = torch.nn.functional.softmax(output, dim=1)

        # Get topk predictions and probabilities
        prob, pred = output.topk(k=topk)
        pred = pred.squeeze().tolist()
        prob = prob.squeeze().tolist()

        if buffer_row.action in pred:
            good_idx.append(buffer_row.name)
            # print(f"Worked for index: {buffer_row.name}\n")
    return good_idx


def predict_and_plot_state_20(
    buffer_row,
    model,
    insp_img_dict,
    input_tokenizer,
    output_tokenizer,
    name,
    head=0,
    max_steps=None,
    figsize=(20, 6),
):
    state_urls, real_action_url, pred, prob = predict_and_get_urls(
        buffer_row=buffer_row,
        model=model,
        insp_img_dict=insp_img_dict,
        input_tokenizer=input_tokenizer,
        output_tokenizer=output_tokenizer,
        head=head,
        topk=max_steps + 1 if max_steps != None else len(state_urls) + 1,
    )

    plot_stream_and_predictions_20(
        state_urls=state_urls,
        real_action_url=real_action_url,
        predictions=pred,
        probabilities=prob,
        insp_img_dict=insp_img_dict,
        output_tokenizer=output_tokenizer,
        figsize=figsize,
        name=name,
    )


def plot_stream_and_predictions_20(
    state_urls,
    real_action_url,
    predictions,
    probabilities,
    insp_img_dict,
    output_tokenizer,
    name,
    figsize=(22, 6),
):
    """
    Plot max_steps of states_urls plus the real action that followed
    in first row and top-(max_steps+1)-predictions in second row.
    """
    plt.figure(figsize=figsize)

    length = 11
    t_count = 20

    pad_num = 0
    if np.nan in state_urls:
        pad_num = state_urls.count(np.nan)
        for nan in range(pad_num):
            plt.subplot(3, 11, nan + 1)
            img = Image.open("./padding_token.png")
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            t_count -= 1
            label = f"-{t_count}" if t_count != 0 else ""
            plt.title(f"$t{label}$")
        state_urls = state_urls[:-pad_num]

    for i in range(0, 11):
        url = state_urls[i]
        img_url = url
        response = requests.get(img_url, stream=True)
        img = Image.open(response.raw)

        plt.subplot(3, 11, i + pad_num + 1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        t_count -= 1
        label = f"-{t_count}" if t_count != 0 else ""
        plt.title(f"$t{label}$")
    for i in range(0, 9 - pad_num):
        url = state_urls[i + 11]
        img_url = url
        response = requests.get(img_url, stream=True)
        img = Image.open(response.raw)

        plt.subplot(3, 11, i + 1 + 11 + pad_num)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        t_count -= 1
        label = f"-{t_count}" if t_count != 0 else ""
        plt.title(f"$t{label}$")

    response = requests.get(real_action_url, stream=True)
    img = Image.open(response.raw)
    plt.subplot(3, 11, 22)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.title("True Next Image")
    plt.gca().spines["top"].set_color("limegreen")
    plt.gca().spines["right"].set_color("limegreen")
    plt.gca().spines["left"].set_color("limegreen")
    plt.gca().spines["bottom"].set_color("limegreen")
    plt.gca().spines["top"].set_linewidth(4)
    plt.gca().spines["right"].set_linewidth(4)
    plt.gca().spines["left"].set_linewidth(4)
    plt.gca().spines["bottom"].set_linewidth(4)

    plt.text(
        0.5,
        0.92,
        "Real User Sequence",
        ha="center",
        fontsize=14,
        transform=plt.gcf().transFigure,
        weight="bold",
    )

    # Add subtitle to the second row of plots
    plt.text(
        0.5,
        0.37,
        f"{name} - Top {length} Predictions",
        ha="center",
        fontsize=14,
        transform=plt.gcf().transFigure,
        weight="bold",
    )

    # Predicitons
    for i, pred in enumerate(
        zip(predictions[:length][::-1], probabilities[:length][::-1])
    ):
        img, p = pred[0], pred[1]
        pred_url = insp_img_dict[output_tokenizer.itos(img)]
        response = requests.get(pred_url, stream=True)
        img = Image.open(response.raw)
        plt.subplot(3, length, i + length + 1 + 11)
        plt.imshow(img)
        plt.title(f"{p*100:.2f}%")
        plt.xticks([])
        plt.yticks([])
    plt.subplots_adjust(wspace=0.25)
    plt.tight_layout()
